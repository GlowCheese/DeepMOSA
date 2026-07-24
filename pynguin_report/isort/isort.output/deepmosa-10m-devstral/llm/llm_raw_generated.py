####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_empty_input():
    assert _ensure_newline_before_comment([]) == []

def test_single_comment_line():
    assert _ensure_newline_before_comment(["# comment"]) == ["# comment"]

def test_single_non_comment_line():
    assert _ensure_newline_before_comment(["code"]) == ["code"]

def test_comment_after_non_comment():
    assert _ensure_newline_before_comment(["code", "# comment"]) == ["code", "", "# comment"]

def test_comment_after_empty_line():
    assert _ensure_newline_before_comment(["", "# comment"]) == ["", "# comment"]

def test_comment_after_comment():
    assert _ensure_newline_before_comment(["# comment1", "# comment2"]) == ["# comment1", "# comment2"]

def test_multiple_comments_after_non_comment():
    assert _ensure_newline_before_comment(["code", "# comment1", "# comment2"]) == ["code", "", "# comment1", "# comment2"]

def test_mixed_lines():
    assert _ensure_newline_before_comment(["code1", "# comment1", "code2", "# comment2"]) == ["code1", "", "# comment1", "code2", "", "# comment2"]

def test_no_newline_needed():
    assert _ensure_newline_before_comment(["# comment", "code"]) == ["# comment", "code"]

def test_none_line_handling():
    assert _ensure_newline_before_comment([None, "# comment"]) == [None, "", "# comment"]

def test_multiple_empty_lines_before_comment():
    assert _ensure_newline_before_comment(["", "", "# comment"]) == ["", "", "# comment"]


# LLM-generated content at query #2
#--------------------------

```python
def test_with_straight_imports_empty_straight_modules():
    parsed = parse.ParsedContent(
        imports={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, [], "straight", [], "import")
    assert result == []

def test_with_straight_imports_combine_straight_imports_no_as_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"module1": [], "module2": []}},
        categorized_comments={"above": {"straight": {"module1": ["# comment1"], "module2": ["# comment2"]}}, "straight": {"module1": ["# inline1"], "module2": ["# inline2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["module1", "module2"], "straight", [], "import")
    assert result == ["# comment1", "# comment2", "import module1, module2  # inline1 inline2"]

def test_with_straight_imports_combine_straight_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"module1": [], "module2": []}},
        categorized_comments={"above": {"straight": {"module1": ["# comment1"], "module2": ["# comment2"]}}, "straight": {"module1": ["# inline1"], "module2": ["# inline2"]}},
        as_map={"straight": {"module1": ["alias1"]}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["module1", "module2"], "straight", [], "import")
    assert result == ["# comment1", "import module1 as alias1  # inline1", "# comment2", "import module2  # inline2"]

def test_with_straight_imports_no_combine_straight_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"module1": [], "module2": []}},
        categorized_comments={"above": {"straight": {"module1": ["# comment1"], "module2": ["# comment2"]}}, "straight": {"module1": ["# inline1"], "module2": ["# inline2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["module1", "module2"], "straight", [], "import")
    assert result == ["# comment1", "import module1  # inline1", "# comment2", "import module2  # inline2"]

def test_with_straight_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"module1": [], "module2": []}},
        categorized_comments={"above": {"straight": {"module1": ["# comment1"], "module2": ["# comment2"]}}, "straight": {"module1": ["# inline1"], "module2": ["# inline2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["module1", "module2"], "straight", ["module1"], "import")
    assert result == ["# comment2", "import module2  # inline2"]

def test_with_straight_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"straight": {"module1": [], "module2": []}},
        categorized_comments={"above": {"straight": {"module1": ["# comment1"], "module2": ["# comment2"]}}, "straight": {"module1": ["# inline1"], "module2": ["# inline2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False, ignore_comments=True)
    result = _with_straight_imports(parsed, config, ["module1", "module2"], "straight", [], "import")
    assert result == ["import module1", "import module2"]


# LLM-generated content at query #3
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
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
            "straight": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2  # comment1; comment2"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", ["module.import1"], "import")
    assert result == ["from module import import2"]

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {"module.import1": ["alias1"], "module.import2": ["alias2"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1 as alias1, import2 as alias2"]

def test_with_from_imports_with_star():
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
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import *  # star comment"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1", "from module import import2"]

def test_with_from_imports_combine_as_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {"module.import1": ["alias1"], "module.import2": ["alias2"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1 as alias1, import2 as alias2"]

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={
            "from": {"module": ["comment1", "comment2"]},
            "above": {"from": {}},
            "nested": {},
            "straight": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2"]


# LLM-generated content at query #4
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
        place_imports={},
        import_placements={},
        sections=[],
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
        place_imports={},
        import_placements={},
        sections=[],
    )
    assert sorted_imports(parsed) == "print('hello')\n"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    assert sorted_imports(parsed) == "import os\n\n"

def test_sorted_imports_single_from_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    assert sorted_imports(parsed) == "from os import path\n\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    assert sorted_imports(parsed) == "import os, sys\n\n"

def test_sorted_imports_multiple_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}, "sys": {"argv": set()}}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    assert sorted_imports(parsed) == "from os import path\nfrom sys import argv\n\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# comment above"]}}, "straight": {"os": ["# inline comment"]}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    assert sorted_imports(parsed) == "# comment above\nimport os  # inline comment\n\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["ospath"]}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    assert sorted_imports(parsed) == "import os as ospath\n\n"

def test_sorted_imports_with_force_sort_within_sections():
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"sys": set(), "os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    assert sorted_imports(parsed, config) == "import os, sys\n\n"

def test_sorted_imports_with_no_sections():
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}, "THIRDPARTY": {"straight": {"django": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB", "THIRDPARTY"],
    )
    assert sorted_imports(parsed, config) == "import django, os\n\n"

def test_sorted_imports_with_remove_imports():
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    assert sorted_imports(parsed, config) == "import sys\n\n"

def test_sorted_imports_with_import_headings():
    config = Config(import_headings={"stdlib": "Standard Library"})
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    assert sorted_imports(parsed, config) == "# Standard Library\nimport os\n\n"


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
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["comment"]}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os  # comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": {"ospath"}}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os as ospath\n"

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

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\n"

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
    config = Config(import_headings={"stdlib": "Standard Library"})
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library\nimport os\n"


# LLM-generated content at query #6
#--------------------------

```python
def test_with_star_comments_when_star_comment_exists():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"test_module": {"*": "star comment"}}}
    module = "test_module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2", "star comment"]

def test_with_star_comments_when_star_comment_does_not_exist():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"test_module": {}}}
    module = "test_module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2"]

def test_with_star_comments_when_module_does_not_exist():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {}}
    module = "non_existent_module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2"]


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_evaluates_to_true():
    config = Config()
    config.no_inline_sort = False
    config.force_single_line = False
    config.only_sections = False
    config.single_line_exclusions = []

    assert not (
        config.no_inline_sort
        or (config.force_single_line and "module" not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #8
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
    assert sorted_imports(parsed, config) == "print('hello')\n"

def test_sorted_imports_with_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict": set()}},
            }
        },
        categorized_comments={},
        as_map={},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            }
        },
        categorized_comments={},
        as_map={},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(combine_straight_imports=True)
    assert sorted_imports(parsed, config) == "import os, sys\n\nprint('hello')\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict": set()}},
            }
        },
        categorized_comments={},
        as_map={},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(force_sort_within_sections=True)
    assert sorted_imports(parsed, config) == "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            }
        },
        categorized_comments={},
        as_map={},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    assert sorted_imports(parsed, config) == "# Third Party Imports\nimport os\nimport sys\n\nprint('hello')\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            }
        },
        categorized_comments={},
        as_map={},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(remove_imports=["os"])
    assert sorted_imports(parsed, config) == "import sys\n\nprint('hello')\n"


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_evaluates_to_false():
    config = Config(
        no_inline_sort=True,
        force_single_line=False,
        single_line_exclusions=[],
        only_sections=False
    )
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and "module" not in config.single_line_exclusions)
    )


# LLM-generated content at query #11
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
    config = DEFAULT_CONFIG
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
    config = DEFAULT_CONFIG
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
    config = DEFAULT_CONFIG
    assert sorted_imports(parsed, config) == "import os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={
            "above": {"straight": {"os": ["# OS module"]}},
            "straight": {"os": ["# For path operations"]},
        },
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = DEFAULT_CONFIG
    assert sorted_imports(parsed, config) == "# OS module\nimport os  # For path operations\n"

def test_sorted_imports_combine_straight_imports():
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

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["path"]}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = DEFAULT_CONFIG
    assert sorted_imports(parsed, config) == "import os as path\n"

def test_sorted_imports_remove_imports():
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

def test_sorted_imports_no_sections():
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
    assert sorted_imports(parsed, config) == "import django\nimport os\n"

def test_sorted_imports_force_sort_within_sections():
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
    assert sorted_imports(parsed, config) == "import os\nimport sys\n"

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

def test_sorted_imports_with_import_footers():
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
    config = Config(import_footers={"stdlib": "End of Standard Library"})
    assert sorted_imports(parsed, config) == "import os\n\n# End of Standard Library\n"

def test_sorted_imports_ensure_newline_before_comments():
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
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "


# LLM-generated content at query #12
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
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"


# LLM-generated content at query #13
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
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
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

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}, "from": {}}},
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

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {}, "from": {"os": ["path"]}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["operating_system"]}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os as operating_system\n"

def test_sorted_imports_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}, "from": {}}},
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

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# Comment above"]}}, "straight": {"os": ["# Inline comment"]}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "# Comment above\nimport os  # Inline comment\n"

def test_sorted_imports_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}, "from": {}}},
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

def test_sorted_imports_no_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}, "THIRDPARTY": {"straight": {"django": ["django"]}, "from": {}}},
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

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
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

def test_sorted_imports_with_import_footers():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(import_footers={"stdlib": "End of Standard Library Imports"})
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n# End of Standard Library Imports\n"

def test_sorted_imports_ensure_newline_before_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\n"

def test_sorted_imports_with_formatting_function():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(formatting_function=lambda x, y, z: x.upper())
    result = sorted_imports(parsed, config)
    assert result == "IMPORT OS\n"

def test_sorted_imports_with_lines


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) or config.only_sections


# LLM-generated content at query #15
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
        line_separator="\n",
        sections=[],
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == ""

def test_sorted_imports_with_no_imports():
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
    assert sorted_imports(parsed, config) == "print('hello')"

def test_sorted_imports_with_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os\n"

def test_sorted_imports_with_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os\nimport sys\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "from os import path\n"

def test_sorted_imports_with_combined_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(combine_straight_imports=True)
    assert sorted_imports(parsed, config) == "import os, sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# Comment above"]}, "from": {}}, "straight": {"os": ["# Inline comment"]}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "# Comment above\nimport os  # Inline comment\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    assert sorted_imports(parsed, config) == "import sys\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {"os": ["ospath"]}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os\nimport os as ospath\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"zlib": set(), "os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    assert sorted_imports(parsed, config) == "import os\nimport zlib\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"stdlib": "Standard Library"})
    assert sorted_imports(parsed, config) == "# Standard Library\nimport os\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}, "THIRDPARTY": {"straight": {"django": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        sections=["STDLIB", "THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=1)
    assert sorted_imports(parsed, config) == "import os\n\nimport django\n"

def test_sorted_imports_with_lines_after_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight":


# LLM-generated content at query #16
#--------------------------

```python
def test_with_straight_imports_empty_modules():
    parsed = parse.ParsedContent(
        imports={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, [], "straight", [], "import") == []

def test_with_straight_imports_combine_no_as_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["os"], "os": []}},
        categorized_comments={
            "above": {"straight": {"sys": ["# sys comment"]}},
            "straight": {"sys": ["# inline sys"], "os": ["# inline os"]},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "straight", [], "import")
    assert result == ["# sys comment", "import sys, os  # # inline sys # inline os"]

def test_with_straight_imports_combine_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["os"], "os": []}},
        categorized_comments={
            "above": {"straight": {"sys": ["# sys comment"]}},
            "straight": {"sys": ["# inline sys"], "os": ["# inline os"]},
        },
        as_map={"straight": {"sys": ["sys_alias"]}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "straight", [], "import")
    assert result == [
        "# sys comment",
        "import sys as sys_alias  # # inline sys",
        "import os  # # inline os",
    ]

def test_with_straight_imports_no_combine():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["os"], "os": []}},
        categorized_comments={
            "above": {"straight": {"sys": ["# sys comment"]}},
            "straight": {"sys": ["# inline sys"], "os": ["# inline os"]},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "straight", [], "import")
    assert result == [
        "# sys comment",
        "import sys  # # inline sys",
        "import os  # # inline os",
    ]

def test_with_straight_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["os"], "os": []}},
        categorized_comments={
            "above": {"straight": {"sys": ["# sys comment"]}},
            "straight": {"sys": ["# inline sys"], "os": ["# inline os"]},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "straight", ["sys"], "import")
    assert result == ["import os  # # inline os"]

def test_with_straight_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["os"], "os": []}},
        categorized_comments={
            "above": {"straight": {"sys": ["# sys comment"]}},
            "straight": {"sys": ["# inline sys"], "os": ["# inline os"]},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False, ignore_comments=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "straight", [], "import")
    assert result == ["# sys comment", "import sys", "import os"]

def test_with_straight_imports_custom_comment_prefix():
    parsed = parse.ParsedContent(
        imports={"straight": {"sys": ["os"], "os": []}},
        categorized_comments={
            "above": {"straight": {"sys": ["# sys comment"]}},
            "straight": {"sys": ["# inline sys"], "os": ["# inline os"]},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False, comment_prefix=" # ")
    result = _with_straight_imports(parsed, config, ["sys", "os"], "straight", [], "import")
    assert result == [
        "# sys comment",
        "import sys # # inline sys",
        "import os # # inline os",
    ]


# LLM-generated content at query #17
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
    assert result == "\nimport os"

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
    assert result == "\nimport os\nimport sys"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["# comment"]}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(ignore_comments=False)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os  # comment"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["path"]}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nimport os as path"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nfrom os import path"

def test_sorted_imports_with_sections():
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
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nfrom __future__ import absolute_import\n\nimport os"

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
    assert result == "\nimport sys"

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
    assert result == "\nimport os, sys"

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
    assert result == "\nimport os\nimport sys"

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
    result = sorted_imports(parsed, config)
    assert result == "\n# Standard Library\nimport os"


# LLM-generated content at query #18
#--------------------------

```python
def test_with_straight_imports_empty_modules():
    parsed = parse.ParsedContent(
        imports={},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, [], "standard", [], "import")
    assert result == []

def test_with_straight_imports_combine_without_as():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"sys": ["comment1"], "os": ["comment2"]},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == ["import sys, os  # comment1 comment2"]

def test_with_straight_imports_combine_with_as():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"sys": ["comment1"], "os": ["comment2"]},
        },
        as_map={"straight": {"sys": ["s"], "os": ["o"]}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == [
        "import sys as s",
        "import os as o",
    ]

def test_with_straight_imports_no_combine():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={
            "above": {"straight": {"sys": ["above comment"]}},
            "straight": {"sys": ["inline comment"], "os": ["inline comment"]},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == [
        "above comment",
        "import sys  # inline comment",
        "import os  # inline comment",
    ]

def test_with_straight_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"sys": ["comment1"], "os": ["comment2"]},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", ["sys"], "import")
    assert result == ["import os  # comment2"]

def test_with_straight_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"sys": ["comment1"], "os": ["comment2"]},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False, ignore_comments=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == ["import sys", "import os"]

def test_with_straight_imports_custom_comment_prefix():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"sys": ["comment1"], "os": ["comment2"]},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False, comment_prefix=" # ")
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == ["import sys # comment1", "import os # comment2"]


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_1():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = ["module1", "module2"]
    section = "section"
    remove_imports = []
    import_type = "import"

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)

    assert isinstance(result, list)


# LLM-generated content at query #20
#--------------------------

```python
def test__with_from_imports_basic_case():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": []}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]

def test__with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["# comment"]}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # comment"]

def test__with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": []}},
        as_map={"from": {"os.path": [], "os.sys": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", ["os.sys"], "import")
    assert result == ["from os import path"]

def test__with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": []}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path as ospath"]

def test__with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"nested": {"os": {"*": ["# star comment"]}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test__with_from_imports_with_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": []}},
        as_map={"from": {"os.path": [], "os.sys": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path", "from os import sys"]

def test__with_from_imports_with_combine_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": []}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path as ospath"]

def test__with_from_imports_with_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["# comment"]}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]


# LLM-generated content at query #21
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        imports={},
        as_map={},
        categorized_comments={},
        place_imports={},
        import_placements={},
        sections=[],
        sections_to_ignore=[],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

def test_sorted_imports_with_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            }
        },
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        sections_to_ignore=[],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys\n\nprint('hello')"

def test_sorted_imports_with_combined_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            }
        },
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        sections_to_ignore=[],
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os, sys\n\nprint('hello')"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": {"path": set()}},
            }
        },
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        sections_to_ignore=[],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nfrom os import path\n\nprint('hello')"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        as_map={"straight": {}},
        categorized_comments={
            "above": {"straight": {"os": ["# Comment above os"]}},
            "straight": {"os": ["# Comment inline os"]},
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        sections_to_ignore=[],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n# Comment above os\nimport os  # Comment inline os\n\nprint('hello')"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            }
        },
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        sections_to_ignore=[],
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "\nimport sys\n\nprint('hello')"

def test_sorted_imports_with_no_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=3,
        imports={
            "FUTURE": {
                "straight": {"__future__": set()},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["FUTURE", "THIRDPARTY"],
        sections_to_ignore=[],
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "\nfrom __future__ import annotations\nimport os\n\nprint('hello')"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=3,
        imports={
            "THIRDPARTY": {
                "straight": {"sys": set(), "os": set()},
                "from": {},
            }
        },
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        sections_to_ignore=[],
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys\n\nprint('hello')"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        sections_to_ignore=[],
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "\n# Third Party Imports\nimport os\n\nprint('hello')"

def test_sorted_imports_with_ensure_newline_before_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=3,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        sections_to_ignore=[],
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\n\nprint('hello')"


# LLM-generated content at query #22
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = ["os"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    parsed.imports = {
        "STDLIB": {
            "from": {
                "os": ["path", "sys"]
            }
        }
    }
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {"from": {}, "above": {"from": {}}, "nested": {}}

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == ["from os import path, sys"]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = ["os"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    parsed.imports = {
        "STDLIB": {
            "from": {
                "os": ["path"]
            }
        }
    }
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {
        "from": {"os": ["# comment"]},
        "above": {"from": {}},
        "nested": {}
    }

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == ["# comment", "from os import path"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = ["os"]
    section = "STDLIB"
    remove_imports = ["os.path"]
    import_type = "import"

    parsed.imports = {
        "STDLIB": {
            "from": {
                "os": ["path", "sys"]
            }
        }
    }
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {"from": {}, "above": {"from": {}}, "nested": {}}

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == ["from os import sys"]

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = ["os"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    parsed.imports = {
        "STDLIB": {
            "from": {
                "os": ["path"]
            }
        }
    }
    parsed.as_map = {"from": {"os.path": ["path as ospath"]}}
    parsed.categorized_comments = {"from": {}, "above": {"from": {}}, "nested": {}}

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == ["from os import path as ospath"]

def test_with_from_imports_with_star():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = ["os"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    parsed.imports = {
        "STDLIB": {
            "from": {
                "os": ["*"]
            }
        }
    }
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {
        "from": {},
        "above": {"from": {}},
        "nested": {"os": {"*": "# star comment"}}
    }

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent()
    config = Config(force_single_line=True)
    from_modules = ["os"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    parsed.imports = {
        "STDLIB": {
            "from": {
                "os": ["path", "sys"]
            }
        }
    }
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {"from": {}, "above": {"from": {}}, "nested": {}}

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == ["from os import path", "from os import sys"]

def test_with_from_imports_combine_as_imports():
    parsed = parse.ParsedContent()
    config = Config(combine_as_imports=True)
    from_modules = ["os"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    parsed.imports = {
        "STDLIB": {
            "from": {
                "os": ["path"]
            }
        }
    }
    parsed.as_map = {"from": {"os.path": ["path as ospath"]}}
    parsed.categorized_comments = {"from": {}, "above": {"from": {}}, "nested": {}}

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == ["from os import path as ospath"]

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent()
    config = Config(ignore_comments=True)
    from_modules = ["os"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    parsed.imports = {
        "STDLIB": {
            "from": {
                "os": ["path"]
            }
        }
    }
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {
        "from": {"os": ["# comment"]},
        "above": {"from": {}},
        "nested": {}
    }

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == ["from os import path"]

def test_with_from_imports_no_inline_sort():
    parsed = parse.ParsedContent()
    config = Config(no_inline_sort=True)
    from_modules = ["os"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    parsed.imports = {
        "STDLIB": {
            "from": {
                "os": ["sys", "path"]
            }
        }
    }
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {"from": {}, "above": {"from": {}}, "nested": {}}

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == ["from os import sys, path"]

def test_with_from_imports_only_sections():
    parsed = parse.ParsedContent()
    config = Config(only_sections=True)
    from_modules = ["os"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    parsed.imports = {
        "STDLIB": {
            "from": {
                "os": ["path"]
            }
        }
    }
    parsed.as_map = {"from": {}}
    parsed.categorized_comments = {"from": {}, "above": {"from": {}}, "nested": {}}

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert result == ["from os import path"]


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #25
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
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

def test_sorted_imports_with_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"numpy": set(), "pandas": set()}}},
        categorized_comments={},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        original_line_count=0,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import numpy\nimport pandas\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"from": {"numpy": {"array": set()}}}},
        categorized_comments={},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        original_line_count=0,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from numpy import array\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"numpy": set()}}},
        categorized_comments={},
        as_map={"straight": {"numpy": ["np"]}},
        sections=["THIRDPARTY"],
        place_imports={},
        original_line_count=0,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import numpy as np\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"numpy": set()}}},
        categorized_comments={"straight": {"numpy": ["# comment"]}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        original_line_count=0,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import numpy  # comment\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"numpy": set(), "pandas": set()}}},
        categorized_comments={},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        original_line_count=0,
    )
    config = Config(remove_imports=["numpy"])
    result = sorted_imports(parsed, config)
    assert result == "import pandas\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"numpy": set(), "pandas": set()}}},
        categorized_comments={},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        original_line_count=0,
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import numpy, pandas\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"pandas": set(), "numpy": set()}}},
        categorized_comments={},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        original_line_count=0,
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import numpy\nimport pandas\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"numpy": set()}}},
        categorized_comments={},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        original_line_count=0,
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport numpy\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": {"__future__": set()}},
            "THIRDPARTY": {"straight": {"numpy": set()}},
        },
        categorized_comments={},
        as_map={"straight": {}},
        sections=["FUTURE", "THIRDPARTY"],
        place_imports={},
        original_line_count=0,
    )
    config = Config(lines_between_sections=1)
    result = sorted_imports(parsed, config)
    assert result == "import __future__\n\nimport numpy\n"

def test_sorted_imports_with_lines_after_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"numpy": set()}}},
        categorized_comments={},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        original_line_count=1,
    )
    config = Config(lines_after_imports=1)
    result = sorted_imports(parsed, config)
    assert result == "import numpy\n\nprint('hello')"


# LLM-generated content at query #26
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
    config = Config()
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

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"module.import1": ["alias1"], "module.import2": ["alias2"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1 as alias1, import2 as alias2"]

def test_with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["*", "import1"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {"module": {"*": ["star_comment"]}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import *  # star_comment", "from module import import1"]

def test_with_from_imports_with_nested_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"module": {"import1": ["nested_comment1"], "import2": ["nested_comment2"]}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1  # nested_comment1", "from module import import2  # nested_comment2"]

def test_with_from_imports_with_force_single_line():
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

def test_with_from_imports_with_combine_as_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"module.import1": ["alias1"], "module.import2": ["alias2"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1 as alias1, import2 as alias2"]

def test_with_from_imports_with_ignore_comments():
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
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2"]

def test_with_from_imports_with_comment_prefix():
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
    config = Config(comment_prefix="--")
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2  -- comment1; comment2"]


# LLM-generated content at query #27
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


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_evaluates_to_false():
    parsed = parse.ParsedContent()
    parsed.as_map = {"straight": []}
    config = Config()
    straight_modules = []
    section = "test"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == []


# LLM-generated content at query #29
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        line_separator="\n",
        import_index=-1,
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
        line_separator="\n",
        import_index=0,
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
        line_separator="\n",
        import_index=0,
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
        line_separator="\n",
        import_index=0,
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={
            "above": {"straight": {"os": ["# OS module"]}},
            "straight": {"os": ["# For path operations"]},
        },
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(ignore_comments=False)
    assert sorted_imports(parsed, config) == "# OS module\nimport os  # For path operations\n"

def test_sorted_imports_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
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

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["path"]}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os as path\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"from": {}}, "from": {"os": {}}},
        as_map={"from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "from os import path\n"

def test_sorted_imports_with_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={
            "FUTURE": {"straight": {"__future__": {"annotations"}}, "from": {}},
            "STDLIB": {"straight": {"os": set()}, "from": {}},
        },
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["FUTURE", "STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "from __future__ import annotations\n\nimport os\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"STDLIB": {"straight": {"sys": set(), "os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(force_sort_within_sections=True)
    assert sorted_imports(parsed, config) == "import os\nimport sys\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
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

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={
            "FUTURE": {"straight": {"__future__": {"annotations"}}, "from": {}},
            "STDLIB": {"straight": {"os": set()}, "from": {}},
        },
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["FUTURE", "STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(lines_between_sections=2)
    assert sorted_imports(parsed, config) == "from __future__ import annotations\n\n\n\nimport os\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
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

def test_sorted_imports_with_only_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={
            "FUTURE": {"straight": {"__future__": {"annotations"}}, "from": {}},
            "STDLIB": {"straight": {"os": set()}, "from": {}},
        },
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["FUTURE", "STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )


# LLM-generated content at query #30
#--------------------------

```python
def test_sorted_imports_predicate_false():
    parsed = parse.ParsedContent(import_index=0, lines_without_imports=[], line_separator='\n')
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == ""


# LLM-generated content at query #31
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
    config = Config()
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

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"module.import1": ["alias1"], "module.import2": ["alias2"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1 as alias1, import2 as alias2"]

def test_with_from_imports_with_star_comment():
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
    assert result == ["from module import *  # star comment", "from module import import1"]

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

def test_with_from_imports_ignore_comments():
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
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2"]


# LLM-generated content at query #32
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #33
#--------------------------

```python
def test_star_comment_is_none():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"module": {}}}
    module = "module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == comments


# LLM-generated content at query #34
#--------------------------

```python
def test_as_imports_predicate():
    parsed = parse.ParsedContent()
    parsed.as_map = {"straight": ["module1", "module2"]}
    straight_modules = ["module1"]
    assert any(module in parsed.as_map["straight"] for module in straight_modules)


# LLM-generated content at query #35
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ("# comment",)}},
        as_map={"from": {"os.path": ["os.path as osp"]}},
        line_separator="\n",
    )
    config = Config()
    result = _with_from_imports(
        parsed, config, ["os"], "STDLIB", [], "import"
    )
    assert result == ["from os import path  # comment", "from os import os.path as osp"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ("# comment",)}},
        as_map={"from": {}},
        line_separator="\n",
    )
    config = Config()
    result = _with_from_imports(
        parsed, config, ["os"], "STDLIB", ["os.sys"], "import"
    )
    assert result == ["from os import path  # comment"]

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"nested": {"os": {"*": "# star comment"}}},
        as_map={"from": {}},
        line_separator="\n",
    )
    config = Config(combine_star=True)
    result = _with_from_imports(
        parsed, config, ["os"], "STDLIB", [], "import"
    )
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ("# comment",)}},
        as_map={"from": {}},
        line_separator="\n",
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(
        parsed, config, ["os"], "STDLIB", [], "import"
    )
    assert result == ["from os import path  # comment", "from os import sys"]

def test_with_from_imports_combine_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ("# comment",)}},
        as_map={"from": {"os.path": ["os.path as osp"]}},
        line_separator="\n",
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(
        parsed, config, ["os"], "STDLIB", [], "import"
    )
    assert result == ["from os import path, os.path as osp  # comment"]

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ("# comment",)}},
        as_map={"from": {}},
        line_separator="\n",
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(
        parsed, config, ["os"], "STDLIB", [], "import"
    )
    assert result == ["from os import path"]

def test_with_from_imports_no_inline_sort():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["sys", "path"]}}},
        categorized_comments={"from": {"os": ("# comment",)}},
        as_map={"from": {}},
        line_separator="\n",
    )
    config = Config(no_inline_sort=True)
    result = _with_from_imports(
        parsed, config, ["os"], "STDLIB", [], "import"
    )
    assert result == ["from os import sys, path  # comment"]

def test_with_from_imports_only_sections():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ("# comment",)}},
        as_map={"from": {}},
        line_separator="\n",
    )
    config = Config(only_sections=True)
    result = _with_from_imports(
        parsed, config, ["os"], "STDLIB", [], "import"
    )
    assert result == ["from os import path  # comment"]


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) or config.only_sections


# LLM-generated content at query #37
#--------------------------

```python
def test_with_from_imports_basic_case():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ()}},
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
        categorized_comments={"from": {"os": ("comment",)}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False, comment_prefix="# ")
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # comment"]

def test_with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"nested": {"os": {"*": "star comment"}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False, comment_prefix="# ")
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

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
    assert result == ["from os import path", "from os import path as ospath"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", ["os.sys"], "import")
    assert result == ["from os import path"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path", "from os import sys"]

def test_with_from_imports_combine_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*", "path"]}}},
        categorized_comments={"nested": {"os": {"*": "star comment"}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True, ignore_comments=False, comment_prefix="# ")
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment", "from os import path"]

def test_with_from_imports_no_inline_sort():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["sys", "path"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(no_inline_sort=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import sys, path"]

def test_with_from_imports_only_sections():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(only_sections=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]

def test_with_from_imports_split_on_trailing_comma():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas={"os"},
    )
    config = Config(split_on_trailing_comma=True, line_length=30)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import (\n    path,\n    sys,\n)"]


# LLM-generated content at query #38
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


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #40
#--------------------------

```python
def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        imports={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"


# LLM-generated content at query #41
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #42
#--------------------------

```python
def test_with_straight_imports_predicate():
    parsed = parse.ParsedContent(
        as_map={"straight": {"module1": ["alias1"], "module2": ["alias2"]}},
        imports={"section": {"straight": {"module1": ["import1"], "module2": ["import2"]}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=True)
    straight_modules = ["module1", "module2"]
    section = "section"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == []


# LLM-generated content at query #43
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
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')"


# LLM-generated content at query #44
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) or config.only_sections


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {"module": []}, "above": {"from": {"module": None}}},
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
        categorized_comments={"from": {"module": ["comment1"]}, "above": {"from": {"module": None}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False, comment_prefix="# ")
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2  # comment1"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {"module": []}, "above": {"from": {"module": None}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", ["module.import1"], "import")
    assert result == ["from module import import2"]

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["*"]}}},
        categorized_comments={"from": {"module": []}, "nested": {"module": {"*": "star comment"}}, "above": {"from": {"module": None}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import *  # star comment"]

def test_with_from_imports_as_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {"module": []}, "above": {"from": {"module": None}}},
        as_map={"from": {"module.import1": ["import1 as alias1"], "module.import2": ["import2 as alias2"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1 as alias1, import2 as alias2"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {"module": []}, "above": {"from": {"module": None}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1", "from module import import2"]

def test_with_from_imports_multiline_reformat():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {"module": []}, "above": {"from": {"module": None}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(line_length=20)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import (\n    import1,\n    import2,\n)"]

def test_with_from_imports_split_on_trailing_comma():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {"module": []}, "above": {"from": {"module": None}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas={"module"},
    )
    config = Config(split_on_trailing_comma=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import (\n    import1,\n    import2,\n)"]

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {"module": ["comment1"]}, "above": {"from": {"module": None}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2"]

def test_with_from_imports_combine_as_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["*", "import1"]}}},
        categorized_comments={"from": {"module": []}, "above": {"from": {"module": None}}},
        as_map={"from": {"module.import1": ["import1 as alias1"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True, combine_star=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import *", "from module import import1 as alias1"]


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #3
#--------------------------

```python
def test_normalize_empty_lines_with_empty_list():
    assert _normalize_empty_lines([]) == [""]

def test_normalize_empty_lines_with_single_empty_line():
    assert _normalize_empty_lines([""]) == ["", ""]

def test_normalize_empty_lines_with_single_non_empty_line():
    assert _normalize_empty_lines(["hello"]) == ["hello", ""]

def test_normalize_empty_lines_with_multiple_empty_lines_at_end():
    assert _normalize_empty_lines(["hello", "", "  "]) == ["hello", ""]

def test_normalize_empty_lines_with_multiple_non_empty_lines():
    assert _normalize_empty_lines(["hello", "world"]) == ["hello", "world", ""]

def test_normalize_empty_lines_with_mixed_empty_and_non_empty_lines():
    assert _normalize_empty_lines(["hello", "", "world", "  "]) == ["hello", "", "world", ""]


# LLM-generated content at query #4
#--------------------------

```python
def test_sorted_imports_with_empty_parsed_content():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        imports={},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
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
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
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
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["THIRDPARTY"],
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
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["THIRDPARTY"],
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
        imports={"THIRDPARTY": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\n\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"numpy": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {"numpy": ["np"]}, "from": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import numpy as np\n\n"

def test_sorted_imports_with_combined_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# Comment above"]}, "from": {}}, "straight": {"os": ["# Inline comment"]}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "# Comment above\nimport os  # Inline comment\n\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"sys": set(), "os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"FUTURE": {"straight": {"__future__": set()}, "from": {}}, "THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["FUTURE", "THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\n\n\n\nimport os\n\n"

def test_sorted_imports_with_ensure_newline_before_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports


# LLM-generated content at query #5
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
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *"]

def test_with_from_imports_star_import_with_comment():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"os": {"*": "star comment"}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_combine_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*", "path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *"]

def test_with_from_imports_no_inline_sort():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["sys", "path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(no_inline_sort=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import sys, path"]

def test_with_from_imports_only_sections():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(only_sections=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path, sys"]


# LLM-generated content at query #6
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        line_separator="\n",
        import_index=-1,
        imports={},
        categorized_comments={},
        as_map={},
        place_imports={},
        import_placements={},
        sections=[],
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["comment"]}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os  # comment\n"

def test_sorted_imports_with_as_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["os_path"]}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os as os_path\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"THIRDPARTY": {"from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"from": {}}, "from": {}},
        as_map={"from": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n"

def test_sorted_imports_with_no_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"FUTURE": {"straight": {"__future__": set()}}, "THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["FUTURE", "THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import __future__\nimport os\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"THIRDPARTY": {"straight": {"sys": set(), "os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"FUTURE": {"straight": {"__future__": set()}}, "THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["FUTURE", "THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "import __future__\n\n\nimport os\n"

def test_sorted_imports_with_ensure_newline_before_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        line_separator="\n",
        import_index=0,
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\n"

def test_sorted_imports_with_lines_after_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello


# LLM-generated content at query #7
#--------------------------

```python
def test_sorted_imports_predicate():
    assert sorted_imports is not None


# LLM-generated content at query #8
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
    result = sorted_imports(parsed, config)
    assert result == "\n"


# LLM-generated content at query #9
#--------------------------

```python
def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('Hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('Hello')"


# LLM-generated content at query #10
#--------------------------

```python
def test_empty_output():
    assert _ensure_newline_before_comment([]) == []

def test_single_comment_line():
    assert _ensure_newline_before_comment(["# comment"]) == ["# comment"]

def test_single_non_comment_line():
    assert _ensure_newline_before_comment(["code"]) == ["code"]

def test_comment_after_code_without_newline():
    assert _ensure_newline_before_comment(["code", "# comment"]) == ["code", "", "# comment"]

def test_comment_after_code_with_newline():
    assert _ensure_newline_before_comment(["code", "", "# comment"]) == ["code", "", "# comment"]

def test_multiple_comments_in_sequence():
    assert _ensure_newline_before_comment(["# comment1", "# comment2"]) == ["# comment1", "# comment2"]

def test_comment_after_empty_line():
    assert _ensure_newline_before_comment(["", "# comment"]) == ["", "# comment"]

def test_mixed_lines_with_comments():
    assert _ensure_newline_before_comment(["code1", "# comment1", "code2", "# comment2"]) == ["code1", "", "# comment1", "code2", "", "# comment2"]

def test_comment_at_start():
    assert _ensure_newline_before_comment(["# comment", "code"]) == ["# comment", "code"]

def test_comment_at_end():
    assert _ensure_newline_before_comment(["code", "# comment"]) == ["code", "", "# comment"]

def test_multiple_non_comment_lines_before_comment():
    assert _ensure_newline_before_comment(["code1", "code2", "# comment"]) == ["code1", "code2", "", "# comment"]

def test_no_newline_needed():
    assert _ensure_newline_before_comment(["code", "", "# comment"]) == ["code", "", "# comment"]


# LLM-generated content at query #11
#--------------------------

```python
def test__with_from_imports_empty():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    assert _with_from_imports(parsed, config, [], "STDLIB", [], "import") == []

def test__with_from_imports_single_import():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    assert _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import") == ["from os import path"]

def test__with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": ["# comment"]},
            "above": {"from": {}},
            "nested": {"os": {"path": "# nested comment"}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False)
    assert _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import") == [
        "from os import path  # nested comment"
    ]

def test__with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    assert _with_from_imports(parsed, config, ["os"], "STDLIB", ["os.path"], "import") == [
        "from os import sys"
    ]

def test__with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    assert _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import") == [
        "from os import path as ospath"
    ]

def test__with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {"os": {"*": "# star comment"}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    assert _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import") == [
        "from os import *  # star comment"
    ]


# LLM-generated content at query #12
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={
            "section": {
                "from": {
                    "module": ["import1", "import2"]
                }
            }
        },
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set()
    )
    config = Config()
    from_modules = ["module"]
    section = "section"
    remove_imports = []
    import_type = "import"
    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert len(result) == 1
    assert "from module import import1, import2" in result[0]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={
            "section": {
                "from": {
                    "module": ["import1", "import2"]
                }
            }
        },
        categorized_comments={
            "from": {"module": ["comment1", "comment2"]},
            "above": {"from": {}},
            "nested": {}
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set()
    )
    config = Config()
    from_modules = ["module"]
    section = "section"
    remove_imports = []
    import_type = "import"
    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert len(result) == 1
    assert "from module import import1, import2" in result[0]
    assert "comment1" in result[0]
    assert "comment2" in result[0]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={
            "section": {
                "from": {
                    "module": ["import1", "import2"]
                }
            }
        },
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set()
    )
    config = Config()
    from_modules = ["module"]
    section = "section"
    remove_imports = ["module.import1"]
    import_type = "import"
    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert len(result) == 1
    assert "from module import import2" in result[0]

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={
            "section": {
                "from": {
                    "module": ["import1", "import2"]
                }
            }
        },
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={
            "from": {
                "module.import1": ["import1 as alias1"],
                "module.import2": ["import2 as alias2"]
            }
        },
        line_separator="\n",
        trailing_commas=set()
    )
    config = Config()
    from_modules = ["module"]
    section = "section"
    remove_imports = []
    import_type = "import"
    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert len(result) == 3
    assert "from module import import1 as alias1" in result[0]
    assert "from module import import2 as alias2" in result[1]
    assert "from module import import1, import2" in result[2]

def test_with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={
            "section": {
                "from": {
                    "module": ["*", "import1"]
                }
            }
        },
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"module": {"*": ["star comment"]}}
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set()
    )
    config = Config()
    from_modules = ["module"]
    section = "section"
    remove_imports = []
    import_type = "import"
    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert len(result) == 1
    assert "from module import *" in result[0]
    assert "star comment" in result[0]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={
            "section": {
                "from": {
                    "module": ["import1", "import2"]
                }
            }
        },
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set()
    )
    config = Config()
    config.force_single_line = True
    from_modules = ["module"]
    section = "section"
    remove_imports = []
    import_type = "import"
    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert len(result) == 2
    assert "from module import import1" in result[0]
    assert "from module import import2" in result[1]

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={
            "section": {
                "from": {
                    "module": ["import1", "import2"]
                }
            }
        },
        categorized_comments={
            "from": {"module": ["comment1", "comment2"]},
            "above": {"from": {}},
            "nested": {}
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set()
    )
    config = Config()
    config.ignore_comments = True
    from_modules = ["module"]
    section = "section"
    remove_imports = []
    import_type = "import"
    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert len(result) == 1
    assert "from module import import1, import2" in result[0]
    assert "comment1" not in result[0]
    assert "comment2" not in result[0]

def test_with_from_imports_combine_as_imports():
    parsed = parse.ParsedContent(
        imports={
            "section": {
                "from": {
                    "module": ["import1", "import2"]
                }
            }
        },
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={
            "from": {
                "module.import1": ["import1 as alias1"],
                "module.import2": ["import2 as alias2"]
            }
        },
        line_separator="\n",
        trailing_commas=set()
    )
    config = Config()
    config.combine_as_imports = True
    from_modules = ["module"]
    section = "section"
    remove_imports = []
    import_type = "import"
    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert len(result) == 1
    assert "from module import import1 as alias1, import2 as alias2" in result[0]

def test_with_from_imports_combine_star():
    parsed = parse.ParsedContent(
        imports={
            "section": {
                "from": {
                    "module": ["*", "import1"]
                }
            }
        },
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"module": {"*": ["star comment"]}}
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set()
    )
    config = Config()
    config.combine_star = True
    from_modules = ["module"]
    section = "section"
    remove_imports = []
    import_type = "import"
    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert len(result) == 1
    assert "from module import *" in result[0]
    assert "star comment" in result[0]

def test_with_from_imports_only_sections():
    parsed = parse.ParsedContent(
        imports={
            "section": {
                "from": {
                    "module": ["import1", "import2"]
                }
            }
        },
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set()
    )
    config = Config()
    config.only_sections = True
    from_modules = ["module"]
    section = "section"
    remove_imports = []
    import_type = "import"
    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert len(result) == 1
    assert "from module import import1, import2" in result[


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_1():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = []
    section = ""
    remove_imports = []
    import_type = ""

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)

    assert result == []


# LLM-generated content at query #14
#--------------------------

```python
def test__with_from_imports_basic():
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

def test__with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
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
    assert result == ["from os import path  # comment1; comment2"]

def test__with_from_imports_remove_imports():
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

def test__with_from_imports_with_as_imports():
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

def test__with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"nested": {"os": {"*": ["star comment"]}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test__with_from_imports_force_single_line():
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

def test__with_from_imports_multiline():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys", "env"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(line_length=20)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import (\n    path,\n    sys,\n    env,\n)"]

def test__with_from_imports_with_above_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {"os": ["above comment"]}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["# above comment", "from os import path"]

def test__with_from_imports_with_nested_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"os": {"path": ["nested comment"]}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # nested comment"]

def test__with_from_imports_with_straight_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {},
            "straight": {"os.path": ["straight comment"]},
        },
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path as ospath  # straight comment"]


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) or config.only_sections


# LLM-generated content at query #16
#--------------------------

```python
def test_with_straight_imports_empty_modules():
    parsed = parse.ParsedContent()
    config = Config()
    result = _with_straight_imports(parsed, config, [], "section", [], "import")
    assert result == []

def test_with_straight_imports_single_module_no_comments():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"straight": {"module1": []}}}
    parsed.as_map = {"straight": {}}
    parsed.categorized_comments = {"above": {"straight": {}}, "straight": {}}
    config = Config()
    result = _with_straight_imports(parsed, config, ["module1"], "section", [], "import")
    assert result == ["import module1"]

def test_with_straight_imports_single_module_with_inline_comment():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"straight": {"module1": []}}}
    parsed.as_map = {"straight": {}}
    parsed.categorized_comments = {"above": {"straight": {}}, "straight": {"module1": ["comment1"]}}
    config = Config()
    result = _with_straight_imports(parsed, config, ["module1"], "section", [], "import")
    assert result == ["import module1  # comment1"]

def test_with_straight_imports_single_module_with_above_comment():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"straight": {"module1": []}}}
    parsed.as_map = {"straight": {}}
    parsed.categorized_comments = {"above": {"straight": {"module1": ["# comment1"]}}, "straight": {}}
    config = Config()
    result = _with_straight_imports(parsed, config, ["module1"], "section", [], "import")
    assert result == ["# comment1", "import module1"]

def test_with_straight_imports_combine_straight_imports():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"straight": {"module1": [], "module2": []}}}
    parsed.as_map = {"straight": {}}
    parsed.categorized_comments = {"above": {"straight": {}}, "straight": {"module1": ["comment1"], "module2": ["comment2"]}}
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["module1", "module2"], "section", [], "import")
    assert result == ["import module1, module2  # comment1 comment2"]

def test_with_straight_imports_ignore_comments():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"straight": {"module1": []}}}
    parsed.as_map = {"straight": {}}
    parsed.categorized_comments = {"above": {"straight": {}}, "straight": {"module1": ["comment1"]}}
    config = Config(ignore_comments=True)
    result = _with_straight_imports(parsed, config, ["module1"], "section", [], "import")
    assert result == ["import module1"]

def test_with_straight_imports_with_as_import():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"straight": {"module1": []}}}
    parsed.as_map = {"straight": {"module1": ["alias1"]}}
    parsed.categorized_comments = {"above": {"straight": {}}, "straight": {}}
    config = Config()
    result = _with_straight_imports(parsed, config, ["module1"], "section", [], "import")
    assert result == ["import module1 as alias1"]

def test_with_straight_imports_remove_import():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"straight": {"module1": []}}}
    parsed.as_map = {"straight": {}}
    parsed.categorized_comments = {"above": {"straight": {}}, "straight": {}}
    config = Config()
    result = _with_straight_imports(parsed, config, ["module1"], "section", ["module1"], "import")
    assert result == []

def test_with_straight_imports_custom_comment_prefix():
    parsed = parse.ParsedContent()
    parsed.imports = {"section": {"straight": {"module1": []}}}
    parsed.as_map = {"straight": {}}
    parsed.categorized_comments = {"above": {"straight": {}}, "straight": {"module1": ["comment1"]}}
    config = Config(comment_prefix="# ")
    result = _with_straight_imports(parsed, config, ["module1"], "section", [], "import")
    assert result == ["import module1  # comment1"]


# LLM-generated content at query #17
#--------------------------

```python
def test_sorted_imports_predicate():
    assert sorted_imports is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #19
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ()}},
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
        categorized_comments={"from": {"os": ("# comment",)}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # comment"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", ["os.path"], "import")
    assert result == []

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path", "from os import path as ospath"]

def test_with_from_imports_with_star_comment():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"nested": {"os": {"*": "# star comment"}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path", "from os import sys"]

def test_with_from_imports_combine_as_imports():
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

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ("# comment",)}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]

def test_with_from_imports_only_sections():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(only_sections=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]

def test_with_from_imports_no_inline_sort():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["sys", "path"]}}},
        categorized_comments={"from": {"os": ()}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(no_inline_sort=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import sys, path"]


# LLM-generated content at query #20
#--------------------------

```python
def test_with_star_comments_when_star_comment_exists():
    parsed = parse.ParsedContent(categorized_comments={"nested": {module: {"*": "star_comment"}}})
    result = _with_star_comments(parsed, module, ["comment1", "comment2"])
    assert result == ["comment1", "comment2", "star_comment"]

def test_with_star_comments_when_star_comment_does_not_exist():
    parsed = parse.ParsedContent(categorized_comments={"nested": {module: {}}})
    result = _with_star_comments(parsed, module, ["comment1", "comment2"])
    assert result == ["comment1", "comment2"]

def test_with_star_comments_when_module_does_not_exist():
    parsed = parse.ParsedContent(categorized_comments={"nested": {}})
    result = _with_star_comments(parsed, module, ["comment1", "comment2"])
    assert result == ["comment1", "comment2"]


# LLM-generated content at query #21
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
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["# comment"]}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(ignore_comments=False)
    result = sorted_imports(parsed, config)
    assert result == "import os  # comment\n"

def test_sorted_imports_combine_straight_imports():
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

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from os import path\n"

def test_sorted_imports_with_section_headings():
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
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library\nimport os\n"

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
    config = Config(lines_between_sections=1)
    result = sorted_imports(parsed, config)
    assert result == "import __future__\n\nimport os\n"

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


# LLM-generated content at query #22
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
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
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
    result = sorted_imports(parsed)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["# comment"]}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
    assert result == "import os  # comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": {"ospath"}}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
    assert result == "import os as ospath\nimport os\n"

def test_sorted_imports_with_remove_imports():
    config = Config(remove_imports=["os"])
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
    result = sorted_imports(parsed, config=config)
    assert result == "import sys\n"

def test_sorted_imports_with_combine_straight_imports():
    config = Config(combine_straight_imports=True)
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
    result = sorted_imports(parsed, config=config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_force_sort_within_sections():
    config = Config(force_sort_within_sections=True)
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
    result = sorted_imports(parsed, config=config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_import_headings():
    config = Config(import_headings={"stdlib": "Standard Library"})
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
    result = sorted_imports(parsed, config=config)
    assert result == "# Standard Library\nimport os\n"

def test_sorted_imports_with_lines_between_sections():
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}, "THIRDPARTY": {"straight": {"numpy": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB", "THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import os\n\n\nimport numpy\n"

def test_sorted_imports_with_lines_after_imports():
    config = Config(lines_after_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import os\n\n\nprint('hello')"

def test_sorted_imports_with_ensure_newline_before_comments():
    config = Config(ensure_newline_before_comments=True)
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
    parsed.lines_without_imports.append("# comment")
    result = sorted_imports(parsed, config=config)
    assert result == "import os\n\n# comment\n"


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #24
#--------------------------

```python
def test__with_from_imports_basic_case():
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

def test__with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["*"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import *"]

def test__with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1"]}}},
        categorized_comments={
            "from": {"module": ["comment1"]},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1  # comment1"]

def test__with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {"module.import1": ["import1 as alias1"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1 as alias1"]

def test__with_from_imports_with_remove_imports():
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

def test__with_from_imports_with_force_single_line():
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

def test__with_from_imports_with_combine_as_imports():
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

def test__with_from_imports_with_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1"]}}},
        categorized_comments={
            "from": {"module": ["comment1"]},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1"]

def test__with_from_imports_with_comment_prefix():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1"]}}},
        categorized_comments={
            "from": {"module": ["comment1"]},
            "above": {"from": {}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(comment_prefix="# ")
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1  # comment1"]

def test__with_from_imports_with_above_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {"module": ["above_comment1"]}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["above_comment1", "from module import import1"]


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #26
#--------------------------

```python
def test_with_straight_imports_empty_modules():
    parsed = parse.ParsedContent(
        imports={"straight": {}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}}
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, [], "straight", [], "import")
    assert result == []

def test_with_straight_imports_combine_no_as_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"module1": [], "module2": []}},
        as_map={"straight": {}},
        categorized_comments={
            "above": {"straight": {"module1": ["# comment above"]}},
            "straight": {"module1": ["# inline comment"]}
        }
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["module1", "module2"], "straight", [], "import")
    assert result == ["# comment above", "import module1, module2  # inline comment"]

def test_with_straight_imports_combine_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"module1": [], "module2": []}},
        as_map={"straight": {"module1": ["alias"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {}}
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["module1", "module2"], "straight", [], "import")
    assert result == ["import module1 as alias", "import module2"]

def test_with_straight_imports_no_combine():
    parsed = parse.ParsedContent(
        imports={"straight": {"module1": [], "module2": []}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}}
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["module1", "module2"], "straight", [], "import")
    assert result == ["import module1", "import module2"]

def test_with_straight_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"straight": {"module1": [], "module2": []}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}}
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["module1", "module2"], "straight", ["module1"], "import")
    assert result == ["import module2"]

def test_with_straight_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"straight": {"module1": []}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {"module1": ["# comment"]}}
    )
    config = Config(combine_straight_imports=False, ignore_comments=True)
    result = _with_straight_imports(parsed, config, ["module1"], "straight", [], "import")
    assert result == ["import module1"]

def test_with_straight_imports_custom_comment_prefix():
    parsed = parse.ParsedContent(
        imports={"straight": {"module1": []}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {"module1": ["# comment"]}}
    )
    config = Config(combine_straight_imports=False, comment_prefix=" # ")
    result = _with_straight_imports(parsed, config, ["module1"], "straight", [], "import")
    assert result == ["import module1 # comment"]


# LLM-generated content at query #27
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

def test_sorted_imports_basic_straight_imports():
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
    assert result == "\nimport os\nimport sys"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"from": {}}, "from": {}},
        as_map={"from": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nfrom os import path"

def test_sorted_imports_combine_straight_imports():
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
    assert result == "\nimport os, sys"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["comment"]}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nimport os  # comment"

def test_sorted_imports_remove_imports():
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
    assert result == "\nimport sys"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["ospath"]}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nimport os as ospath"

def test_sorted_imports_with_section_headings():
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
    assert result == "\n# Third Party Imports\nimport os"

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
    assert result == "\nimport os\nimport sys"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}}, "FIRSTPARTY": {"straight": {"sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY", "FIRSTPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(lines_between_sections=1)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\n\nimport sys"

def test_sorted_imports_with_ensure_newline_before_comments():
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
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os"


# LLM-generated content at query #28
#--------------------------

```python
def test__with_from_imports_basic_case():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2"]

def test__with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["*"]}}},
        categorized_comments={"from": {}, "nested": {"module": {"*": "star comment"}}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import *  # star comment"]

def test__with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {"module.import1": ["import1 as alias1"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1 as alias1"]

def test__with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", ["module.import1"], "import")
    assert result == ["from module import import2"]

def test__with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1", "from module import import2"]

def test__with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1"]}}},
        categorized_comments={"from": {"module": ["comment1"]}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1  # comment1"]

def test__with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1"]}}},
        categorized_comments={"from": {"module": ["comment1"]}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1"]

def test__with_from_imports_multiline_reformat():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2", "import3"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(line_length=20)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert len(result) > 1


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_1():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        imports={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    straight_modules = []
    section = "straight"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(
        parsed, config, straight_modules, section, remove_imports, import_type
    )
    assert result == []


# LLM-generated content at query #30
#--------------------------

```python
def test_sorted_imports_predicate_false():
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


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_1_evaluates_to_false():
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) or config.only_sections


# LLM-generated content at query #32
#--------------------------

```python
def test_with_straight_imports_combine_straight_imports():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": ["sys"], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}}
    )
    config = Config(combine_straight_imports=True, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    assert _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type) == ["import os, sys"]

def test_with_straight_imports_combine_straight_imports_with_inline_comments():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": ["sys"], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["comment1"], "sys": ["comment2"]}}
    )
    config = Config(combine_straight_imports=True, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    assert _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type) == ["import os, sys  # comment1 comment2"]

def test_with_straight_imports_combine_straight_imports_with_above_comments():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": ["sys"], "sys": []}}},
        categorized_comments={"above": {"straight": {"os": ["# above comment"]}}, "straight": {}}
    )
    config = Config(combine_straight_imports=True, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    assert _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type) == ["# above comment", "import os, sys"]

def test_with_straight_imports_combine_straight_imports_with_as_imports():
    parsed = parse.ParsedContent(
        as_map={"straight": {"os": ["os_alias"]}},
        imports={"STDLIB": {"straight": {"os": ["sys"], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}}
    )
    config = Config(combine_straight_imports=True, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    assert _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type) == ["import os as os_alias", "import sys"]

def test_with_straight_imports_combine_straight_imports_removed():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": ["sys"], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}}
    )
    config = Config(combine_straight_imports=True, ignore_comments=True, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    assert _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type) == ["import os, sys"]

def test_with_straight_imports_no_combine_straight_imports():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": ["sys"], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}}
    )
    config = Config(combine_straight_imports=False, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    assert _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type) == ["import os", "import sys"]

def test_with_straight_imports_no_combine_straight_imports_with_inline_comments():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": ["sys"], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["comment1"], "sys": ["comment2"]}}
    )
    config = Config(combine_straight_imports=False, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    assert _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type) == ["import os  # comment1", "import sys  # comment2"]

def test_with_straight_imports_no_combine_straight_imports_with_above_comments():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": ["sys"], "sys": []}}},
        categorized_comments={"above": {"straight": {"os": ["# above comment"]}}, "straight": {}}
    )
    config = Config(combine_straight_imports=False, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    assert _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type) == ["# above comment", "import os", "import sys"]

def test_with_straight_imports_no_combine_straight_imports_with_as_imports():
    parsed = parse.ParsedContent(
        as_map={"straight": {"os": ["os_alias"]}},
        imports={"STDLIB": {"straight": {"os": ["sys"], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}}
    )
    config = Config(combine_straight_imports=False, ignore_comments=False, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    assert _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type) == ["import os as os_alias", "import sys"]

def test_with_straight_imports_no_combine_straight_imports_removed():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        imports={"STDLIB": {"straight": {"os": ["sys"], "sys": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}}
    )
    config = Config(combine_straight_imports=False, ignore_comments=True, comment_prefix="# ")
    straight_modules = ["os", "sys"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"
    assert _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type) == ["import os", "import sys"]


# LLM-generated content at query #33
#--------------------------

```python
def test_star_comment_is_none():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"module": {}}}
    module = "module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == comments


# LLM-generated content at query #34
#--------------------------

```python
def test_sorted_imports_empty_parsed_content():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={},
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=[],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "\n"

def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=[],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "print('hello')\n"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "import os\n\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "import os\nimport sys\n\n"

def test_sorted_imports_single_from_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "from os import path\n\n"

def test_sorted_imports_multiple_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set(), "sys": set()}}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "from os import path, sys\n\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        categorized_comments={
            "above": {"straight": {}, "from": {}},
            "straight": {"os": ["# comment"]},
            "from": {},
        },
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "import os  # comment\n\n"

def test_sorted_imports_with_as_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {"os": ["alias"]}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed) == "import os as alias\n\n"

def test_sorted_imports_with_remove_imports():
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed, config) == "import sys\n\n"

def test_sorted_imports_with_combine_straight_imports():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed, config) == "import os, sys\n\n"

def test_sorted_imports_with_force_sort_within_sections():
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"sys": set(), "os": set()}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed, config) == "import os\nimport sys\n\n"

def test_sorted_imports_with_import_headings():
    config = Config(import_headings={"stdlib": "Standard Library"})
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed, config) == "# Standard Library\nimport os\n\n"


# LLM-generated content at query #35
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
    )
    assert sorted_imports(parsed) == "print('hello')"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
    )
    assert sorted_imports(parsed) == "import os\n"

def test_sorted_imports_single_from_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": {"path": set()}},
            }
        },
        sections=["THIRDPARTY"],
    )
    assert sorted_imports(parsed) == "from os import path\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {"sys": set(), "os": set()},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
    )
    assert sorted_imports(parsed) == "import os\nimport sys\n"

def test_sorted_imports_multiple_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": {"path": set()}, "sys": {"argv": set()}},
            }
        },
        sections=["THIRDPARTY"],
    )
    assert sorted_imports(parsed) == "from os import path\nfrom sys import argv\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
        categorized_comments={
            "above": {"straight": {"os": ["# Comment above"]}},
            "straight": {"os": ["# Inline comment"]},
        },
    )
    assert sorted_imports(parsed) == "# Comment above\nimport os  # Inline comment\n"

def test_sorted_imports_with_as_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
        as_map={"straight": {"os": ["osp"]}},
    )
    assert sorted_imports(parsed) == "import os as osp\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
    )
    config = Config(remove_imports=["os"])
    assert sorted_imports(parsed, config) == "import sys\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
    )
    config = Config(combine_straight_imports=True)
    assert sorted_imports(parsed, config) == "import os, sys\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {"sys": set(), "os": set()},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
    )
    config = Config(force_sort_within_sections=True)
    assert sorted_imports(parsed, config) == "import os\nimport sys\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    assert sorted_imports(parsed, config) == "# Third Party Imports\nimport os\n"

def test_sorted_imports_with_ensure_newline_before_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
    )
    config = Config(ensure_newline_before_comments=True)
    assert sorted_imports(parsed, config) == "import os\n"

def test_sorted_imports_with_lines_after_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
    )
    config = Config(lines_after_imports=2)
    assert sorted_imports(parsed, config) == "import os\n\n\nprint('hello')"

def test_sorted_imports_with_no_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "FUTURE": {
                "straight": {"__future__": set()},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        sections=["FUTURE", "THIRDPARTY"],
    )
    config = Config(no_sections=True)
    assert sorted_imports(parsed, config) == "from __future__ import absolute_import\nimport os\n"

def test_sorted_imports_with_from_first():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"sys": {"argv": set()}},
            }
        },
        sections=["THIRDPARTY"],
    )
    config = Config(from_first=True)
    assert sorted_imports(parsed, config) == "from sys import argv\nimport os\n"

def test_sorted_imports_with_star_first():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": {"*": set()}, "sys": {"argv": set()}},
            }
        },
        sections=["THIRDPARTY"],
    )
    config = Config(star_first=True)
    assert sorted_imports(parsed, config) == "from os import *\nfrom sys import argv\n"

def test_sorted_imports_with_reverse_sort():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
       


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    )


# LLM-generated content at query #37
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
    ) and not config.only_sections


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #39
#--------------------------

```python
def test_combine_straight_imports_without_as_imports():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        imports={"STANDARD_LIBRARY": {"straight": {"module1": [], "module2": []}}}
    )
    straight_modules = ["module1", "module2"]
    section = "STANDARD_LIBRARY"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == ["import module1, module2"]


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_evaluates_to_false():
    config = Config(
        no_inline_sort=True,
        force_single_line=False,
        single_line_exclusions=[],
        only_sections=False
    )
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and "module" not in config.single_line_exclusions)
    )


# LLM-generated content at query #41
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
    assert result == []


# LLM-generated content at query #42
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #43
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
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "print('hello')"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        original_line_count=1,
        place_imports={},
        import_placements={},
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
            "above": {"straight": {"os": ["# Comment above"]}},
            "straight": {"os": ["# Inline comment"]},
        },
        as_map={"straight": {}},
        sections=["STDLIB"],
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "# Comment above\nimport os  # Inline comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["path"]}},
        sections=["STDLIB"],
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "import os as path\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"from": {}}, "from": {"os": {}}},
        as_map={"from": {}},
        sections=["STDLIB"],
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "from os import path\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"sys": set(), "os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    assert sorted_imports(parsed, config) == "import os\nimport sys\n"

def test_sorted_imports_with_no_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}, "THIRDPARTY": {"straight": {"django": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB", "THIRDPARTY"],
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    assert sorted_imports(parsed, config) == "import django\nimport os\n"


# LLM-generated content at query #44
#--------------------------

```python
def test_sorted_imports_predicate():
    assert sorted_imports is not None


# LLM-generated content at query #45
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #46
#--------------------------

```python
def test_predicate_evaluates_to_false():
    config = Config()
    config.no_inline_sort = True
    config.force_single_line = False
    config.only_sections = True

    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    )


# LLM-generated content at query #47
#--------------------------

```python
def test_with_from_imports_basic_case():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": []}},
        as_map={"from": {"os.path": []}},
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
    assert result == ["from os import path"]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["# comment"]}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path  # comment"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": []}},
        as_map={"from": {"os.path": [], "os.sys": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=["os.sys"],
        import_type="import",
    )
    assert result == ["from os import path"]

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": []}},
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
    assert result == ["from os import path", "from os import path as ospath"]

def test_with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"nested": {"os": {"*": ["# all"]}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import *  # all"]


# LLM-generated content at query #48
#--------------------------

```python
def test_with_straight_imports_predicate():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        imports={"section": {"straight": {}}}
    )
    config = Config(combine_straight_imports=True)
    straight_modules = ["module1"]
    section = "section"
    remove_imports = []
    import_type = "import"
    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == ["import module1"]


