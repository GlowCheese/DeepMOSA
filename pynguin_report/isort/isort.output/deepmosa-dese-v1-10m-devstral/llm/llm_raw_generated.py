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

def test_comment_after_empty_line():
    assert _ensure_newline_before_comment(["", "# comment"]) == ["", "# comment"]

def test_comment_after_non_empty_line():
    assert _ensure_newline_before_comment(["code", "# comment"]) == ["code", "", "# comment"]

def test_multiple_comments():
    assert _ensure_newline_before_comment(["# comment1", "# comment2"]) == ["# comment1", "# comment2"]

def test_comment_after_comment():
    assert _ensure_newline_before_comment(["# comment1", "# comment2"]) == ["# comment1", "# comment2"]

def test_mixed_lines():
    assert _ensure_newline_before_comment(["code1", "# comment", "code2"]) == ["code1", "", "# comment", "code2"]

def test_comment_at_end():
    assert _ensure_newline_before_comment(["code", "# comment"]) == ["code", "", "# comment"]

def test_no_comments():
    assert _ensure_newline_before_comment(["code1", "code2"]) == ["code1", "code2"]

def test_empty_lines_with_comments():
    assert _ensure_newline_before_comment(["", "# comment", ""]) == ["", "# comment", ""]

def test_comment_after_empty_line_no_extra_newline():
    assert _ensure_newline_before_comment(["", "# comment"]) == ["", "# comment"]


# LLM-generated content at query #2
#--------------------------

```python
def test_with_from_imports_basic():
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
    config = Config(ignore_comments=False)
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

def test_with_from_imports_as_imports():
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
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *"]

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

def test_with_from_imports_above_comments():
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


# LLM-generated content at query #3
#--------------------------

```python
def test_with_straight_imports_empty_modules():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, [], "standard", [], "import")
    assert result == []

def test_with_straight_imports_combined_no_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == ["import sys, os"]

def test_with_straight_imports_combined_with_inline_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"sys": ["comment1"], "os": ["comment2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == ["import sys, os  # comment1 comment2"]

def test_with_straight_imports_combined_with_above_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {"sys": ["# above comment"]}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == ["# above comment", "import sys, os"]

def test_with_straight_imports_combined_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"sys": ["s"]}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == ["import sys as s", "import os"]

def test_with_straight_imports_combined_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"sys": ["comment1"], "os": ["comment2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True, ignore_comments=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == ["import sys, os"]

def test_with_straight_imports_combined_custom_prefix():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"sys": ["comment1"], "os": ["comment2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True, comment_prefix=" # ")
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == ["import sys, os  # comment1 comment2"]

def test_with_straight_imports_removed_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", ["sys"], "import")
    assert result == ["import os"]

def test_with_straight_imports_no_combine():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"sys": ["comment1"], "os": ["comment2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import")
    assert result == ["import sys  # comment1", "import os  # comment2"]

def test_with_straight_imports_with_from_import():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "from ... import")
    assert result == ["from ... import sys, os"]


# LLM-generated content at query #4
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
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
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
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
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
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# comment above"]}}, "straight": {"os": ["# inline comment"]}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    assert sorted_imports(parsed, config) == "# comment above\nimport os  # inline comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": {"path"}}},
        sections=["THIRDPARTY"],
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
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
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
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
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
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}, "FUTURE": {"straight": {"__future__": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY", "FUTURE"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(no_sections=True)
    assert sorted_imports(parsed, config) == "from __future__ import annotations\nimport os\n"

def test_sorted_imports_with_force_sort_within_sections():
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
    config = Config(force_sort_within_sections=True)
    assert sorted_imports(parsed, config) == "import os\nimport sys\n"

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
    assert sorted_imports(parsed, config) == "# Third Party Imports\nimport os\n"

def test_sorted_imports_with_import_footers():
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
    config = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    assert sorted_imports(parsed, config) == "import os\n\n# End of Third Party Imports\n"

def test_sorted_imports_with_ensure_newline_before_comments():
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
    config = Config(ensure_newline_before_comments=True)
    assert sorted_imports(parsed, config) == "import os\n"

def test_sorted_imports_with_formatting_function():
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
    config = Config(formatting_function=lambda x, y, z: x)
    assert sorted_imports(parsed, config) == "import os\n"

def test_sorted_imports_with_lines_before_imports():
    parsed


# LLM-generated content at query #5
#--------------------------

```python
def test_combine_straight_imports_with_no_as_imports():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        imports={"straight": {"module1": [], "module2": []}},
    )
    straight_modules = ["module1", "module2"]
    section = "straight"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)

    assert result == ["import module1, module2"]


# LLM-generated content at query #6
#--------------------------

```python
def test_sorted_imports_basic_case():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        sections=["THIRDPARTY"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        sections=["THIRDPARTY"],
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"os": ["# comment for os"], "sys": ["# comment for sys"]}
        },
        as_map={"straight": {}},
        place_imports={},
        import_placements={}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os  # comment for os\nimport sys  # comment for sys\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        sections=["THIRDPARTY"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["path"], "sys": ["argv"]}},
        place_imports={},
        import_placements={}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os as path\nimport sys as argv\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        sections=["THIRDPARTY"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={}
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        imports={},
        sections=[],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        sections=["THIRDPARTY"],
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"os": ["# comment for os"], "sys": ["# comment for sys"]}
        },
        as_map={"straight": {}},
        place_imports={},
        import_placements={}
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os  # comment for os\nimport sys  # comment for sys\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        sections=["THIRDPARTY"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={}
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\nimport sys\n"

def test_sorted_imports_with_ensure_newline_before_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        sections=["THIRDPARTY"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={}
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_lines_after_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        sections=["THIRDPARTY"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={}
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\n\nprint('hello')\n"

def test_sorted_imports_with_lines_before_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        sections=["THIRDPARTY"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={}
    )
    config = Config(lines_before_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport os\nimport sys\nprint('hello')\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        sections=["THIRDPARTY"],
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={}
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n"


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #8
#--------------------------

```python
def test_any_with_straight_imports():
    parsed = parse.ParsedContent(
        as_map={"straight": {"module1": ["alias1"], "module2": []}},
        imports={"section": {"straight": {"module1": ["alias1"], "module2": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    straight_modules = ["module1", "module2"]
    assert any(module in parsed.as_map["straight"] for module in straight_modules)


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

```python
def test_with_star_comments_when_star_comment_exists():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"module": {"*": "star comment"}}}
    module = "module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2", "star comment"]

def test_with_star_comments_when_star_comment_does_not_exist():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"module": {}}}
    module = "module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2"]

def test_with_star_comments_when_module_does_not_exist():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {}}
    module = "module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2"]


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #12
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


# LLM-generated content at query #13
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
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"


# LLM-generated content at query #14
#--------------------------

```python
def test_sorted_imports_predicate():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        line_separator="\n",
        imports={"FUTURE": {"straight": {}, "from": {}}},
        sections=["FUTURE"],
        place_imports={},
        import_placements={},
        original_line_count=1
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n"


# LLM-generated content at query #15
#--------------------------

```python
def test_as_imports_predicate_with_straight_modules_in_as_map():
    parsed = parse.ParsedContent()
    parsed.as_map = {"straight": {"module1": ["alias1"], "module2": ["alias2"]}}
    straight_modules = ["module1", "module2"]
    as_imports = any(module in parsed.as_map["straight"] for module in straight_modules)
    assert as_imports is True


# LLM-generated content at query #16
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
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
    assert result == ["from os import path  # comment"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
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
        remove_imports=["os.path"],
        import_type="import",
    )
    assert result == []

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"nested": {"os": {"*": "star comment"}}},
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
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
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
    assert result == ["from os import path as ospath  # comment"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": [], "os.sys": []}},
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
    assert result == ["from os import path  # comment", "from os import sys"]

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": []}},
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

def test_with_from_imports_no_inline_sort():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["sys", "path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.sys": [], "os.path": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(no_inline_sort=True)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import sys, path  # comment"]

def test_with_from_imports_combine_as_imports_no_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True, combine_star=False)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import path as ospath  # comment"]

def test_with_from_imports_above_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": ["comment"]},
            "above": {"from": {"os": ["above comment"]}},
        },
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
    assert result == ["# above comment", "from os import path  # comment"]

def test_with_from_imports_nested_comment():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": ["comment"]},
            "nested": {"os": {"path": "nested comment"}},
        },
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
    assert result == ["from os import path  # comment; nested comment"]

def test_with_from_imports_noqa_comment_hanging_indent():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": ["comment"]},
            "nested": {"os": {"path": "noqa: F401"}},
        },
        as_map={"from": {"os.path": []}},
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
    assert result == ["from os import path  # comment; noqa: F401"]

def test_with_from_imports_split_on_trailing_comma():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": [], "os.sys": []}},
        line_separator="\n",
        trailing_commas={"os"},
    )
    config = Config(split_on_trailing_comma=True)
    result = _with_from_imports(
        parsed=parsed,
        config=config,
        from_modules=["os"],
        section="STDLIB",
        remove_imports=[],
        import_type="import",
    )
    assert result == ["from os import (\n    path,\n    sys,\n)  # comment"]

def test_with_from_imports_force_grid_wrap():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys", "env"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": [], "os.sys": [], "os.env": []}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_grid_wrap=2)
    result = _with_from_imports(
        parsed=parsed,


# LLM-generated content at query #17
#--------------------------

```python
def test_with_straight_imports_no_modules():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, [], "standard", [], "import") == []

def test_with_straight_imports_combine_no_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import") == [
        "import sys, os"
    ]

def test_with_straight_imports_combine_with_inline_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"sys": ["comment1"], "os": ["comment2"]},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import") == [
        "import sys, os  # comment1 comment2"
    ]

def test_with_straight_imports_combine_with_above_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={
            "above": {"straight": {"sys": ["# above comment"]}},
            "straight": {},
        },
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import") == [
        "# above comment",
        "import sys, os"
    ]

def test_with_straight_imports_no_combine_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": ["sys"]}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"sys": ["s"]}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys"], "standard", [], "import") == [
        "import sys as s"
    ]

def test_with_straight_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "standard", ["sys"], "import") == [
        "import os"
    ]

def test_with_straight_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": []}}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"sys": ["comment"]},
        },
        as_map={"straight": {}},
    )
    config = Config(ignore_comments=True)
    assert _with_straight_imports(parsed, config, ["sys"], "standard", [], "import") == [
        "import sys"
    ]

def test_with_straight_imports_custom_comment_prefix():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": []}}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"sys": ["comment"]},
        },
        as_map={"straight": {}},
    )
    config = Config(comment_prefix=" # ")
    assert _with_straight_imports(parsed, config, ["sys"], "standard", [], "import") == [
        "import sys # comment"
    ]


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_151_evaluates_to_false():
    output = ["  ", "  ", "  "]
    assert not (output and output[-1].strip() == "")


# LLM-generated content at query #19
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        imports={},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=[],
        forced_separate=[],
    )
    config = Config()
    assert sorted_imports(parsed, config) == "print('hello')"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    config = Config()
    assert sorted_imports(parsed, config) == "\nimport os"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    config = Config()
    assert sorted_imports(parsed, config) == "\nimport os\nimport sys"

def test_sorted_imports_single_from_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {}, "from": {"os": [("path", None)]}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    config = Config()
    assert sorted_imports(parsed, config) == "\nfrom os import path"

def test_sorted_imports_multiple_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {}, "from": {"os": [("path", None), ("environ", None)]}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    config = Config()
    assert sorted_imports(parsed, config) == "\nfrom os import environ, path"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# comment above"]}, "from": {}}, "straight": {"os": ["# inline comment"]}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    config = Config()
    assert sorted_imports(parsed, config) == "\n# comment above\nimport os  # inline comment"

def test_sorted_imports_with_as_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {"os": ["os_path"]}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    config = Config()
    assert sorted_imports(parsed, config) == "\nimport os as os_path"

def test_sorted_imports_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    config = Config(combine_straight_imports=True)
    assert sorted_imports(parsed, config) == "\nimport os, sys"

def test_sorted_imports_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    config = Config(remove_imports=["os"])
    assert sorted_imports(parsed, config) == "\nimport sys"

def test_sorted_imports_no_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"FUTURE": {"straight": {"__future__": ["__future__"]}, "from": {}}, "STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["FUTURE", "STDLIB"],
        forced_separate=[],
    )
    config = Config(no_sections=True)
    assert sorted_imports(parsed, config) == "\nimport __future__\nimport os"

def test_sorted_imports_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"sys": ["sys"], "os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    config = Config(force_sort_within_sections=True)
    assert sorted_imports(parsed, config) == "\nimport os\nimport sys"

def test_sorted_imports_with_section_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        forced_separate=[],
    )
    config = Config(


# LLM-generated content at query #20
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


# LLM-generated content at query #21
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
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n"


# LLM-generated content at query #22
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
    assert result == ["from module import *  # star_comment"]

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

def test_with_from_imports_combine_as_imports():
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


# LLM-generated content at query #23
#--------------------------

```python
def test_config_no_sections_false():
    config = Config(no_sections=False)
    assert not config.no_sections


# LLM-generated content at query #24
#--------------------------

```python
def test_no_sections_predicate_false():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[],
        line_separator='\n',
        imports={},
        place_imports={},
        import_placements={},
        sections=[],
        original_line_count=0
    )
    config = Config(no_sections=False)
    assert not config.no_sections


# LLM-generated content at query #25
#--------------------------

```python
def test_while_loop_removes_empty_lines_at_start():
    parsed = parse.ParsedContent(
        import_index=0,
        original_line_count=5,
        lines_without_imports=["", "", "code", "more code"],
        line_separator="\n",
        imports={},
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result.startswith("code")


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_evaluates_to_false():
    parsed = parse.ParsedContent(
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        imports={"STDLIB": {"straight": {}}},
    )
    config = Config(combine_straight_imports=True)
    straight_modules = ["module1", "module2"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == []


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
        categorized_comments={
            "above": {"straight": {"os": ["# OS module"]}},
            "straight": {"os": ["# For OS operations"]},
        },
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(ignore_comments=False)
    result = sorted_imports(parsed, config)
    assert result == "# OS module\nimport os  # For OS operations\n"

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
    result = sorted_imports(parsed, config)
    assert result == "import sys\n"

def test_sorted_imports_no_sections():
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
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\nimport os\n"

def test_sorted_imports_from_first():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "STDLIB": {
                "straight": {"os": set()},
                "from": {"sys": {"argv": set()}},
            }
        },
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from sys import argv\nimport os\n"

def test_sorted_imports_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "STDLIB": {
                "straight": {"os": set(), "sys": set()},
                "from": {"json": {"load": set()}},
            }
        },
        categorized_comments={
            "above": {"straight": {}, "from": {}},
            "straight": {},
            "from": {},
        },
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from json import load\nimport os\nimport sys\n"


# LLM-generated content at query #28
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        imports={},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=[],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

def test_sorted_imports_single_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\n"

def test_sorted_imports_multiple_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# Comment above"]}, "from": {}}, "straight": {"os": ["# Inline comment"]}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "# Comment above\nimport os  # Inline comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {"os": ["path"]}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os as path\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_no_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}, "THIRDPARTY": {"straight": {"numpy": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB", "THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import numpy\nimport os\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"sys": set(), "os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"stdlib": "Standard Library"})
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library\nimport os\n"


# LLM-generated content at query #29
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment1", "comment2"]}},
        as_map={"from": {"os.path": [], "os.sys": []}},
        line_separator="\n",
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path, sys  # comment1; comment2"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": [], "os.sys": []}},
        line_separator="\n",
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", ["os.path"], "import")
    assert result == ["from os import sys  # comment"]

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"nested": {"os": {"*": ["star comment"]}}},
        as_map={"from": {}},
        line_separator="\n",
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_as_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path as ospath  # comment"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": [], "os.sys": []}},
        line_separator="\n",
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path", "from os import sys"]

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]

def test_with_from_imports_no_inline_sort():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["sys", "path"]}}},
        categorized_comments={"from": {"os": []}},
        as_map={"from": {"os.sys": [], "os.path": []}},
        line_separator="\n",
    )
    config = Config(no_inline_sort=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import sys, path"]

def test_with_from_imports_only_sections():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {"os.path": []}},
        line_separator="\n",
    )
    config = Config(only_sections=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # comment"]


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_at_line_166_evaluates_to_true():
    parsed = parse.ParsedContent(
        lines_without_imports=["line1", "line2"],
        original_line_count=2,
        import_index=0,
        imports={"section": {"straight": {}, "from": {}}},
        place_imports={},
        import_placements={},
        line_separator="\n",
        sections=[],
        place_imports={},
    )
    config = Config()
    extension = "py"
    import_type = "import"
    output = sorted_imports(parsed, config, extension, import_type)
    assert output == "\n".join(["line1", "line2"])


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_153_evaluates_to_false():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        line_separator="\n",
        imports={},
        place_imports={},
        original_line_count=1,
        import_placements={},
    )
    config = Config()
    extension = "py"
    import_type = "import"
    result = sorted_imports(parsed, config, extension, import_type)
    assert result == ""


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
def test_for_loop_iterates_over_sections():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[],
        sections=["FUTURE", "STDLIB"],
        imports={"FUTURE": {"straight": {}, "from": {}}, "STDLIB": {"straight": {}, "from": {}}},
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config()
    sections = itertools.chain(parsed.sections, config.forced_separate)
    for section in sections:
        assert section in ["FUTURE", "STDLIB"]


# LLM-generated content at query #34
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        imports={},
        categorized_comments={},
        as_map={},
        sections=[],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

def test_sorted_imports_single_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\n"

def test_sorted_imports_multiple_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"os": ["# comment"]}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os  # comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["os_path"]}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os as os_path\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"THIRDPARTY": {"straight": {"sys": set(), "os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
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
    config = Config(lines_between_sections=1)
    result = sorted_imports(parsed, config)
    assert result == "import __future__\n\nimport os\n"

def test_sorted_imports_with_ensure_newline_before_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\n"

def test_sorted_imports_with_lines_after_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nprint('hello')"

def test_sorted_imports_with_formatting_function():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
    )
    config = Config(formatting_function=lambda


# LLM-generated content at query #35
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


# LLM-generated content at query #36
#--------------------------

```python
def test_sorted_imports_empty_parsed_content():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        imports={},
        as_map={"straight": {}, "from": {}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=[],
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
        as_map={"straight": {}, "from": {}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=[],
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')\n"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {}, "from": {}},
        categorized_comments={"above": {"straight": {"os": []}, "from": {}}, "straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    result = sorted_imports(parsed)
    assert result == "import os\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        as_map={"straight": {}, "from": {}},
        categorized_comments={"above": {"straight": {"os": [], "sys": []}, "from": {}}, "straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    result = sorted_imports(parsed)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {}, "from": {}},
        categorized_comments={"above": {"straight": {"os": ["# Comment above"]}, "from": {}}, "straight": {"os": ["# Inline comment"]}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    result = sorted_imports(parsed)
    assert result == "# Comment above\nimport os  # Inline comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {"os": ["path"]}, "from": {}},
        categorized_comments={"above": {"straight": {"os": []}, "from": {}}, "straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    result = sorted_imports(parsed)
    assert result == "import os as path\n"

def test_sorted_imports_with_remove_imports():
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        as_map={"straight": {}, "from": {}},
        categorized_comments={"above": {"straight": {"os": [], "sys": []}, "from": {}}, "straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    result = sorted_imports(parsed, config)
    assert result == "import sys\n"

def test_sorted_imports_with_combine_straight_imports():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        as_map={"straight": {}, "from": {}},
        categorized_comments={"above": {"straight": {"os": [], "sys": []}, "from": {}}, "straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {}, "from": {"os": {"path": set()}}}},
        as_map={"straight": {}, "from": {}},
        categorized_comments={"above": {"straight": {}, "from": {"os": []}}, "straight": {}, "from": {"os": []}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    result = sorted_imports(parsed)
    assert result == "from os import path\n"

def test_sorted_imports_with_section_headings():
    config = Config(import_headings={"stdlib": "Standard Library"})
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        as_map={"straight": {}, "from": {}},
        categorized_comments={"above": {"straight": {"os": []}, "from": {}}, "straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library\nimport os\n"

def test_sorted_imports_with_force_sort_within_sections():
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"sys": set(), "os": set()}, "from": {}}},
        as_map={"straight": {}, "from": {}},
        categorized_comments={"above": {"straight": {"sys": [], "os": []}, "from": {}}, "straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_lines_between_sections():
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}, "THIRDPARTY": {"straight": {"django": set()}, "from": {}}},
        as_map={"straight": {}, "from": {}},
        categorized_comments={"above": {"straight": {"os": [], "django": []}, "from": {}}, "straight": {}, "from": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB", "THIRDPARTY"],
    )
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nimport django\n"

def test_sorted_imports_with_ensure_newline_before_comments():
    config = Config(ensure_newline_before_comments=True)
    parsed = parse.ParsedContent(
        lines_without


# LLM-generated content at query #37
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
        lines_without_imports=["print('hello')"],
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
    assert result == "import os\n\nprint('hello')"

def test_sorted_imports_multiple_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
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
    assert result == "import os\nimport sys\n\nprint('hello')"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
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
    assert result == "import os  # comment\n\nprint('hello')"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
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
    result = sorted_imports(parsed)
    assert result == "import os as path\n\nprint('hello')"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
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
    result = sorted_imports(parsed)
    assert result == "from os import path\n\nprint('hello')"

def test_sorted_imports_with_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
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
    result = sorted_imports(parsed)
    assert result == "from __future__ import absolute_import\n\nimport os\n\nprint('hello')"

def test_sorted_imports_with_config():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
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
    result = sorted_imports(parsed, config)
    assert result == "import os, sys\n\nprint('hello')"

def test_sorted_imports_with_remove_imports():
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
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
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\nprint('hello')"

def test_sorted_imports_with_lines_between_sections():
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
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
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\n\n\n\nimport os\n\nprint('hello')"

def test_sorted_imports_with_import_headings():
    config = Config(import_headings={"stdlib": "Standard Library"})
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
        original_line_count=1,
    )
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library\nimport os\n\nprint('hello')"


# LLM-generated content at query #38
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
    config = Config(ignore_comments=False, comment_prefix="# ")
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["# comment1; comment2", "from os import path"]

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

def test__with_from_imports_star_import():
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

def test__with_from_imports_star_import_with_comment():
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
    config = Config(ignore_comments=False, comment_prefix="# ")
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test__with_from_imports_no_inline_sort():
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

def test__with_from_imports_combine_star():
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

def test__with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={
            "from": {"os": ["comment1"]},
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


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_162():
    parsed = parse.ParsedContent(
        import_index=5,
        original_line_count=10,
        lines_without_imports=[],
        line_separator="\n",
        sections=[],
        imports={},
        place_imports={},
        import_placements={}
    )
    config = Config()
    assert parsed.import_index < parsed.original_line_count


# LLM-generated content at query #40
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
    result = sorted_imports(parsed)
    assert result == "import os\n"

def test_sorted_imports_multiple_straight_imports():
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
    result = sorted_imports(parsed)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_combine_straight_imports():
    config = Config(combine_straight_imports=True)
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
    result = sorted_imports(parsed, config=config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_comments():
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
    result = sorted_imports(parsed)
    assert result == "import os  # comment\n"

def test_sorted_imports_with_as_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["os_module"]}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed)
    assert result == "import os as os_module\n"

def test_sorted_imports_with_section_heading():
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
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
    result = sorted_imports(parsed, config=config)
    assert result == "# Third Party Imports\nimport os\n"

def test_sorted_imports_with_force_sort_within_sections():
    config = Config(force_sort_within_sections=True)
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
    result = sorted_imports(parsed, config=config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_lines_after_imports():
    config = Config(lines_after_imports=2)
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
    result = sorted_imports(parsed, config=config)
    assert result == "import os\n\n\nprint('hello')"

def test_sorted_imports_with_remove_imports():
    config = Config(remove_imports=["os"])
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
    result = sorted_imports(parsed, config=config)
    assert result == "import sys\n"

def test_sorted_imports_with_no_sections():
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}, "FUTURE": {"straight": {"__future__": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["FUTURE", "THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    result = sorted_imports(parsed, config=config)
    assert result == "from __future__ import absolute_import\nimport os\n"


# LLM-generated content at query #41
#--------------------------

```python
def test_with_straight_imports_predicate_false():
    parsed = parse.ParsedContent(
        as_map={"straight": {"module1": ["alias1"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        imports={"STDLIB": {"straight": {"module1": ["import1"]}}}
    )
    config = Config(combine_straight_imports=True)
    straight_modules = ["module1"]
    section = "STDLIB"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)
    assert result == []


# LLM-generated content at query #42
#--------------------------

```python
def test_formatting_function_applied():
    config = Config(formatting_function=lambda x, y, z: x.replace("\n", "\r\n"))
    parsed = parse.ParsedContent(
        lines_without_imports=["import sys", "import os"],
        imports={"STDLIB": {"straight": {"sys": [], "os": []}, "from": {}}},
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    result = sorted_imports(parsed, config, "py", "import")
    assert "\r\n" in result


# LLM-generated content at query #43
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) or config.only_sections


# LLM-generated content at query #44
#--------------------------

```python
def test_with_star_comments_when_star_comment_exists():
    parsed = parse.ParsedContent(
        categorized_comments={
            "nested": {
                "test_module": {
                    "*": "This is a star comment"
                }
            }
        }
    )
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, "test_module", comments)
    assert result == ["comment1", "comment2", "This is a star comment"]

def test_with_star_comments_when_star_comment_does_not_exist():
    parsed = parse.ParsedContent(
        categorized_comments={
            "nested": {
                "test_module": {}
            }
        }
    )
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, "test_module", comments)
    assert result == ["comment1", "comment2"]

def test_with_star_comments_when_module_does_not_exist():
    parsed = parse.ParsedContent(
        categorized_comments={
            "nested": {}
        }
    )
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, "non_existent_module", comments)
    assert result == ["comment1", "comment2"]


# LLM-generated content at query #45
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
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"


# LLM-generated content at query #46
#--------------------------

```python
def test_no_star_comment_returns_original_comments():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"test_module": {}}}
    module = "test_module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == comments


# LLM-generated content at query #47
#--------------------------

```python
def test_predicate_at_line_153_evaluates_to_false():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[""],
        imports={},
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n"


# LLM-generated content at query #48
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

def test_sorted_imports_single_straight_import():
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

def test_sorted_imports_multiple_straight_imports():
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

def test_sorted_imports_with_as_import():
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
    assert sorted_imports(parsed, config) == "import os\nimport os as path\n"

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
    assert sorted_imports(parsed, config) == "# comment above\nimport os  # inline comment\n"

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

def test_sorted_imports_no_sections():
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
    config = Config(no_sections=True)
    assert sorted_imports(parsed, config) == "import numpy\nimport os\n"

def test_sorted_imports_from_first():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {"sys": {"argv"}}}},
        categorized_comments={"above": {"straight": {}, "from": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}, "from": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(from_first=True)
    assert sorted_imports(parsed, config) == "from sys import argv\n\nimport os\n"

def test_sorted_imports_force_sort_within_sections():
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

def test_sorted_imports_import_headings():
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

def test_sorted_imports_lines_before_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=1,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(lines_before_imports=2)
    assert sorted_imports(parsed, config) == "\n\nimport os\n\nprint('hello')"

def test_sorted_imports_lines_after_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "


# LLM-generated content at query #49
#--------------------------

```python
def test_sorted_imports_predicate():
    assert sorted_imports(parsed=parse.ParsedContent(import_index=0, lines_without_imports=[], line_separator="\n"), config=DEFAULT_CONFIG, extension="py", import_type="import") is not None


# LLM-generated content at query #50
#--------------------------

```python
def test__with_from_imports_basic_case():
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

def test__with_from_imports_with_comments():
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
    config = Config(ignore_comments=False)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path  # comment"]

def test__with_from_imports_remove_imports():
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
    assert result == ["from os import path", "from os import path as ospath"]

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

def test__with_from_imports_star_import():
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
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test__with_from_imports_combine_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*", "path"]}}},
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

def test__with_from_imports_no_inline_sort():
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

def test__with_from_imports_ignore_comments():
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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    config = Config(ignore_comments=False, comment_prefix="# ")
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

def test_with_from_imports_with_nested_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"module": {"import1": ["nested comment"]}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False, comment_prefix="# ")
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1  # nested comment", "from module import import2"]

def test_with_from_imports_with_above_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {"module": ["above comment"]}},
            "nested": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["above comment", "from module import import1, import2"]

def test_with_from_imports_with_straight_comments():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {},
            "straight": {"module.import1": ["straight comment"]},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1  # straight comment", "from module import import2"]

def test_with_from_imports_with_noqa_comment():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"module": {"import1": ["noqa: F401"]}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(
        multi_line_output=wrap.Modes.HANGING_INDENT,
        ignore_comments=False,
        comment_prefix="# ",
    )
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import import1, import2  # noqa: F401"]


# LLM-generated content at query #2
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
    config = Config(ignore_comments=False)
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

def test_with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["*", "import1"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {"module": {"*": ["star_comment"]}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import *  # star_comment", "from module import import1"]

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


# LLM-generated content at query #3
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
        place_imports={},
        import_placements={},
        sections=[],
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
        categorized_comments={"above": {"straight": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
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
        categorized_comments={"above": {"straight": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
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
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"os": ["# comment"]},
            "from": {},
        },
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os  # comment\n"

def test_sorted_imports_with_as_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}, "from": {}},
        as_map={"straight": {"os": {"alias"}}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os as alias\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
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
        categorized_comments={"above": {"straight": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
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
        imports={
            "FUTURE": {"straight": {"__future__": set()}, "from": {}},
            "STDLIB": {"straight": {"os": set()}, "from": {}},
        },
        categorized_comments={"above": {"straight": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["FUTURE", "STDLIB"],
        original_line_count=1,
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\nimport os\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
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
        categorized_comments={"above": {"straight": {}}, "straight": {}, "from": {}},
        as_map={"straight": {}},
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
        original_line_count=1,
    )
    config = Config(import_headings={"stdlib": "Standard Library"})
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library\nimport os\n"


# LLM-generated content at query #4
#--------------------------

```python
def test_with_star_comments_when_star_comment_exists():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"module": {"*": "star_comment"}}}
    module = "module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2", "star_comment"]

def test_with_star_comments_when_star_comment_does_not_exist():
    parsed = parse.ParsedContent()
    parsed.categorized_comments = {"nested": {"module": {}}}
    module = "module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2"]


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #6
#--------------------------

```python
def test_with_straight_imports_empty_modules():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, [], "standard", [], "import") == []

def test_with_straight_imports_combined_no_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import") == ["import sys, os"]

def test_with_straight_imports_combined_with_inline_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"sys": ["comment1"], "os": ["comment2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import") == ["import sys, os  # comment1 comment2"]

def test_with_straight_imports_combined_with_above_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {"sys": ["# above comment"]}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import") == ["# above comment", "import sys, os"]

def test_with_straight_imports_combined_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"sys": ["s"]}},
    )
    config = Config(combine_straight_imports=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import") == ["import os", "import sys as s"]

def test_with_straight_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "standard", ["sys"], "import") == ["import os"]

def test_with_straight_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"sys": ["comment1"], "os": ["comment2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False, ignore_comments=True)
    assert _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import") == ["import sys", "import os"]

def test_with_straight_imports_custom_comment_prefix():
    parsed = parse.ParsedContent(
        imports={"standard": {"straight": {"sys": [], "os": []}}},
        categorized_comments={"above": {"straight": {}}, "straight": {"sys": ["comment1"], "os": ["comment2"]}},
        as_map={"straight": {}},
    )
    config = Config(combine_straight_imports=False, comment_prefix=" # ")
    assert _with_straight_imports(parsed, config, ["sys", "os"], "standard", [], "import") == ["import sys # comment1", "import os # comment2"]


# LLM-generated content at query #7
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

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={
            "from": {"os": ["# comment"]},
            "above": {"from": {}},
            "nested": {"os": {"path": "# path comment"}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path, sys  # comment"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
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
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {"os.path": ["os.path as osp"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path as osp"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
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
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {"os": {"*": "# star comment"}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_combine_as_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*", "path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {"os": {"*": "# star comment"}}},
        as_map={"from": {"os.path": ["os.path as osp"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True, combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # star comment", "from os import path as osp"]

def test_with_from_imports_no_inline_sort():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["sys", "path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(no_inline_sort=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import sys, path"]

def test_with_from_imports_force_alphabetical_sort():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["sys", "path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_alphabetical_sort_within_sections=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path, sys"]

def test_with_from_imports_reverse_sort():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(reverse_sort=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import sys, path"]


# LLM-generated content at query #8
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["comment"]}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path; comment"]


# LLM-generated content at query #9
#--------------------------

```python
def test__with_from_imports_basic_case():
    parsed = parse.ParsedContent(
        imports={
            "THIRDPARTY": {
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
    result = _with_from_imports(parsed, config, ["module"], "THIRDPARTY", [], "import")
    assert result == ["from module import import1, import2"]


# LLM-generated content at query #10
#--------------------------

```python
def test_sorted_imports_basic_case():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"os": ["# Comment for os"], "sys": ["# Comment for sys"]},
        },
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(ignore_comments=False)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os  # Comment for os\nimport sys  # Comment for sys\n"

def test_sorted_imports_with_remove_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(remove_imports=["sys"])
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\n"

def test_sorted_imports_with_combine_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(combine_straight_imports=True)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os, sys\n"

def test_sorted_imports_with_no_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        imports={
            "STDLIB": {"straight": {"os": set()}},
            "THIRDPARTY": {"straight": {"sys": set()}},
        },
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB", "THIRDPARTY"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys\n"

def test_sorted_imports_with_force_sort_within_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys\n"

def test_sorted_imports_with_import_headings():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(import_headings={"stdlib": "Standard Library"})
    result = sorted_imports(parsed, config)
    assert result == "\n# Standard Library\nimport os\nimport sys\n"

def test_sorted_imports_with_import_footers():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set(), "sys": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(import_footers={"stdlib": "End of Standard Library"})
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys\n\n# End of Standard Library\n"

def test_sorted_imports_with_lines_between_sections():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        imports={
            "STDLIB": {"straight": {"os": set()}},
            "THIRDPARTY": {"straight": {"sys": set()}},
        },
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB", "THIRDPARTY"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\n\n\nimport sys\n"

def test_sorted_imports_with_lines_after_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        import_index=0,
        original_line_count=3,
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\n\n\ndef main():\n    pass\n"

def test_sorted_imports_with_formatting_function():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={},
        import_placements={},
        line_separator="\n",
    )
    config = Config(formatting_function=lambda x, y, z: x.upper())
    result = sorted_imports(parsed, config)
    assert result == "\nIMPORT OS\n"

def test_sorted_imports_with_place_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["", "# Placeholder"],
        import_index=0,
        original_line_count=2,
        imports={"STDLIB": {"straight": {"os": set()}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["STDLIB"],
        place_imports={"STDLIB": ["import sys"]},
        import_placements={"# Placeholder": "STDLIB"},
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\n\n# Placeholder\nimport sys\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        original_line_count=1,
        imports={"STDLIB": {"straight": {"os": set()}}


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_1():
    parsed = parse.ParsedContent()
    config = Config()
    from_modules = ["module1"]
    section = "SECTION"
    remove_imports = []
    import_type = "import"

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)
    assert isinstance(result, list)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    config = Config(ignore_comments=False)
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
    result = _with_from_imports(parsed, config, ["module"], "section", ["import1"], "import")
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

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["*"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {"module": {"*": ["star comment"]}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert result == ["from module import *  # star comment"]

def test_with_from_imports_multiline_reformat():
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2", "import3"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(line_length=20)
    result = _with_from_imports(parsed, config, ["module"], "section", [], "import")
    assert len(result) > 1


# LLM-generated content at query #2
#--------------------------

```python
def test_empty_list():
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

def test_multiple_comments_with_code():
    assert _ensure_newline_before_comment(["code1", "# comment1", "code2", "# comment2"]) == ["code1", "", "# comment1", "code2", "", "# comment2"]

def test_no_newline_needed():
    assert _ensure_newline_before_comment(["# comment1", "code", "# comment2"]) == ["# comment1", "code", "", "# comment2"]

def test_mixed_empty_and_non_empty_lines():
    assert _ensure_newline_before_comment(["", "code", "# comment", "", "code"]) == ["", "code", "", "# comment", "", "code"]


# LLM-generated content at query #3
#--------------------------

```python
def test_sorted_imports_no_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')"

def test_sorted_imports_with_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_combine_straight_imports():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import os, sys\n"

def test_sorted_imports_with_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": {"path": set()}},
            },
        },
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed)
    assert result == "from os import path\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
        categorized_comments={
            "above": {"straight": {"os": ["# Comment above os"]}},
            "straight": {"os": ["# Inline comment for os"]},
        },
    )
    result = sorted_imports(parsed)
    assert result == "# Comment above os\nimport os  # Inline comment for os\n"

def test_sorted_imports_with_force_sort_within_sections():
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"sys": set(), "os": set()},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_import_headings():
    config = Config(import_headings={"thirdparty": "Third Party"})
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed, config=config)
    assert result == "# Third Party\nimport os\n"

def test_sorted_imports_with_lines_between_sections():
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "STDLIB": {
                "straight": {"os": set()},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"sys": set()},
                "from": {},
            },
        },
        sections=["STDLIB", "THIRDPARTY"],
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import os\n\n\nimport sys\n"

def test_sorted_imports_with_ensure_newline_before_comments():
    config = Config(ensure_newline_before_comments=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import os\n"

def test_sorted_imports_with_formatting_function():
    def custom_format(content: str, extension: str, config: Config) -> str:
        return content.upper()

    config = Config(formatting_function=custom_format)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed, config=config)
    assert result == "IMPORT OS\n"

def test_sorted_imports_with_remove_imports():
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import sys\n"

def test_sorted_imports_with_no_sections():
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "STDLIB": {
                "straight": {"os": set()},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"sys": set()},
                "from": {},
            },
        },
        sections=["STDLIB", "THIRDPARTY"],
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import os\nimport sys\n"

def test_sorted_imports_with_only_sections():
    config = Config(only_sections=["THIRDPARTY"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "STDLIB": {
                "straight": {"os": set()},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"sys": set()},
                "from": {},
            },
        },
        sections=["STDLIB", "THIRDPARTY"],
    )
    result = sorted_imports(parsed, config=config)
    assert result == "import sys\n"

def test_sorted_imports_with_from_first():
    config = Config(from_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"sys": {"path": set()}},
            },
        },
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed, config=config)
    assert result == "from sys import path\nimport os\n"

def test_sorted_imports_with_star_first():
    config = Config(star_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": {"*": set(), "path": set()}},
            },
        },
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed, config=config)
    assert result == "from os import *\nfrom os import path\n"

def test_sorted_imports_with_lines_after_imports():
    config = Config(lines_after_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    result = sorted_imports(


# LLM-generated content at query #4
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
    result = sorted_imports(parsed)
    assert result == "print('hello')"


# LLM-generated content at query #5
#--------------------------

```python
def test_with_straight_imports_empty_modules():
    parsed = parse.ParsedContent(
        imports={"STD_LIB": {"straight": {}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, [], "STD_LIB", [], "import")
    assert result == []

def test_with_straight_imports_combine_no_as():
    parsed = parse.ParsedContent(
        imports={"STD_LIB": {"straight": {"sys": [], "os": []}}},
        as_map={"straight": {}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"sys": ["comment1"], "os": ["comment2"]},
        },
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "STD_LIB", [], "import")
    assert result == ["import sys, os  # comment1 comment2"]

def test_with_straight_imports_combine_with_as():
    parsed = parse.ParsedContent(
        imports={"STD_LIB": {"straight": {"sys": [], "os": []}}},
        as_map={"straight": {"sys": ["s"]}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=True)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "STD_LIB", [], "import")
    assert result == ["import sys as s", "import os"]

def test_with_straight_imports_no_combine():
    parsed = parse.ParsedContent(
        imports={"STD_LIB": {"straight": {"sys": [], "os": []}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "STD_LIB", [], "import")
    assert result == ["import sys", "import os"]

def test_with_straight_imports_above_comments():
    parsed = parse.ParsedContent(
        imports={"STD_LIB": {"straight": {"sys": []}}},
        as_map={"straight": {}},
        categorized_comments={
            "above": {"straight": {"sys": ["# above comment"]}},
            "straight": {},
        },
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["sys"], "STD_LIB", [], "import")
    assert result == ["# above comment", "import sys"]

def test_with_straight_imports_inline_comments():
    parsed = parse.ParsedContent(
        imports={"STD_LIB": {"straight": {"sys": []}}},
        as_map={"straight": {}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"sys": ["inline comment"]},
        },
    )
    config = Config(combine_straight_imports=False, comment_prefix=" # ")
    result = _with_straight_imports(parsed, config, ["sys"], "STD_LIB", [], "import")
    assert result == ["import sys # inline comment"]

def test_with_straight_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"STD_LIB": {"straight": {"sys": [], "os": []}}},
        as_map={"straight": {}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
    )
    config = Config(combine_straight_imports=False)
    result = _with_straight_imports(parsed, config, ["sys", "os"], "STD_LIB", ["sys"], "import")
    assert result == ["import os"]

def test_with_straight_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STD_LIB": {"straight": {"sys": []}}},
        as_map={"straight": {}},
        categorized_comments={
            "above": {"straight": {}},
            "straight": {"sys": ["inline comment"]},
        },
    )
    config = Config(combine_straight_imports=False, ignore_comments=True)
    result = _with_straight_imports(parsed, config, ["sys"], "STD_LIB", [], "import")
    assert result == ["import sys"]


# LLM-generated content at query #6
#--------------------------

```python
def test_with_from_imports_basic_case():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {"os": []}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path, sys"]


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_evaluates_to_false():
    config = Config(
        no_inline_sort=True,
        force_single_line=False,
        single_line_exclusions=[],
        only_sections=True
    )
    parsed = parse.ParsedContent(
        imports={"section": {"from": {"module": ["import1", "import2"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "straight": {}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set()
    )
    from_modules = ["module"]
    section = "section"
    remove_imports = []
    import_type = "import"

    result = _with_from_imports(parsed, config, from_modules, section, remove_imports, import_type)

    assert not (
        not config.no_inline_sort
        or (config.force_single_line and "module" not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #8
#--------------------------

```python
def test_as_imports_predicate_with_empty_straight_modules():
    parsed = parse.ParsedContent(as_map={"straight": {}}, categorized_comments={"above": {"straight": {}}, "straight": {}}, imports={"straight": {}})
    config = Config(combine_straight_imports=True)
    straight_modules = []
    section = "straight"
    remove_imports = []
    import_type = "import"

    result = _with_straight_imports(parsed, config, straight_modules, section, remove_imports, import_type)

    assert result == []


# LLM-generated content at query #9
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
    assert sorted_imports(parsed) == "print('hello')"

def test_sorted_imports_single_straight_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    assert sorted_imports(parsed) == "import os\n"

def test_sorted_imports_single_from_import():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {}, "from": {"os": {"path": set()}}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    assert sorted_imports(parsed) == "from os import path\n"

def test_sorted_imports_multiple_straight_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    assert sorted_imports(parsed) == "import os\nimport sys\n"

def test_sorted_imports_multiple_from_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {}, "from": {"os": {"path": set()}, "sys": {"argv": set()}}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    assert sorted_imports(parsed) == "from os import path\nfrom sys import argv\n"

def test_sorted_imports_with_comments():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {"os": ["# Comment above"]}}, "straight": {"os": ["# Inline comment"]}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    assert sorted_imports(parsed) == "# Comment above\nimport os  # Inline comment\n"

def test_sorted_imports_with_as_imports():
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {"os": ["ospath"]}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    assert sorted_imports(parsed) == "import os as ospath\n"

def test_sorted_imports_with_config():
    config = Config(combine_straight_imports=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    assert sorted_imports(parsed, config) == "import os, sys\n"

def test_sorted_imports_with_remove_imports():
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    assert sorted_imports(parsed, config) == "import sys\n"

def test_sorted_imports_with_import_headings():
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    assert sorted_imports(parsed, config) == "# Third Party Imports\nimport os\n"

def test_sorted_imports_with_import_footers():
    config = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    assert sorted_imports(parsed, config) == "import os\n\n# End of Third Party Imports\n"

def test_sorted_imports_with_lines_between_sections():
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        import_index=0,
        line_separator="\n",
        imports={"FUTURE": {"straight": {"__future__": set()}, "from": {}}, "THIRDPARTY": {"straight": {"os": set()}, "from": {}}},
        categorized_comments={"above": {"straight": {}}, "straight": {}},
        as_map={"straight": {}},
        sections=["FUTURE", "THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=0,
    )
    assert sorted_imports(parsed, config) == "import __future__\n\n\nimport os\n"

def test_sorted_imports_with_lines_after_imports():
    config = Config(lines_after_imports=2)
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
        original_line_count=1,
    )
    assert sorted_imports(parsed, config) == "import os\n\n\nprint('hello')"

def test_sorted_imports_with_lines_before_imports():
    config = Config(lines_before_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_1():
    assert not (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) or config.only_sections


# LLM-generated content at query #12
#--------------------------

```python
def test_with_star_comments_when_star_comment_exists():
    parsed = parse.ParsedContent(categorized_comments={"nested": {module: {"*": "star_comment"}}})
    module = "test_module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2", "star_comment"]

def test_with_star_comments_when_star_comment_does_not_exist():
    parsed = parse.ParsedContent(categorized_comments={"nested": {module: {}}})
    module = "test_module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2"]

def test_with_star_comments_when_module_does_not_exist():
    parsed = parse.ParsedContent(categorized_comments={"nested": {}})
    module = "test_module"
    comments = ["comment1", "comment2"]
    result = _with_star_comments(parsed, module, comments)
    assert result == ["comment1", "comment2"]


# LLM-generated content at query #13
#--------------------------

```python
def test_with_from_imports_basic():
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

def test_with_from_imports_multiple_imports():
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
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["# comment"]}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False)
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

def test_with_from_imports_with_star():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["*"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}, "nested": {"os": {"*": ["# all"]}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import *  # all"]

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

def test_with_from_imports_ignore_comments():
    parsed = parse.ParsedContent(
        imports={"STDLIB": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {"os": ["# comment"]}, "above": {"from": {}}, "nested": {}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=True)
    result = _with_from_imports(parsed, config, ["os"], "STDLIB", [], "import")
    assert result == ["from os import path"]


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_1():
    assert (
        not config.no_inline_sort
        or (config.force_single_line and module not in config.single_line_exclusions)
    ) and not config.only_sections


# LLM-generated content at query #15
#--------------------------

```python
def test_ensure_newline_before_comments_config_enabled():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=["# Comment"],
        original_line_count=1,
        imports={},
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "\n# Comment"


# LLM-generated content at query #16
#--------------------------

```python
def test_with_from_imports_basic():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "standard", [], "import")
    assert result == ["from os import path, sys"]

def test_with_from_imports_with_comments():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={
            "from": {"os": ["# comment"]},
            "above": {"from": {}},
            "nested": {},
            "straight": {},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(ignore_comments=False, comment_prefix="# ")
    result = _with_from_imports(parsed, config, ["os"], "standard", [], "import")
    assert result == ["from os import path, sys  # comment"]

def test_with_from_imports_remove_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "standard", ["os.path"], "import")
    assert result == ["from os import sys"]

def test_with_from_imports_with_as_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {"os.path": ["path as ospath"]}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_as_imports=True)
    result = _with_from_imports(parsed, config, ["os"], "standard", [], "import")
    assert result == ["from os import path as ospath"]

def test_with_from_imports_force_single_line():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(force_single_line=True)
    result = _with_from_imports(parsed, config, ["os"], "standard", [], "import")
    assert result == ["from os import path", "from os import sys"]

def test_with_from_imports_star_import():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["*"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"os": {"*": ["# star comment"]}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(combine_star=True)
    result = _with_from_imports(parsed, config, ["os"], "standard", [], "import")
    assert result == ["from os import *  # star comment"]

def test_with_from_imports_multiline_reformat():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path", "sys", "module"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(line_length=20)
    result = _with_from_imports(parsed, config, ["os"], "standard", [], "import")
    assert len(result) > 1

def test_with_from_imports_split_on_trailing_comma():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas={"os"},
    )
    config = Config(split_on_trailing_comma=True)
    result = _with_from_imports(parsed, config, ["os"], "standard", [], "import")
    assert result == ["from os import (\n    path,\n    sys,\n)"]

def test_with_from_imports_noqa_comment():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": ["path", "sys"]}}},
        categorized_comments={
            "from": {},
            "above": {"from": {}},
            "nested": {"os": {"sys": ["# noqa: F401"]}},
        },
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config(multi_line_output=wrap.Modes.HANGING_INDENT)
    result = _with_from_imports(parsed, config, ["os"], "standard", [], "import")
    assert result == ["from os import path, sys  # noqa: F401"]

def test_with_from_imports_empty_from_imports():
    parsed = parse.ParsedContent(
        imports={"standard": {"from": {"os": []}}},
        categorized_comments={"from": {}, "above": {"from": {}}},
        as_map={"from": {}},
        line_separator="\n",
        trailing_commas=set(),
    )
    config = Config()
    result = _with_from_imports(parsed, config, ["os"], "standard", [], "import")
    assert result == []


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    parsed = parse.ParsedContent(
        import_index=0,
        lines_without_imports=[],
        imports={},
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = Config(no_sections=True, forced_separate=[])
    extension = "py"
    import_type = "import"
    result = sorted_imports(parsed, config, extension, import_type)
    assert result == "\n"


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
def test_predicate_at_line_162_evaluates_to_false():
    parsed = parse.ParsedContent(
        import_index=5,
        original_line_count=3,
        lines_without_imports=[],
        line_separator="\n",
        sections=[],
        imports={},
        place_imports={},
        import_placements={}
    )
    config = Config()
    extension = "py"
    import_type = "import"
    assert not (parsed.import_index < parsed.original_line_count)


# LLM-generated content at query #20
#--------------------------

```python
def test_config_no_sections_false():
    config = Config(no_sections=False)
    assert not config.no_sections


