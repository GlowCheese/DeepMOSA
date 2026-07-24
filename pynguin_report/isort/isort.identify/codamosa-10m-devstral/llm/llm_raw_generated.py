####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Import___str__():
    # Test with file_path
    import_obj = Import(
        line_number=10,
        indented=True,
        module="os",
        attribute="path",
        alias="osp",
        cimport=False,
        file_path=Path("/path/to/file.py")
    )
    assert str(import_obj) == "/path/to/file.py:10 indented from os import path as osp"

    # Test without file_path
    import_obj = Import(
        line_number=5,
        indented=False,
        module="sys",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=None
    )
    assert str(import_obj) == ":5 cimport sys"

    # Test with attribute but no alias
    import_obj = Import(
        line_number=15,
        indented=True,
        module="collections",
        attribute="defaultdict",
        alias=None,
        cimport=False,
        file_path=Path("/another/path.py")
    )
    assert str(import_obj) == "/another/path.py:15 indented from collections import defaultdict"

    # Test with module only (no attribute)
    import_obj = Import(
        line_number=20,
        indented=False,
        module="math",
        attribute=None,
        alias="m",
        cimport=False,
        file_path=None
    )
    assert str(import_obj) == ":20 import math as m"


# LLM-generated content at query #2
#--------------------------

```python
def test_Import___str__():
    # Test basic import without alias
    import_obj = Import(line_number=1, indented=False, module="os", file_path=None)
    assert str(import_obj) == ":1 import os"

    # Test import with alias
    import_obj = Import(line_number=2, indented=True, module="numpy", alias="np", file_path=None)
    assert str(import_obj) == ":2 indented import numpy as np"

    # Test from import with attribute
    import_obj = Import(line_number=3, indented=False, module="sys", attribute="path", file_path=None)
    assert str(import_obj) == ":3 from sys import path"

    # Test from import with attribute and alias
    import_obj = Import(line_number=4, indented=True, module="collections", attribute="defaultdict", alias="dd", file_path=None)
    assert str(import_obj) == ":4 indented from collections import defaultdict as dd"

    # Test cimport
    import_obj = Import(line_number=5, indented=False, module="cython", cimport=True, file_path=None)
    assert str(import_obj) == ":5 cimport cython"

    # Test with file path
    import_obj = Import(line_number=6, indented=False, module="sys", file_path=Path("/path/to/file.py"))
    assert str(import_obj) == "/path/to/file.py:6 import sys"


# LLM-generated content at query #3
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = iter(["import os\n", "import sys\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[1].module == "sys"

    # Test import with alias
    input_stream = iter(["import numpy as np\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias == "np"

    # Test from import
    input_stream = iter(["from os import path\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"

    # Test from import with alias
    input_stream = iter(["from os import path as p\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"
    assert imports_list[0].alias == "p"

    # Test cimport
    input_stream = iter(["cimport numpy as np\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias == "np"
    assert imports_list[0].cimport is True

    # Test multiline import
    input_stream = iter(["from os import (\n", "    path,\n", "    environ\n", ")\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"
    assert imports_list[1].module == "os"
    assert imports_list[1].attribute == "environ"

    # Test indented import
    input_stream = iter(["    import os\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].indented is True

    # Test with file path
    file_path = Path("/test/path.py")
    input_stream = iter(["import os\n"])
    imports_list = list(imports(input_stream, file_path=file_path))
    assert len(imports_list) == 1
    assert imports_list[0].file_path == file_path

    # Test top_only
    input_stream = iter(["import os\n", "def func():\n", "    import sys\n"])
    imports_list = list(imports(input_stream, top_only=True))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"

    # Test with comments
    input_stream = iter(["import os  # comment\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"

    # Test with redundant alias
    config = Config(remove_redundant_aliases=True)
    input_stream = iter(["import os as os\n"])
    imports_list = list(imports(input_stream, config=config))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].alias is None

    # Test with section comments
    config = Config(section_comments=["# isort: off", "# isort: on"])
    input_stream = iter(["# isort: off\n", "import os\n", "# isort: on\n", "import sys\n"])
    imports_list = list(imports(input_stream, config=config))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[1].module == "sys"


# LLM-generated content at query #4
#--------------------------

```python
def test_Import___str__():
    # Test basic import without alias
    import_obj = Import(line_number=1, indented=False, module="os")
    assert str(import_obj) == ":1 import os"

    # Test import with alias
    import_obj = Import(line_number=2, indented=True, module="numpy", alias="np")
    assert str(import_obj) == ":2 indented import numpy as np"

    # Test cimport
    import_obj = Import(line_number=3, indented=False, module="cython", cimport=True)
    assert str(import_obj) == ":3 cimport cython"

    # Test from import with attribute
    import_obj = Import(line_number=4, indented=False, module="os", attribute="path")
    assert str(import_obj) == ":4 from os import path"

    # Test from cimport with attribute and alias
    import_obj = Import(line_number=5, indented=True, module="libc", attribute="stdio", alias="cstdio", cimport=True)
    assert str(import_obj) == ":5 indented from libc cimport stdio as cstdio"

    # Test with file path
    import_obj = Import(line_number=6, indented=False, module="sys", file_path=Path("/path/to/file.py"))
    assert str(import_obj) == "/path/to/file.py:6 import sys"


# LLM-generated content at query #5
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="os")
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(line_number=2, indented=True, module="numpy", alias="np")
    assert import_obj.statement() == "import numpy as np"

    # Test from import without alias
    import_obj = Import(line_number=3, indented=False, module="collections", attribute="defaultdict")
    assert import_obj.statement() == "from collections import defaultdict"

    # Test from import with alias
    import_obj = Import(line_number=4, indented=True, module="typing", attribute="List", alias="TList")
    assert import_obj.statement() == "from typing import List as TList"

    # Test cimport without alias
    import_obj = Import(line_number=5, indented=False, module="cython", cimport=True)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=6, indented=True, module="libc", alias="lc", cimport=True)
    assert import_obj.statement() == "cimport libc as lc"

    # Test from cimport without alias
    import_obj = Import(line_number=7, indented=False, module="cython", attribute="cdivision", cimport=True)
    assert import_obj.statement() == "from cython cimport cdivision"

    # Test from cimport with alias
    import_obj = Import(line_number=8, indented=True, module="libc", attribute="stdio", alias="cstdio", cimport=True)
    assert import_obj.statement() == "from libc cimport stdio as cstdio"


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Import___str__():
    # Test basic import without alias
    import_obj = Import(1, False, "os", None, None, False, None)
    assert str(import_obj) == ":1 import os"

    # Test basic import with alias
    import_obj = Import(2, False, "numpy", None, "np", False, None)
    assert str(import_obj) == ":2 import numpy as np"

    # Test from import without alias
    import_obj = Import(3, False, "collections", "defaultdict", None, False, None)
    assert str(import_obj) == ":3 from collections import defaultdict"

    # Test from import with alias
    import_obj = Import(4, False, "collections", "defaultdict", "dd", False, None)
    assert str(import_obj) == ":4 from collections import defaultdict as dd"

    # Test cimport
    import_obj = Import(5, False, "cython", None, None, True, None)
    assert str(import_obj) == ":5 cimport cython"

    # Test indented import
    import_obj = Import(6, True, "sys", None, None, False, None)
    assert str(import_obj) == ":6 indented import sys"

    # Test with file path
    import_obj = Import(7, False, "os", None, None, False, Path("/path/to/file.py"))
    assert str(import_obj) == "/path/to/file.py:7 import os"


# LLM-generated content at query #2
#--------------------------

```python
def test_Import___str__():
    # Test with all attributes
    import_obj = Import(
        line_number=10,
        indented=True,
        module="os",
        attribute="path",
        alias="osp",
        cimport=False,
        file_path=Path("/tmp/test.py")
    )
    assert str(import_obj) == "/tmp/test.py:10 indented from os import path as osp"

    # Test without attribute and alias
    import_obj = Import(
        line_number=5,
        indented=False,
        module="sys",
        cimport=True,
        file_path=Path("/tmp/test.py")
    )
    assert str(import_obj) == "/tmp/test.py:5 cimport sys"

    # Test without file_path
    import_obj = Import(
        line_number=1,
        indented=False,
        module="math",
        attribute="sqrt",
        alias=None,
        cimport=False,
        file_path=None
    )
    assert str(import_obj) == ":1 from math import sqrt"

    # Test with no alias
    import_obj = Import(
        line_number=3,
        indented=True,
        module="collections",
        attribute="defaultdict",
        cimport=False,
        file_path=Path("/tmp/test.py")
    )
    assert str(import_obj) == "/tmp/test.py:3 indented from collections import defaultdict"

    # Test with straight import and alias
    import_obj = Import(
        line_number=7,
        indented=False,
        module="numpy",
        alias="np",
        cimport=False,
        file_path=Path("/tmp/test.py")
    )
    assert str(import_obj) == "/tmp/test.py:7 import numpy as np"


# LLM-generated content at query #3
#--------------------------

```python
def test_imports():
    # Test basic imports
    test_input = "import os\nimport sys\n"
    expected = [
        Import(1, False, "os"),
        Import(2, False, "sys"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test from imports
    test_input = "from os import path\nfrom sys import argv\n"
    expected = [
        Import(1, False, "os", "path"),
        Import(2, False, "sys", "argv"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test aliased imports
    test_input = "import numpy as np\nfrom pandas import DataFrame as df\n"
    expected = [
        Import(1, False, "numpy", alias="np"),
        Import(2, False, "pandas", "DataFrame", alias="df"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test cimports
    test_input = "cimport numpy\nfrom pandas cimport DataFrame\n"
    expected = [
        Import(1, False, "numpy", cimport=True),
        Import(2, False, "pandas", "DataFrame", cimport=True),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test indented imports
    test_input = "if True:\n    import os\n    from sys import argv\n"
    expected = [
        Import(2, True, "os"),
        Import(3, True, "sys", "argv"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test multiline imports
    test_input = "from os import (\n    path,\n    environ,\n)\n"
    expected = [
        Import(1, False, "os", "path"),
        Import(1, False, "os", "environ"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test redundant aliases
    config = Config(remove_redundant_aliases=True)
    test_input = "import numpy as numpy\nfrom pandas import DataFrame as DataFrame\n"
    expected = [
        Import(1, False, "numpy"),
        Import(2, False, "pandas", "DataFrame"),
    ]
    assert list(imports(test_input.splitlines(), config=config)) == expected

    # Test comments and skipping lines
    test_input = "# This is a comment\nimport os  # inline comment\n'''\nmultiline string\n'''\nimport sys\n"
    expected = [
        Import(2, False, "os"),
        Import(5, False, "sys"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test semicolon separated imports
    test_input = "import os; import sys\nfrom pandas import DataFrame; import numpy\n"
    expected = [
        Import(1, False, "os"),
        Import(1, False, "sys"),
        Import(2, False, "pandas", "DataFrame"),
        Import(2, False, "numpy"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test file path
    file_path = Path("/path/to/file.py")
    test_input = "import os\n"
    expected = [
        Import(1, False, "os", file_path=file_path),
    ]
    assert list(imports(test_input.splitlines(), file_path=file_path)) == expected

    # Test top_only
    test_input = "import os\ndef foo():\n    import sys\n"
    expected = [
        Import(1, False, "os"),
    ]
    assert list(imports(test_input.splitlines(), top_only=True)) == expected


# LLM-generated content at query #4
#--------------------------

```python
def test_Import___str__():
    # Test case 1: Basic import without alias
    import_obj = Import(line_number=1, indented=False, module="os", file_path=None)
    assert str(import_obj) == ":1 import os"

    # Test case 2: Import with alias
    import_obj = Import(line_number=2, indented=True, module="numpy", alias="np", file_path=None)
    assert str(import_obj) == ":2 indented import numpy as np"

    # Test case 3: From import with attribute
    import_obj = Import(line_number=3, indented=False, module="sys", attribute="path", file_path=None)
    assert str(import_obj) == ":3 from sys import path"

    # Test case 4: From import with attribute and alias
    import_obj = Import(line_number=4, indented=True, module="collections", attribute="defaultdict", alias="dd", file_path=None)
    assert str(import_obj) == ":4 indented from collections import defaultdict as dd"

    # Test case 5: Cimport
    import_obj = Import(line_number=5, indented=False, module="cython", cimport=True, file_path=None)
    assert str(import_obj) == ":5 cimport cython"

    # Test case 6: With file path
    import_obj = Import(line_number=6, indented=False, module="json", file_path=Path("/path/to/file.py"))
    assert str(import_obj) == "/path/to/file.py:6 import json"


# LLM-generated content at query #5
#--------------------------

```python
def test_Import_statement():
    # Test basic import statement
    import_obj = Import(line_number=1, indented=False, module="os")
    assert import_obj.statement() == "import os"

    # Test import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np")
    assert import_obj.statement() == "import numpy as np"

    # Test cimport statement
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True)
    assert import_obj.statement() == "cimport cython"

    # Test from import statement
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path")
    assert import_obj.statement() == "from os import path"

    # Test from cimport statement
    import_obj = Import(line_number=1, indented=False, module="libc", attribute="stdio", cimport=True)
    assert import_obj.statement() == "from libc cimport stdio"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="collections", attribute="defaultdict", alias="dd")
    assert import_obj.statement() == "from collections import defaultdict as dd"


# LLM-generated content at query #6
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os\n", "import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test import with alias
    input_stream = ["import numpy as np\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"

    # Test from import
    input_stream = ["from collections import defaultdict\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "defaultdict"

    # Test from import with alias
    input_stream = ["from pathlib import Path as P\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias == "P"

    # Test cimport
    input_stream = ["cimport numpy\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True

    # Test multiline import
    input_stream = ["from collections import (\n", "    defaultdict,\n", "    OrderedDict\n", ")\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "collections"
    assert result[0].attribute == "defaultdict"
    assert result[1].module == "collections"
    assert result[1].attribute == "OrderedDict"

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy\n"]
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias is None

    # Test indented import
    input_stream = ["    import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    assert result[0].module == "os"

    # Test with file path
    file_path = Path("/test/path.py")
    input_stream = ["import os\n"]
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    # Test top_only parameter
    input_stream = ["import os\n", "def foo():\n", "    import sys\n"]
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test skipped lines
    input_stream = ["# This is a comment\n", "import os\n", "\"\"\"Docstring\"\"\"\n", "import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test semicolon separated imports
    input_stream = ["import os; import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test escaped newline
    input_stream = ["import os \\\n", "    , sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test yield and raise handling
    input_stream = ["yield\n", "import os\n", "raise Exception\n", "import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"


# LLM-generated content at query #7
#--------------------------

```python
def test_imports():
    # Test simple import
    input_stream = ["import os\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].alias is None
    assert imports_list[0].attribute is None
    assert not imports_list[0].cimport

    # Test import with alias
    input_stream = ["import numpy as np\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias == "np"
    assert imports_list[0].attribute is None
    assert not imports_list[0].cimport

    # Test from import
    input_stream = ["from collections import defaultdict\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "collections"
    assert imports_list[0].attribute == "defaultdict"
    assert imports_list[0].alias is None
    assert not imports_list[0].cimport

    # Test from import with alias
    input_stream = ["from pathlib import Path as P\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "pathlib"
    assert imports_list[0].attribute == "Path"
    assert imports_list[0].alias == "P"
    assert not imports_list[0].cimport

    # Test cimport
    input_stream = ["cimport numpy\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].cimport

    # Test multiple imports
    input_stream = ["import os, sys\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[1].module == "sys"

    # Test multiline import
    input_stream = ["from collections import (\n    defaultdict,\n    OrderedDict\n)\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "collections"
    assert imports_list[0].attribute == "defaultdict"
    assert imports_list[1].module == "collections"
    assert imports_list[1].attribute == "OrderedDict"

    # Test indented import
    input_stream = ["    import os\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].indented

    # Test with comments
    input_stream = ["import os  # some comment\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"

    # Test with semicolon
    input_stream = ["import os; import sys\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[1].module == "sys"

    # Test top_only with statement
    input_stream = ["import os\n", "def foo():\n", "    pass\n"]
    imports_list = list(imports(input_stream, top_only=True))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"

    # Test with file_path
    file_path = Path("/tmp/test.py")
    input_stream = ["import os\n"]
    imports_list = list(imports(input_stream, file_path=file_path))
    assert len(imports_list) == 1
    assert imports_list[0].file_path == file_path

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy\n"]
    imports_list = list(imports(input_stream, config=config))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias is None

    # Test with parentheses in from import
    input_stream = ["from typing import (List, Dict)\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "typing"
    assert imports_list[0].attribute == "List"
    assert imports_list[1].module == "typing"
    assert imports_list[1].attribute == "Dict"

    # Test with backslash continuation
    input_stream = ["from typing import \\\n    List, Dict\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "typing"
    assert imports_list[0].attribute == "List"
    assert imports_list[1].module == "typing"
    assert imports_list[1].attribute == "Dict"


# LLM-generated content at query #8
#--------------------------

```python
def test_imports():
    # Test case 1: Simple import
    input_stream = ["import os\n", "import sys\n"]
    expected = [
        Import(line_number=1, indented=False, module="os"),
        Import(line_number=2, indented=False, module="sys"),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 2: Import with alias
    input_stream = ["import numpy as np\n", "import pandas as pd\n"]
    expected = [
        Import(line_number=1, indented=False, module="numpy", alias="np"),
        Import(line_number=2, indented=False, module="pandas", alias="pd"),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 3: From import
    input_stream = ["from collections import defaultdict\n", "from typing import List\n"]
    expected = [
        Import(line_number=1, indented=False, module="collections", attribute="defaultdict"),
        Import(line_number=2, indented=False, module="typing", attribute="List"),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 4: From import with alias
    input_stream = ["from numpy import array as arr\n", "from pandas import DataFrame as DF\n"]
    expected = [
        Import(line_number=1, indented=False, module="numpy", attribute="array", alias="arr"),
        Import(line_number=2, indented=False, module="pandas", attribute="DataFrame", alias="DF"),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 5: Cimport
    input_stream = ["cimport numpy\n", "from numpy cimport ndarray\n"]
    expected = [
        Import(line_number=1, indented=False, module="numpy", cimport=True),
        Import(line_number=2, indented=False, module="numpy", attribute="ndarray", cimport=True),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 6: Indented import
    input_stream = ["    import os\n", "    import sys\n"]
    expected = [
        Import(line_number=1, indented=True, module="os"),
        Import(line_number=2, indented=True, module="sys"),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 7: Multiline import
    input_stream = ["from collections import (\n", "    defaultdict,\n", "    OrderedDict\n", ")\n"]
    expected = [
        Import(line_number=1, indented=False, module="collections", attribute="defaultdict"),
        Import(line_number=1, indented=False, module="collections", attribute="OrderedDict"),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 8: Import with comment
    input_stream = ["import os  # Operating system\n", "import sys  # System\n"]
    expected = [
        Import(line_number=1, indented=False, module="os"),
        Import(line_number=2, indented=False, module="sys"),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 9: Import with redundant alias
    input_stream = ["import os as os\n", "import sys as sys\n"]
    expected = [
        Import(line_number=1, indented=False, module="os"),
        Import(line_number=2, indented=False, module="sys"),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 10: Import with file path
    input_stream = ["import os\n", "import sys\n"]
    file_path = Path("/path/to/file.py")
    expected = [
        Import(line_number=1, indented=False, module="os", file_path=file_path),
        Import(line_number=2, indented=False, module="sys", file_path=file_path),
    ]
    assert list(imports(input_stream, file_path=file_path)) == expected


# LLM-generated content at query #9
#--------------------------

```python
def test_imports():
    # Test case 1: Simple import
    input_stream = ["import os\n", "import sys\n"]
    expected = [
        Import(1, False, "os", None, None, False, None),
        Import(2, False, "sys", None, None, False, None),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 2: Import with alias
    input_stream = ["import numpy as np\n"]
    expected = [Import(1, False, "numpy", None, "np", False, None)]
    assert list(imports(input_stream)) == expected

    # Test case 3: From import
    input_stream = ["from collections import defaultdict\n"]
    expected = [Import(1, False, "collections", "defaultdict", None, False, None)]
    assert list(imports(input_stream)) == expected

    # Test case 4: From import with alias
    input_stream = ["from pathlib import Path as P\n"]
    expected = [Import(1, False, "pathlib", "Path", "P", False, None)]
    assert list(imports(input_stream)) == expected

    # Test case 5: Cimport
    input_stream = ["cimport numpy as np\n"]
    expected = [Import(1, False, "numpy", None, "np", True, None)]
    assert list(imports(input_stream)) == expected

    # Test case 6: Multiple imports on one line
    input_stream = ["import os, sys\n"]
    expected = [
        Import(1, False, "os", None, None, False, None),
        Import(1, False, "sys", None, None, False, None),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 7: Indented import
    input_stream = ["    import os\n"]
    expected = [Import(1, True, "os", None, None, False, None)]
    assert list(imports(input_stream)) == expected

    # Test case 8: Import with parentheses
    input_stream = ["from typing import (\n    List,\n    Dict,\n)\n"]
    expected = [
        Import(1, False, "typing", "List", None, False, None),
        Import(3, False, "typing", "Dict", None, False, None),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 9: Skip non-import lines
    input_stream = ["x = 1\n", "import os\n", "y = 2\n"]
    expected = [Import(2, False, "os", None, None, False, None)]
    assert list(imports(input_stream)) == expected

    # Test case 10: Skip comments
    input_stream = ["# This is a comment\n", "import os\n"]
    expected = [Import(2, False, "os", None, None, False, None)]
    assert list(imports(input_stream)) == expected

    # Test case 11: Skip multiline strings
    input_stream = ['"""\nThis is a multiline string\n"""\n', "import os\n"]
    expected = [Import(4, False, "os", None, None, False, None)]
    assert list(imports(input_stream)) == expected

    # Test case 12: Skip yield statements
    input_stream = ["yield\n", "import os\n"]
    expected = [Import(2, False, "os", None, None, False, None)]
    assert list(imports(input_stream)) == expected

    # Test case 13: Skip raise statements
    input_stream = ["raise ValueError\n", "import os\n"]
    expected = [Import(2, False, "os", None, None, False, None)]
    assert list(imports(input_stream)) == expected

    # Test case 14: Skip line continuations
    input_stream = ["import os \\\n", "    import sys\n"]
    expected = [
        Import(1, False, "os", None, None, False, None),
        Import(2, False, "sys", None, None, False, None),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 15: Skip section comments
    input_stream = ["# isort: off\n", "import os\n", "import sys\n", "# isort: on\n", "import json\n"]
    expected = [
        Import(2, False, "os", None, None, False, None),
        Import(3, False, "sys", None, None, False, None),
        Import(5, False, "json", None, None, False, None),
    ]
    assert list(imports(input_stream)) == expected

    # Test case 16: Skip redundant aliases
    input_stream = ["import os as os\n"]
    expected = []
    assert list(imports(input_stream)) == expected

    # Test case 17: Skip redundant aliases in from imports
    input_stream = ["from os import path as path\n"]
    expected = []
    assert list(imports(input_stream)) == expected

    # Test case 18: Skip redundant aliases with config
    config = Config(remove_redundant_aliases=False)
    input_stream = ["import os as os\n"]
    expected = [Import(1, False, "os", None, "os", False, None)]
    assert list(imports(input_stream, config=config)) == expected

    # Test case 19: Skip redundant aliases in from imports with config
    config = Config(remove_redundant_aliases=False)
    input_stream = ["from os import path as path\n"]
    expected = [Import(1, False, "os", "path", "path", False, None)]
    assert list(imports(input_stream, config=config)) == expected

    # Test case 20: Skip top level imports only
    input_stream = ["import os\n", "def foo():\n", "    import sys\n"]
    expected = [Import(1, False, "os", None, None, False, None)]
    assert list(imports(input_stream, top_only=True)) == expected


# LLM-generated content at query #10
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os\n", "import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test import with alias
    input_stream = ["import numpy as np\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"

    # Test from import
    input_stream = ["from collections import defaultdict\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "defaultdict"

    # Test from import with alias
    input_stream = ["from pathlib import Path as P\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias == "P"

    # Test multiple imports
    input_stream = ["import os, sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test cimport
    input_stream = ["cimport numpy\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True

    # Test indented import
    input_stream = ["    import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    assert result[0].module == "os"

    # Test multiline import
    input_stream = ["from collections import (\n", "    defaultdict,\n", "    OrderedDict\n", ")\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "collections"
    assert result[0].attribute == "defaultdict"
    assert result[1].module == "collections"
    assert result[1].attribute == "OrderedDict"

    # Test import with comment
    input_stream = ["import os  # Operating system interfaces\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test skip non-import lines
    input_stream = ["x = 1\n", "import os\n", "y = 2\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test top_only parameter
    input_stream = ["import os\n", "def foo():\n", "    import sys\n"]
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test file_path parameter
    input_stream = ["import os\n"]
    result = list(imports(input_stream, file_path=Path("/test.py")))
    assert len(result) == 1
    assert result[0].file_path == Path("/test.py")

    # Test config parameter
    input_stream = ["import os as os\n"]
    result = list(imports(input_stream, config=Config(remove_redundant_aliases=True)))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias is None

    # Test empty input
    input_stream = []
    result = list(imports(input_stream))
    assert len(result) == 0

    # Test skip_line with quotes
    input_stream = ['x = "import os"\n', "import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "sys"


# LLM-generated content at query #11
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert not result[0].indented

    # Test import with alias
    input_stream = ["import numpy as np\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[0].line_number == 1

    # Test from import
    input_stream = ["from sys import path\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "sys"
    assert result[0].attribute == "path"
    assert result[0].line_number == 1

    # Test from import with alias
    input_stream = ["from pandas import DataFrame as df\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pandas"
    assert result[0].attribute == "DataFrame"
    assert result[0].alias == "df"
    assert result[0].line_number == 1

    # Test multiple imports
    input_stream = ["import os, sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test cimport
    input_stream = ["cimport numpy\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport

    # Test indented import
    input_stream = ["    import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented

    # Test multiline import
    input_stream = ["from collections import (\n", "    OrderedDict,\n", "    defaultdict\n", ")\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[1].module == "collections"
    assert result[1].attribute == "defaultdict"

    # Test comment handling
    input_stream = ["# This is a comment\n", "import os  # inline comment\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test skip_line functionality
    input_stream = ['"""docstring"""', "import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test top_only parameter
    input_stream = ["import os\n", "def foo():\n", "    import sys\n"]
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test file_path parameter
    file_path = Path("/test/file.py")
    input_stream = ["import os\n"]
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy\n"]
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias is None

    # Test statement method
    import_obj = Import(1, False, "os", None, None, False, None)
    assert import_obj.statement() == "import os"

    import_obj = Import(1, False, "os", "path", None, False, None)
    assert import_obj.statement() == "from os import path"

    import_obj = Import(1, False, "numpy", None, "np", False, None)
    assert import_obj.statement() == "import numpy as np"

    import_obj = Import(1, False, "os", "path", "p", False, None)
    assert import_obj.statement() == "from os import path as p"

    import_obj = Import(1, False, "numpy", None, None, True, None)
    assert import_obj.statement() == "cimport numpy"

    # Test __str__ method
    import_obj = Import(1, False, "os", None, None, False, Path("/test/file.py"))
    assert str(import_obj) == "/test/file.py:1 import os"

    import_obj = Import(1, True, "os", None, None, False, None)
    assert str(import_obj) == ":1 indented import os"


# LLM-generated content at query #12
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = iter(["import os\n"])
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias is None
    assert result[0].attribute is None
    assert not result[0].cimport

    # Test import with alias
    input_stream = iter(["import numpy as np\n"])
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[0].attribute is None
    assert not result[0].cimport

    # Test from import
    input_stream = iter(["from sys import path\n"])
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "sys"
    assert result[0].attribute == "path"
    assert result[0].alias is None
    assert not result[0].cimport

    # Test from import with alias
    input_stream = iter(["from pandas import DataFrame as df\n"])
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pandas"
    assert result[0].attribute == "DataFrame"
    assert result[0].alias == "df"
    assert not result[0].cimport

    # Test cimport
    input_stream = iter(["cimport numpy\n"])
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport

    # Test multiple imports
    input_stream = iter(["import os, sys\n"])
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test multiline import
    input_stream = iter([
        "from collections import (\n",
        "    OrderedDict,\n",
        "    defaultdict,\n",
        ")\n"
    ])
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[1].module == "collections"
    assert result[1].attribute == "defaultdict"

    # Test indented import
    input_stream = iter(["    import os\n"])
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented

    # Test with comments
    input_stream = iter(["import os  # some comment\n"])
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test with semicolon
    input_stream = iter(["import os; import sys\n"])
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test with file path
    file_path = Path("/some/path")
    input_stream = iter(["import os\n"])
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    # Test top_only
    input_stream = iter([
        "import os\n",
        "def foo():\n",
        "    import sys\n"
    ])
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test with redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = iter(["import numpy as numpy\n"])
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias is None

    # Test empty input
    input_stream = iter([])
    result = list(imports(input_stream))
    assert len(result) == 0

    # Test skipped lines
    input_stream = iter([
        "# comment\n",
        "import os\n",
        "\"\"\"docstring\"\"\"\n",
        "import sys\n"
    ])
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"


# LLM-generated content at query #13
#--------------------------

```python
def test_imports():
    # Test simple import
    input_stream = ["import os"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].cimport is False

    # Test import with alias
    input_stream = ["import numpy as np"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"

    # Test from import
    input_stream = ["from sys import path"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "sys"
    assert result[0].attribute == "path"

    # Test multiple imports
    input_stream = ["import os", "import sys"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test cimport
    input_stream = ["cimport numpy"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True

    # Test indented import
    input_stream = ["    import os"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True

    # Test import with comment
    input_stream = ["import os # comment"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test multiline import
    input_stream = ["from sys import (", "    path,", "    argv", ")"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[0].attribute == "path"
    assert result[1].module == "sys"
    assert result[1].attribute == "argv"

    # Test import with backslash
    input_stream = ["from sys import path, \\", "    argv"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[0].attribute == "path"
    assert result[1].module == "sys"
    assert result[1].attribute == "argv"

    # Test empty input
    input_stream = []
    result = list(imports(input_stream))
    assert len(result) == 0

    # Test non-import statements
    input_stream = ["x = 1", "print('hello')"]
    result = list(imports(input_stream))
    assert len(result) == 0

    # Test statement declarations with top_only
    input_stream = ["import os", "def foo():", "    import sys"]
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy"]
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias is None


# LLM-generated content at query #14
#--------------------------

```python
def test_imports():
    # Test basic imports
    input_stream = ["import os", "import sys"]
    expected = [
        Import(1, False, "os"),
        Import(2, False, "sys"),
    ]
    assert list(imports(input_stream)) == expected

    # Test from imports
    input_stream = ["from os import path", "from sys import argv"]
    expected = [
        Import(1, False, "os", "path"),
        Import(2, False, "sys", "argv"),
    ]
    assert list(imports(input_stream)) == expected

    # Test aliased imports
    input_stream = ["import numpy as np", "from pandas import DataFrame as df"]
    expected = [
        Import(1, False, "numpy", alias="np"),
        Import(2, False, "pandas", "DataFrame", alias="df"),
    ]
    assert list(imports(input_stream)) == expected

    # Test cimports
    input_stream = ["cimport numpy", "from cython cimport int"]
    expected = [
        Import(1, False, "numpy", cimport=True),
        Import(2, False, "cython", "int", cimport=True),
    ]
    assert list(imports(input_stream)) == expected

    # Test multiline imports
    input_stream = ["from os import (\n    path,\n    walk\n)"]
    expected = [
        Import(1, False, "os", "path"),
        Import(1, False, "os", "walk"),
    ]
    assert list(imports(input_stream)) == expected

    # Test indented imports
    input_stream = ["    import os", "        import sys"]
    expected = [
        Import(1, True, "os"),
        Import(2, True, "sys"),
    ]
    assert list(imports(input_stream)) == expected

    # Test redundant aliases removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy", "from os import path as path"]
    expected = [
        Import(1, False, "numpy"),
        Import(2, False, "os", "path"),
    ]
    assert list(imports(input_stream, config=config)) == expected

    # Test comments and skipped lines
    input_stream = ["# This is a comment", "import os", "'''Docstring'''", "import sys"]
    expected = [
        Import(2, False, "os"),
        Import(4, False, "sys"),
    ]
    assert list(imports(input_stream)) == expected

    # Test statement termination
    input_stream = ["import os; import sys"]
    expected = [
        Import(1, False, "os"),
        Import(1, False, "sys"),
    ]
    assert list(imports(input_stream)) == expected

    # Test file path
    file_path = Path("/path/to/file.py")
    input_stream = ["import os"]
    expected = [Import(1, False, "os", file_path=file_path)]
    assert list(imports(input_stream, file_path=file_path)) == expected


# LLM-generated content at query #15
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias is None
    assert result[0].attribute is None
    assert not result[0].cimport
    assert not result[0].indented

    # Test import with alias
    input_stream = ["import numpy as np\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[0].attribute is None
    assert not result[0].cimport
    assert not result[0].indented

    # Test from import
    input_stream = ["from collections import defaultdict\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "defaultdict"
    assert result[0].alias is None
    assert not result[0].cimport
    assert not result[0].indented

    # Test from import with alias
    input_stream = ["from pathlib import Path as P\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias == "P"
    assert not result[0].cimport
    assert not result[0].indented

    # Test cimport
    input_stream = ["cimport numpy\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias is None
    assert result[0].attribute is None
    assert result[0].cimport
    assert not result[0].indented

    # Test multiple imports
    input_stream = ["import os, sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test indented import
    input_stream = ["    import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented

    # Test multiline import
    input_stream = ["from collections import (\n", "    defaultdict,\n", "    OrderedDict\n", ")\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "collections"
    assert result[0].attribute == "defaultdict"
    assert result[1].module == "collections"
    assert result[1].attribute == "OrderedDict"

    # Test with comments
    input_stream = ["import os  # some comment\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test with semicolon
    input_stream = ["import os; import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test with redundant alias (should be removed if config says so)
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy\n"]
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias is None

    # Test with file path
    file_path = Path("/some/path")
    input_stream = ["import os\n"]
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    # Test top_only
    input_stream = ["import os\n", "def foo():\n", "    pass\n"]
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"


# LLM-generated content at query #16
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert not result[0].cimport

    # Test import with alias
    input_stream = ["import numpy as np"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"

    # Test from import
    input_stream = ["from sys import argv"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "sys"
    assert result[0].attribute == "argv"

    # Test multiple imports
    input_stream = ["import os, sys"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test cimport
    input_stream = ["cimport numpy"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport

    # Test multiline import
    input_stream = ["from os import (\n    path,\n    sys\n)"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "sys"

    # Test import with redundant alias
    input_stream = ["import os as os"]
    config = Config(remove_redundant_aliases=True)
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias is None

    # Test indented import
    input_stream = ["    import os"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented

    # Test import with comment
    input_stream = ["import os  # This is a comment"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test empty input
    input_stream = []
    result = list(imports(input_stream))
    assert len(result) == 0

    # Test non-import line
    input_stream = ["x = 5"]
    result = list(imports(input_stream))
    assert len(result) == 0

    # Test import with backslash continuation
    input_stream = ["from os import \\\n    path"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Import___str__():
    # Test with cimport and alias
    import_obj = Import(
        line_number=10,
        indented=True,
        module="numpy",
        attribute="array",
        alias="np_array",
        cimport=True,
        file_path=Path("test.py")
    )
    assert str(import_obj) == "test.py:10 indented from numpy cimport array as np_array"

    # Test with regular import and no alias
    import_obj = Import(
        line_number=5,
        indented=False,
        module="sys",
        cimport=False,
        file_path=None
    )
    assert str(import_obj) == ":5 import sys"

    # Test with from import and attribute
    import_obj = Import(
        line_number=15,
        indented=False,
        module="os.path",
        attribute="join",
        cimport=False,
        file_path=Path("example.py")
    )
    assert str(import_obj) == "example.py:15 from os.path import join"

    # Test with indented import
    import_obj = Import(
        line_number=20,
        indented=True,
        module="collections",
        attribute="defaultdict",
        cimport=False,
        file_path=Path("script.py")
    )
    assert str(import_obj) == "script.py:20 indented from collections import defaultdict"


# LLM-generated content at query #2
#--------------------------

```python
def test_Import___str__():
    # Test with all fields
    import_obj = Import(
        line_number=10,
        indented=True,
        module="test_module",
        attribute="test_attribute",
        alias="test_alias",
        cimport=True,
        file_path=Path("/test/path")
    )
    assert str(import_obj) == "/test/path:10 indented cimport test_attribute from test_module as test_alias"

    # Test without alias
    import_obj = Import(
        line_number=5,
        indented=False,
        module="test_module",
        attribute="test_attribute",
        cimport=False,
        file_path=None
    )
    assert str(import_obj) == ":5 import test_attribute from test_module"

    # Test without attribute
    import_obj = Import(
        line_number=3,
        indented=True,
        module="test_module",
        alias="test_alias",
        cimport=True,
        file_path=Path("/another/path")
    )
    assert str(import_obj) == "/another/path:3 indented cimport test_module as test_alias"

    # Test with no optional fields
    import_obj = Import(
        line_number=1,
        indented=False,
        module="test_module"
    )
    assert str(import_obj) == ":1 import test_module"


# LLM-generated content at query #3
#--------------------------

```python
def test_imports():
    # Test case 1: Simple import
    input_stream = ["import os\n", "import sys\n"]
    expected = [
        Import(1, False, "os"),
        Import(2, False, "sys"),
    ]
    result = list(imports(input_stream))
    assert result == expected

    # Test case 2: Import with alias
    input_stream = ["import numpy as np\n", "import pandas as pd\n"]
    expected = [
        Import(1, False, "numpy", alias="np"),
        Import(2, False, "pandas", alias="pd"),
    ]
    result = list(imports(input_stream))
    assert result == expected

    # Test case 3: From import
    input_stream = ["from collections import defaultdict\n", "from typing import List\n"]
    expected = [
        Import(1, False, "collections", attribute="defaultdict"),
        Import(2, False, "typing", attribute="List"),
    ]
    result = list(imports(input_stream))
    assert result == expected

    # Test case 4: From import with alias
    input_stream = ["from numpy import array as arr\n", "from pandas import DataFrame as df\n"]
    expected = [
        Import(1, False, "numpy", attribute="array", alias="arr"),
        Import(2, False, "pandas", attribute="DataFrame", alias="df"),
    ]
    result = list(imports(input_stream))
    assert result == expected

    # Test case 5: Cimport
    input_stream = ["cimport numpy as np\n", "from numpy cimport ndarray\n"]
    expected = [
        Import(1, False, "numpy", alias="np", cimport=True),
        Import(2, False, "numpy", attribute="ndarray", cimport=True),
    ]
    result = list(imports(input_stream))
    assert result == expected

    # Test case 6: Indented import
    input_stream = ["def foo():\n", "    import os\n", "    import sys\n"]
    expected = [
        Import(2, True, "os"),
        Import(3, True, "sys"),
    ]
    result = list(imports(input_stream))
    assert result == expected

    # Test case 7: Multi-line import
    input_stream = ["from collections import (\n", "    defaultdict,\n", "    OrderedDict\n", ")\n"]
    expected = [
        Import(1, False, "collections", attribute="defaultdict"),
        Import(1, False, "collections", attribute="OrderedDict"),
    ]
    result = list(imports(input_stream))
    assert result == expected

    # Test case 8: Import with comment
    input_stream = ["import os  # Operating system\n", "import sys  # System\n"]
    expected = [
        Import(1, False, "os"),
        Import(2, False, "sys"),
    ]
    result = list(imports(input_stream))
    assert result == expected

    # Test case 9: Skip non-import lines
    input_stream = ["x = 1\n", "import os\n", "y = 2\n", "import sys\n"]
    expected = [
        Import(2, False, "os"),
        Import(4, False, "sys"),
    ]
    result = list(imports(input_stream))
    assert result == expected

    # Test case 10: Top only imports
    input_stream = ["import os\n", "def foo():\n", "    import sys\n"]
    expected = [
        Import(1, False, "os"),
    ]
    result = list(imports(input_stream, top_only=True))
    assert result == expected


# LLM-generated content at query #4
#--------------------------

```python
def test_Import___str__():
    # Test case 1: Basic import without alias
    import1 = Import(line_number=1, indented=False, module="os", file_path=None)
    assert str(import1) == ":1 import os"

    # Test case 2: Import with alias
    import2 = Import(line_number=2, indented=True, module="numpy", alias="np", file_path=Path("test.py"))
    assert str(import2) == "test.py:2 indented import numpy as np"

    # Test case 3: From import with attribute
    import3 = Import(line_number=3, indented=False, module="sys", attribute="path", file_path=None)
    assert str(import3) == ":3 from sys import path"

    # Test case 4: Cimport with alias
    import4 = Import(line_number=4, indented=False, module="cython", alias="cy", cimport=True, file_path=Path("module.pyx"))
    assert str(import4) == "module.pyx:4 cimport cython as cy"

    # Test case 5: From cimport with attribute and alias
    import5 = Import(line_number=5, indented=True, module="libc", attribute="stdio", alias="cstdio", cimport=True, file_path=Path("example.pyx"))
    assert str(import5) == "example.pyx:5 indented from libc cimport stdio as cstdio"


# LLM-generated content at query #5
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="os")
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np")
    assert import_obj.statement() == "import numpy as np"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", alias="cy", cimport=True)
    assert import_obj.statement() == "cimport cython as cy"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path")
    assert import_obj.statement() == "from os import path"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", alias="p")
    assert import_obj.statement() == "from os import path as p"

    # Test from cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", cimport=True)
    assert import_obj.statement() == "from cython cimport view"

    # Test from cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", alias="v", cimport=True)
    assert import_obj.statement() == "from cython cimport view as v"


# LLM-generated content at query #6
#--------------------------

```python
def test_Import___str__():
    import_obj = Import(
        line_number=1,
        indented=False,
        module="os",
        attribute="path",
        alias="osp",
        cimport=False,
        file_path=Path("/test.py")
    )
    assert str(import_obj) == "/test.py:1 import os.path as osp"

    import_obj = Import(
        line_number=2,
        indented=True,
        module="sys",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=None
    )
    assert str(import_obj) == ":2 indented cimport sys"


# LLM-generated content at query #7
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os\n", "import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test import with alias
    input_stream = ["import numpy as np\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"

    # Test from import
    input_stream = ["from collections import defaultdict\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "defaultdict"

    # Test from import with alias
    input_stream = ["from pathlib import Path as P\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias == "P"

    # Test cimport
    input_stream = ["cimport numpy\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True

    # Test multiline import
    input_stream = ["from collections import (\n", "    defaultdict,\n", "    OrderedDict,\n", ")\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "collections"
    assert result[0].attribute == "defaultdict"
    assert result[1].module == "collections"
    assert result[1].attribute == "OrderedDict"

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy\n"]
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias is None

    # Test indented import
    input_stream = ["    import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    assert result[0].module == "os"

    # Test with file path
    file_path = Path("/path/to/file.py")
    input_stream = ["import sys\n"]
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path
    assert result[0].module == "sys"

    # Test top_only parameter
    input_stream = ["import os\n", "def foo():\n", "    import sys\n"]
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test with comments
    input_stream = ["import os  # some comment\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test with semicolon
    input_stream = ["import os; import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test with backslash continuation
    input_stream = ["from collections import \\\n", "    defaultdict\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "defaultdict"


# LLM-generated content at query #8
#--------------------------

```python
def test_Import___str__():
    # Test basic import without alias
    import1 = Import(line_number=1, indented=False, module="os")
    assert str(import1) == ":1 import os"

    # Test basic import with alias
    import2 = Import(line_number=2, indented=False, module="numpy", alias="np")
    assert str(import2) == ":2 import numpy as np"

    # Test cimport
    import3 = Import(line_number=3, indented=False, module="libc", cimport=True)
    assert str(import3) == ":3 cimport libc"

    # Test from import
    import4 = Import(line_number=4, indented=False, module="os", attribute="path")
    assert str(import4) == ":4 from os import path"

    # Test from import with alias
    import5 = Import(line_number=5, indented=False, module="os", attribute="path", alias="osp")
    assert str(import5) == ":5 from os import path as osp"

    # Test indented import
    import6 = Import(line_number=6, indented=True, module="sys")
    assert str(import6) == ":6 indented import sys"

    # Test with file path
    import7 = Import(line_number=7, indented=False, module="sys", file_path=Path("/path/to/file.py"))
    assert str(import7) == "/path/to/file.py:7 import sys"

    # Test cimport from import with alias
    import8 = Import(line_number=8, indented=False, module="libc", attribute="stdio", alias="cstdio", cimport=True)
    assert str(import8) == ":8 from libc cimport stdio as cstdio"


# LLM-generated content at query #9
#--------------------------

```python
def test_Import___str__():
    # Test with minimal parameters
    import_obj = Import(line_number=1, indented=False, module="os")
    assert str(import_obj) == ":1 import os"

    # Test with indented import
    import_obj = Import(line_number=2, indented=True, module="sys")
    assert str(import_obj) == ":2 indented import sys"

    # Test with attribute
    import_obj = Import(line_number=3, indented=False, module="collections", attribute="defaultdict")
    assert str(import_obj) == ":3 from collections import defaultdict"

    # Test with alias
    import_obj = Import(line_number=4, indented=False, module="numpy", alias="np")
    assert str(import_obj) == ":4 import numpy as np"

    # Test with both attribute and alias
    import_obj = Import(line_number=5, indented=True, module="pandas", attribute="DataFrame", alias="pd")
    assert str(import_obj) == ":5 indented from pandas import DataFrame as pd"

    # Test with cimport
    import_obj = Import(line_number=6, indented=False, module="cython", cimport=True)
    assert str(import_obj) == ":6 cimport cython"

    # Test with file_path
    import_obj = Import(line_number=7, indented=False, module="pathlib", file_path=Path("/tmp/test.py"))
    assert str(import_obj) == "/tmp/test.py:7 import pathlib"

    # Test with all parameters
    import_obj = Import(
        line_number=8,
        indented=True,
        module="django",
        attribute="models",
        alias="dm",
        cimport=True,
        file_path=Path("/home/user/project/main.py")
    )
    assert str(import_obj) == "/home/user/project/main.py:8 indented from django cimport models as dm"


# LLM-generated content at query #10
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="sys")
    assert import_obj.statement() == "import sys"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np")
    assert import_obj.statement() == "import numpy as np"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path")
    assert import_obj.statement() == "from os import path"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="collections", attribute="defaultdict", alias="dd")
    assert import_obj.statement() == "from collections import defaultdict as dd"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", alias="cy", cimport=True)
    assert import_obj.statement() == "cimport cython as cy"

    # Test cimport from import without alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", cimport=True)
    assert import_obj.statement() == "from cython cimport view"

    # Test cimport from import with alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", alias="v", cimport=True)
    assert import_obj.statement() == "from cython cimport view as v"


# LLM-generated content at query #11
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="sys")
    assert import_obj.statement() == "import sys"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np")
    assert import_obj.statement() == "import numpy as np"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path")
    assert import_obj.statement() == "from os import path"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="collections", attribute="defaultdict", alias="dd")
    assert import_obj.statement() == "from collections import defaultdict as dd"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", alias="cy", cimport=True)
    assert import_obj.statement() == "cimport cython as cy"

    # Test from cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", cimport=True)
    assert import_obj.statement() == "from cython cimport view"

    # Test from cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", alias="cv", cimport=True)
    assert import_obj.statement() == "from cython cimport view as cv"


# LLM-generated content at query #12
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="os", file_path=None)
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np", file_path=None)
    assert import_obj.statement() == "import numpy as np"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True, file_path=None)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", alias="cy", cimport=True, file_path=None)
    assert import_obj.statement() == "cimport cython as cy"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", file_path=None)
    assert import_obj.statement() == "from os import path"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", alias="p", file_path=None)
    assert import_obj.statement() == "from os import path as p"

    # Test from cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="cython", cimport=True, file_path=None)
    assert import_obj.statement() == "from cython cimport cython"

    # Test from cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="cython", alias="cy", cimport=True, file_path=None)
    assert import_obj.statement() == "from cython cimport cython as cy"


# LLM-generated content at query #13
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias is None
    assert result[0].attribute is None

    # Test import with alias
    input_stream = ["import numpy as np\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"

    # Test from import
    input_stream = ["from sys import path\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "sys"
    assert result[0].attribute == "path"

    # Test from import with alias
    input_stream = ["from collections import OrderedDict as OD\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"

    # Test multiple imports
    input_stream = ["import os, sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test cimport
    input_stream = ["cimport numpy\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True

    # Test multiline import
    input_stream = ["from typing import (\n    List,\n    Dict,\n)\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "typing"
    assert result[0].attribute == "List"
    assert result[1].module == "typing"
    assert result[1].attribute == "Dict"

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy\n"]
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None

    # Test indented import
    input_stream = ["    import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True

    # Test with file path
    file_path = Path("/test/file.py")
    input_stream = ["import os\n"]
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    # Test top_only with statement
    input_stream = ["import os\n", "def func():\n", "    pass\n"]
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test skipping lines
    input_stream = ["# This is a comment\n", "import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test with semicolon
    input_stream = ["import os; import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test with backslash continuation
    input_stream = ["from typing import \\\n    List\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "typing"
    assert result[0].attribute == "List"


# LLM-generated content at query #14
#--------------------------

```python
def test_imports():
    # Test basic straight import
    input_stream = ["import os\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute is None
    assert imports_list[0].alias is None
    assert not imports_list[0].cimport

    # Test straight import with alias
    input_stream = ["import numpy as np\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias == "np"

    # Test from import
    input_stream = ["from collections import defaultdict\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "collections"
    assert imports_list[0].attribute == "defaultdict"

    # Test from import with alias
    input_stream = ["from pathlib import Path as P\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "pathlib"
    assert imports_list[0].attribute == "Path"
    assert imports_list[0].alias == "P"

    # Test cimport
    input_stream = ["cimport numpy\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].cimport

    # Test multiple imports on one line
    input_stream = ["import sys, os\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "sys"
    assert imports_list[1].module == "os"

    # Test parenthesized imports
    input_stream = ["from typing import (\n    List,\n    Dict,\n)\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "typing"
    assert imports_list[0].attribute == "List"
    assert imports_list[1].module == "typing"
    assert imports_list[1].attribute == "Dict"

    # Test indented imports
    input_stream = ["    import pandas\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].indented

    # Test with comments
    input_stream = ["import math  # math module\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "math"

    # Test with semicolons
    input_stream = ["import json; import ast\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "json"
    assert imports_list[1].module == "ast"

    # Test top_only parameter
    input_stream = ["import re\n", "def foo():\n", "    import bar\n"]
    imports_list = list(imports(input_stream, top_only=True))
    assert len(imports_list) == 1
    assert imports_list[0].module == "re"

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import pandas as pandas\n"]
    imports_list = list(imports(input_stream, config=config))
    assert len(imports_list) == 1
    assert imports_list[0].alias is None

    # Test file_path parameter
    file_path = Path("/test/file.py")
    input_stream = ["import sys\n"]
    imports_list = list(imports(input_stream, file_path=file_path))
    assert len(imports_list) == 1
    assert imports_list[0].file_path == file_path


# LLM-generated content at query #15
#--------------------------

```python
def test_Import___str__():
    # Test case 1: Basic import without alias
    import_obj = Import(line_number=1, indented=False, module="os", file_path=None)
    assert str(import_obj) == ":1 import os"

    # Test case 2: Import with alias
    import_obj = Import(line_number=2, indented=True, module="numpy", alias="np", file_path=None)
    assert str(import_obj) == ":2 indented import numpy as np"

    # Test case 3: From import with attribute
    import_obj = Import(line_number=3, indented=False, module="collections", attribute="defaultdict", file_path=None)
    assert str(import_obj) == ":3 from collections import defaultdict"

    # Test case 4: From import with attribute and alias
    import_obj = Import(line_number=4, indented=True, module="typing", attribute="List", alias="list", file_path=None)
    assert str(import_obj) == ":4 indented from typing import List as list"

    # Test case 5: Cimport
    import_obj = Import(line_number=5, indented=False, module="cython", cimport=True, file_path=None)
    assert str(import_obj) == ":5 cimport cython"

    # Test case 6: With file path
    import_obj = Import(line_number=6, indented=False, module="sys", file_path=Path("/path/to/file.py"))
    assert str(import_obj) == "/path/to/file.py:6 import sys"


# LLM-generated content at query #16
#--------------------------

```python
def test_imports():
    # Test basic straight import
    input_stream = ["import os\n", "import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0] == Import(1, False, "os")
    assert result[1] == Import(2, False, "sys")

    # Test import with alias
    input_stream = ["import numpy as np\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0] == Import(1, False, "numpy", alias="np")

    # Test from import
    input_stream = ["from collections import defaultdict\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0] == Import(1, False, "collections", attribute="defaultdict")

    # Test from import with alias
    input_stream = ["from pathlib import Path as P\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0] == Import(1, False, "pathlib", attribute="Path", alias="P")

    # Test cimport
    input_stream = ["cimport numpy\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0] == Import(1, False, "numpy", cimport=True)

    # Test multiple imports on one line
    input_stream = ["import os, sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0] == Import(1, False, "os")
    assert result[1] == Import(1, False, "sys")

    # Test indented import
    input_stream = ["    import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0] == Import(1, True, "os")

    # Test import with parentheses
    input_stream = ["from collections import (\n    defaultdict,\n    OrderedDict\n)\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0] == Import(1, False, "collections", attribute="defaultdict")
    assert result[1] == Import(1, False, "collections", attribute="OrderedDict")

    # Test import with backslash continuation
    input_stream = ["from collections import defaultdict, \\\n    OrderedDict\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0] == Import(1, False, "collections", attribute="defaultdict")
    assert result[1] == Import(1, False, "collections", attribute="OrderedDict")

    # Test with comments
    input_stream = ["# This is a comment\nimport os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0] == Import(2, False, "os")

    # Test with section comments
    config = Config(section_comments=["# isort: off", "# isort: on"])
    input_stream = ["# isort: off\nimport os\n# isort: on\nimport sys\n"]
    result = list(imports(input_stream, config=config))
    assert len(result) == 2
    assert result[0] == Import(2, False, "os")
    assert result[1] == Import(4, False, "sys")

    # Test top_only parameter
    input_stream = ["import os\ndef foo():\n    pass\n"]
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0] == Import(1, False, "os")

    # Test file_path parameter
    file_path = Path("/path/to/file.py")
    input_stream = ["import os\n"]
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0] == Import(1, False, "os", file_path=file_path)

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy\n"]
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0] == Import(1, False, "numpy")

    # Test empty input
    input_stream = []
    result = list(imports(input_stream))
    assert len(result) == 0


# LLM-generated content at query #17
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="os")
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np")
    assert import_obj.statement() == "import numpy as np"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", alias="cy", cimport=True)
    assert import_obj.statement() == "cimport cython as cy"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path")
    assert import_obj.statement() == "from os import path"

    # Test from cimport without alias
    import_obj = Import(line_number=1, indented=False, module="libc", attribute="stdio", cimport=True)
    assert import_obj.statement() == "from libc cimport stdio"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", alias="p")
    assert import_obj.statement() == "from os import path as p"

    # Test from cimport with alias
    import_obj = Import(line_number=1, indented=False, module="libc", attribute="stdio", alias="s", cimport=True)
    assert import_obj.statement() == "from libc cimport stdio as s"


# LLM-generated content at query #18
#--------------------------

```python
def test_Import___str__():
    # Test with all attributes
    import_obj = Import(
        line_number=10,
        indented=True,
        module="os",
        attribute="path",
        alias="ospath",
        cimport=False,
        file_path=Path("/test.py")
    )
    assert str(import_obj) == "/test.py:10 indented from os import path as ospath"

    # Test without attribute and alias
    import_obj = Import(
        line_number=5,
        indented=False,
        module="sys",
        cimport=True,
        file_path=None
    )
    assert str(import_obj) == ":5 cimport sys"

    # Test with attribute but no alias
    import_obj = Import(
        line_number=15,
        indented=True,
        module="collections",
        attribute="defaultdict",
        file_path=Path("/example.py")
    )
    assert str(import_obj) == "/example.py:15 indented from collections import defaultdict"

    # Test with alias but no attribute (straight import)
    import_obj = Import(
        line_number=20,
        indented=False,
        module="numpy",
        alias="np",
        file_path=Path("/script.py")
    )
    assert str(import_obj) == "/script.py:20 import numpy as np"


# LLM-generated content at query #19
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(1, False, "os")
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(1, False, "os", alias="operating_system")
    assert import_obj.statement() == "import os as operating_system"

    # Test cimport without alias
    import_obj = Import(1, False, "numpy", cimport=True)
    assert import_obj.statement() == "cimport numpy"

    # Test cimport with alias
    import_obj = Import(1, False, "numpy", alias="np", cimport=True)
    assert import_obj.statement() == "cimport numpy as np"

    # Test from import without alias
    import_obj = Import(1, False, "os", attribute="path")
    assert import_obj.statement() == "from os import path"

    # Test from import with alias
    import_obj = Import(1, False, "os", attribute="path", alias="p")
    assert import_obj.statement() == "from os import path as p"

    # Test from cimport without alias
    import_obj = Import(1, False, "numpy", attribute="array", cimport=True)
    assert import_obj.statement() == "from numpy cimport array"

    # Test from cimport with alias
    import_obj = Import(1, False, "numpy", attribute="array", alias="arr", cimport=True)
    assert import_obj.statement() == "from numpy cimport array as arr"


# LLM-generated content at query #20
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os"]
    expected = [Import(1, False, "os")]
    assert list(imports(input_stream)) == expected

    # Test import with alias
    input_stream = ["import numpy as np"]
    expected = [Import(1, False, "numpy", alias="np")]
    assert list(imports(input_stream)) == expected

    # Test from import
    input_stream = ["from sys import argv"]
    expected = [Import(1, False, "sys", attribute="argv")]
    assert list(imports(input_stream)) == expected

    # Test cimport
    input_stream = ["cimport numpy"]
    expected = [Import(1, False, "numpy", cimport=True)]
    assert list(imports(input_stream)) == expected

    # Test multiple imports
    input_stream = ["import os, sys"]
    expected = [Import(1, False, "os"), Import(1, False, "sys")]
    assert list(imports(input_stream)) == expected

    # Test indented import
    input_stream = ["    import os"]
    expected = [Import(1, True, "os")]
    assert list(imports(input_stream)) == expected

    # Test import with comment
    input_stream = ["import os  # comment"]
    expected = [Import(1, False, "os")]
    assert list(imports(input_stream)) == expected

    # Test multiline import
    input_stream = ["from collections import (OrderedDict,\n    defaultdict)"]
    expected = [
        Import(1, False, "collections", attribute="OrderedDict"),
        Import(2, False, "collections", attribute="defaultdict")
    ]
    assert list(imports(input_stream)) == expected

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy"]
    expected = [Import(1, False, "numpy")]
    assert list(imports(input_stream, config=config)) == expected

    # Test file path
    file_path = Path("/test.py")
    input_stream = ["import os"]
    expected = [Import(1, False, "os", file_path=file_path)]
    assert list(imports(input_stream, file_path=file_path)) == expected


# LLM-generated content at query #21
#--------------------------

```python
def test_Import___str__():
    # Test basic import without alias
    import_obj = Import(1, False, "os", None, None, False, None)
    assert str(import_obj) == ":1 import os"

    # Test basic import with alias
    import_obj = Import(2, False, "numpy", None, "np", False, None)
    assert str(import_obj) == ":2 import numpy as np"

    # Test cimport
    import_obj = Import(3, False, "module", None, None, True, None)
    assert str(import_obj) == ":3 cimport module"

    # Test from import
    import_obj = Import(4, False, "collections", "defaultdict", None, False, None)
    assert str(import_obj) == ":4 from collections import defaultdict"

    # Test from cimport with alias
    import_obj = Import(5, False, "libc", "stdio", "c_stdio", True, None)
    assert str(import_obj) == ":5 from libc cimport stdio as c_stdio"

    # Test indented import
    import_obj = Import(6, True, "sys", None, None, False, None)
    assert str(import_obj) == ":6 indented import sys"

    # Test with file path
    import_obj = Import(7, False, "pathlib", None, None, False, Path("/some/path.py"))
    assert str(import_obj) == "/some/path.py:7 import pathlib"


# LLM-generated content at query #22
#--------------------------

```python
def test_Import___str__():
    # Test with basic import
    import_obj = Import(line_number=1, indented=False, module="os")
    assert str(import_obj) == ":1 import os"

    # Test with indented import
    import_obj = Import(line_number=2, indented=True, module="sys")
    assert str(import_obj) == ":2 indented import sys"

    # Test with attribute
    import_obj = Import(line_number=3, indented=False, module="os", attribute="path")
    assert str(import_obj) == ":3 from os import path"

    # Test with alias
    import_obj = Import(line_number=4, indented=False, module="numpy", alias="np")
    assert str(import_obj) == ":4 import numpy as np"

    # Test with both attribute and alias
    import_obj = Import(line_number=5, indented=False, module="pandas", attribute="DataFrame", alias="pd")
    assert str(import_obj) == ":5 from pandas import DataFrame as pd"

    # Test with cimport
    import_obj = Import(line_number=6, indented=False, module="libc", cimport=True)
    assert str(import_obj) == ":6 cimport libc"

    # Test with file_path
    import_obj = Import(line_number=7, indented=False, module="sys", file_path=Path("/path/to/file.py"))
    assert str(import_obj) == "/path/to/file.py:7 import sys"

    # Test with all parameters
    import_obj = Import(
        line_number=8,
        indented=True,
        module="tensorflow",
        attribute="keras",
        alias="tf",
        cimport=True,
        file_path=Path("/path/to/file.py")
    )
    assert str(import_obj) == "/path/to/file.py:8 indented from tensorflow cimport keras as tf"


# LLM-generated content at query #23
#--------------------------

```python
def test_Import___str__():
    # Test basic import without alias
    import_obj = Import(line_number=1, indented=False, module="os", file_path=None)
    assert str(import_obj) == ":1 import os"

    # Test import with alias
    import_obj = Import(line_number=2, indented=True, module="numpy", alias="np", file_path=None)
    assert str(import_obj) == ":2 indented import numpy as np"

    # Test cimport
    import_obj = Import(line_number=3, indented=False, module="cython", cimport=True, file_path=None)
    assert str(import_obj) == ":3 cimport cython"

    # Test from import with attribute
    import_obj = Import(line_number=4, indented=False, module="os", attribute="path", file_path=None)
    assert str(import_obj) == ":4 from os import path"

    # Test from cimport with attribute and alias
    import_obj = Import(line_number=5, indented=True, module="libc", attribute="stdio", alias="libc_stdio", cimport=True, file_path=None)
    assert str(import_obj) == ":5 indented from libc cimport stdio as libc_stdio"

    # Test with file path
    import_obj = Import(line_number=6, indented=False, module="sys", file_path=Path("/path/to/file.py"))
    assert str(import_obj) == "/path/to/file.py:6 import sys"


# LLM-generated content at query #24
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute is None
    assert imports_list[0].alias is None
    assert imports_list[0].cimport is False

    # Test import with alias
    input_stream = ["import numpy as np"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias == "np"

    # Test from import
    input_stream = ["from collections import defaultdict"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "collections"
    assert imports_list[0].attribute == "defaultdict"

    # Test from import with alias
    input_stream = ["from pathlib import Path as P"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "pathlib"
    assert imports_list[0].attribute == "Path"
    assert imports_list[0].alias == "P"

    # Test multiple imports
    input_stream = ["import sys, os"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "sys"
    assert imports_list[1].module == "os"

    # Test cimport
    input_stream = ["cimport numpy"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].cimport is True

    # Test multiline import
    input_stream = ["from typing import (", "    List,"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "typing"
    assert imports_list[0].attribute == "List"

    # Test indented import
    input_stream = ["    import sys"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].indented is True

    # Test with file path
    file_path = Path("/test/file.py")
    input_stream = ["import os"]
    imports_list = list(imports(input_stream, file_path=file_path))
    assert len(imports_list) == 1
    assert imports_list[0].file_path == file_path

    # Test with config
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import os as os"]
    imports_list = list(imports(input_stream, config=config))
    assert len(imports_list) == 1
    assert imports_list[0].alias is None

    # Test top_only
    input_stream = ["import os", "def foo():", "    import sys"]
    imports_list = list(imports(input_stream, top_only=True))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"


# LLM-generated content at query #25
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="os", file_path=None)
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np", file_path=None)
    assert import_obj.statement() == "import numpy as np"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True, file_path=None)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", alias="cy", cimport=True, file_path=None)
    assert import_obj.statement() == "cimport cython as cy"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", file_path=None)
    assert import_obj.statement() == "from os import path"

    # Test from cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="cython", cimport=True, file_path=None)
    assert import_obj.statement() == "from cython cimport cython"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", alias="p", file_path=None)
    assert import_obj.statement() == "from os import path as p"

    # Test from cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="cython", alias="cy", cimport=True, file_path=None)
    assert import_obj.statement() == "from cython cimport cython as cy"


# LLM-generated content at query #26
#--------------------------

```python
def test_imports():
    # Test basic imports
    test_input = "import os\nimport sys\n"
    expected_output = [
        Import(1, False, "os"),
        Import(2, False, "sys"),
    ]
    assert list(imports(test_input.splitlines())) == expected_output

    # Test from imports
    test_input = "from os import path\nfrom sys import argv\n"
    expected_output = [
        Import(1, False, "os", "path"),
        Import(2, False, "sys", "argv"),
    ]
    assert list(imports(test_input.splitlines())) == expected_output

    # Test aliased imports
    test_input = "import numpy as np\nfrom pandas import DataFrame as df\n"
    expected_output = [
        Import(1, False, "numpy", alias="np"),
        Import(2, False, "pandas", "DataFrame", alias="df"),
    ]
    assert list(imports(test_input.splitlines())) == expected_output

    # Test cimport
    test_input = "cimport numpy\nfrom pandas cimport DataFrame\n"
    expected_output = [
        Import(1, False, "numpy", cimport=True),
        Import(2, False, "pandas", "DataFrame", cimport=True),
    ]
    assert list(imports(test_input.splitlines())) == expected_output

    # Test indented imports
    test_input = "def foo():\n    import os\n    from sys import argv\n"
    expected_output = [
        Import(2, True, "os"),
        Import(3, True, "sys", "argv"),
    ]
    assert list(imports(test_input.splitlines())) == expected_output

    # Test multiline imports
    test_input = "from os import (\n    path,\n    environ,\n)\n"
    expected_output = [
        Import(1, False, "os", "path"),
        Import(2, False, "os", "environ"),
    ]
    assert list(imports(test_input.splitlines())) == expected_output

    # Test comments
    test_input = "import os  # comment\n# comment\nimport sys\n"
    expected_output = [
        Import(1, False, "os"),
        Import(3, False, "sys"),
    ]
    assert list(imports(test_input.splitlines())) == expected_output

    # Test semicolon separated imports
    test_input = "import os; import sys\n"
    expected_output = [
        Import(1, False, "os"),
        Import(1, False, "sys"),
    ]
    assert list(imports(test_input.splitlines())) == expected_output

    # Test redundant aliases
    config = Config(remove_redundant_aliases=True)
    test_input = "import numpy as numpy\nfrom pandas import DataFrame as DataFrame\n"
    expected_output = [
        Import(1, False, "numpy"),
        Import(2, False, "pandas", "DataFrame"),
    ]
    assert list(imports(test_input.splitlines(), config=config)) == expected_output


# LLM-generated content at query #27
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].alias is None
    assert imports_list[0].attribute is None

    # Test import with alias
    input_stream = ["import numpy as np\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias == "np"

    # Test from import
    input_stream = ["from collections import defaultdict\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "collections"
    assert imports_list[0].attribute == "defaultdict"

    # Test from import with alias
    input_stream = ["from pathlib import Path as P\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "pathlib"
    assert imports_list[0].attribute == "Path"
    assert imports_list[0].alias == "P"

    # Test multiple imports
    input_stream = ["import sys, os\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "sys"
    assert imports_list[1].module == "os"

    # Test cimport
    input_stream = ["cimport numpy\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].cimport is True

    # Test indented import
    input_stream = ["    import os\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].indented is True

    # Test multiline import
    input_stream = ["from collections import (\n    defaultdict,\n    Counter\n)\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "collections"
    assert imports_list[0].attribute == "defaultdict"
    assert imports_list[1].module == "collections"
    assert imports_list[1].attribute == "Counter"

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy\n"]
    imports_list = list(imports(input_stream, config=config))
    assert len(imports_list) == 1
    assert imports_list[0].alias is None

    # Test comment handling
    input_stream = ["import os  # comment\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"

    # Test top_only parameter
    input_stream = ["import os\n", "def foo():\n", "    import sys\n"]
    imports_list = list(imports(input_stream, top_only=True))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"

    # Test file_path parameter
    file_path = Path("/test/path.py")
    input_stream = ["import os\n"]
    imports_list = list(imports(input_stream, file_path=file_path))
    assert len(imports_list) == 1
    assert imports_list[0].file_path == file_path


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = iter(["import os\n", "import sys\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[1].module == "sys"

    # Test import with alias
    input_stream = iter(["import numpy as np\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias == "np"

    # Test from import
    input_stream = iter(["from collections import defaultdict\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "collections"
    assert imports_list[0].attribute == "defaultdict"

    # Test from import with alias
    input_stream = iter(["from pathlib import Path as P\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "pathlib"
    assert imports_list[0].attribute == "Path"
    assert imports_list[0].alias == "P"

    # Test cimport
    input_stream = iter(["cimport numpy\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].cimport is True

    # Test multiline import
    input_stream = iter(["from collections import (\n", "    defaultdict,\n", "    Counter\n", ")\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "collections"
    assert imports_list[0].attribute == "defaultdict"
    assert imports_list[1].module == "collections"
    assert imports_list[1].attribute == "Counter"

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = iter(["import numpy as numpy\n"])
    imports_list = list(imports(input_stream, config=config))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias is None

    # Test indented import
    input_stream = iter(["    import os\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].indented is True

    # Test with file path
    file_path = Path("/test/path.py")
    input_stream = iter(["import os\n"])
    imports_list = list(imports(input_stream, file_path=file_path))
    assert len(imports_list) == 1
    assert imports_list[0].file_path == file_path

    # Test top_only
    input_stream = iter(["import os\n", "def foo():\n", "    import sys\n"])
    imports_list = list(imports(input_stream, top_only=True))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"

    # Test skipping lines
    input_stream = iter(["# comment\n", "import os\n", "\"\"\"docstring\"\"\"\n", "import sys\n"])
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[1].module == "sys"


# LLM-generated content at query #2
#--------------------------

```python
def test_imports():
    # Test basic imports
    test_input = "import os\nimport sys\n"
    expected = [
        Import(1, False, "os"),
        Import(2, False, "sys"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test from imports
    test_input = "from os import path\nfrom sys import argv\n"
    expected = [
        Import(1, False, "os", "path"),
        Import(2, False, "sys", "argv"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test cimport
    test_input = "cimport numpy\nfrom numpy cimport ndarray\n"
    expected = [
        Import(1, False, "numpy", cimport=True),
        Import(2, False, "numpy", "ndarray", cimport=True),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test aliases
    test_input = "import numpy as np\nfrom os import path as p\n"
    expected = [
        Import(1, False, "numpy", alias="np"),
        Import(2, False, "os", "path", alias="p"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test redundant aliases
    test_input = "import numpy as numpy\nfrom os import path as path\n"
    config = Config(remove_redundant_aliases=True)
    expected = [
        Import(1, False, "numpy"),
        Import(2, False, "os", "path"),
    ]
    assert list(imports(test_input.splitlines(), config=config)) == expected

    # Test multiline imports
    test_input = "from os import (\n    path,\n    sep\n)\n"
    expected = [
        Import(1, False, "os", "path"),
        Import(3, False, "os", "sep"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test indented imports
    test_input = "if True:\n    import sys\n"
    expected = [
        Import(2, True, "sys"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test comments
    test_input = "import os  # comment\nfrom sys import argv  # another comment\n"
    expected = [
        Import(1, False, "os"),
        Import(2, False, "sys", "argv"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test semicolon separated imports
    test_input = "import os; import sys\n"
    expected = [
        Import(1, False, "os"),
        Import(1, False, "sys"),
    ]
    assert list(imports(test_input.splitlines())) == expected

    # Test top_only
    test_input = "import os\ndef foo():\n    import sys\n"
    expected = [
        Import(1, False, "os"),
    ]
    assert list(imports(test_input.splitlines(), top_only=True)) == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_Import___str__():
    import_obj = Import(
        line_number=1,
        indented=False,
        module="os",
        attribute="path",
        alias="osp",
        cimport=False,
        file_path=Path("/test/path.py")
    )
    assert str(import_obj) == "/test/path.py:1 import os.path as osp"

    import_obj = Import(
        line_number=2,
        indented=True,
        module="sys",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=None
    )
    assert str(import_obj) == ":2 indented cimport sys"

    import_obj = Import(
        line_number=3,
        indented=False,
        module="numpy",
        attribute="array",
        alias=None,
        cimport=False,
        file_path=Path("script.py")
    )
    assert str(import_obj) == "script.py:3 import numpy.array"


# LLM-generated content at query #4
#--------------------------

```python
def test_imports():
    # Test basic imports
    input_stream = ["import os", "import sys"]
    expected = [
        Import(1, False, "os"),
        Import(2, False, "sys"),
    ]
    assert list(imports(input_stream)) == expected

    # Test from imports
    input_stream = ["from os import path", "from sys import argv"]
    expected = [
        Import(1, False, "os", "path"),
        Import(2, False, "sys", "argv"),
    ]
    assert list(imports(input_stream)) == expected

    # Test aliased imports
    input_stream = ["import numpy as np", "from pandas import DataFrame as df"]
    expected = [
        Import(1, False, "numpy", alias="np"),
        Import(2, False, "pandas", "DataFrame", alias="df"),
    ]
    assert list(imports(input_stream)) == expected

    # Test cimports
    input_stream = ["cimport numpy", "from cython cimport int"]
    expected = [
        Import(1, False, "numpy", cimport=True),
        Import(2, False, "cython", "int", cimport=True),
    ]
    assert list(imports(input_stream)) == expected

    # Test multiline imports
    input_stream = ["from os import (", "    path,", "    environ", ")"]
    expected = [
        Import(1, False, "os", "path"),
        Import(1, False, "os", "environ"),
    ]
    assert list(imports(input_stream)) == expected

    # Test indented imports
    input_stream = ["    import os", "        import sys"]
    expected = [
        Import(1, True, "os"),
        Import(2, True, "sys"),
    ]
    assert list(imports(input_stream)) == expected

    # Test comments and skipped lines
    input_stream = ["# This is a comment", "import os  # inline comment", "'''docstring'''", "import sys"]
    expected = [
        Import(2, False, "os"),
        Import(4, False, "sys"),
    ]
    assert list(imports(input_stream)) == expected

    # Test redundant aliases
    input_stream = ["import os as os", "from sys import argv as argv"]
    config = Config(remove_redundant_aliases=True)
    expected = [
        Import(1, False, "os"),
        Import(2, False, "sys", "argv"),
    ]
    assert list(imports(input_stream, config=config)) == expected

    # Test file path
    input_stream = ["import os"]
    file_path = Path("/path/to/file.py")
    expected = [
        Import(1, False, "os", file_path=file_path),
    ]
    assert list(imports(input_stream, file_path=file_path)) == expected

    # Test top_only
    input_stream = ["import os", "def foo():", "    import sys"]
    expected = [
        Import(1, False, "os"),
    ]
    assert list(imports(input_stream, top_only=True)) == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(1, False, "module", None, None, False, None)
    assert import_obj.statement() == "import module"

    # Test straight import with alias
    import_obj = Import(1, False, "module", None, "alias", False, None)
    assert import_obj.statement() == "import module as alias"

    # Test cimport without alias
    import_obj = Import(1, False, "module", None, None, True, None)
    assert import_obj.statement() == "cimport module"

    # Test cimport with alias
    import_obj = Import(1, False, "module", None, "alias", True, None)
    assert import_obj.statement() == "cimport module as alias"

    # Test from import without alias
    import_obj = Import(1, False, "module", "attribute", None, False, None)
    assert import_obj.statement() == "from module import attribute"

    # Test from import with alias
    import_obj = Import(1, False, "module", "attribute", "alias", False, None)
    assert import_obj.statement() == "from module import attribute as alias"

    # Test from cimport without alias
    import_obj = Import(1, False, "module", "attribute", None, True, None)
    assert import_obj.statement() == "from module cimport attribute"

    # Test from cimport with alias
    import_obj = Import(1, False, "module", "attribute", "alias", True, None)
    assert import_obj.statement() == "from module cimport attribute as alias"


# LLM-generated content at query #6
#--------------------------

```python
def test_Import___str__():
    # Test basic import with no alias
    import_obj = Import(line_number=1, indented=False, module="sys")
    assert str(import_obj) == ":1 import sys"

    # Test indented import with no alias
    import_obj = Import(line_number=5, indented=True, module="os")
    assert str(import_obj) == ":5 indented import os"

    # Test import with alias
    import_obj = Import(line_number=10, indented=False, module="numpy", alias="np")
    assert str(import_obj) == ":10 import numpy as np"

    # Test indented import with alias
    import_obj = Import(line_number=15, indented=True, module="pandas", alias="pd")
    assert str(import_obj) == ":15 indented import pandas as pd"

    # Test from import with no alias
    import_obj = Import(line_number=20, indented=False, module="collections", attribute="defaultdict")
    assert str(import_obj) == ":20 from collections import defaultdict"

    # Test from import with alias
    import_obj = Import(line_number=25, indented=False, module="typing", attribute="List", alias="TList")
    assert str(import_obj) == ":25 from typing import List as TList"

    # Test cimport with no alias
    import_obj = Import(line_number=30, indented=False, module="cython", cimport=True)
    assert str(import_obj) == ":30 cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=35, indented=False, module="cython", alias="cy", cimport=True)
    assert str(import_obj) == ":35 cimport cython as cy"

    # Test with file path
    import_obj = Import(line_number=40, indented=False, module="sys", file_path=Path("/path/to/file.py"))
    assert str(import_obj) == "/path/to/file.py:40 import sys"


# LLM-generated content at query #7
#--------------------------

```python
def test_Import___str__():
    # Test basic import without alias
    import_obj = Import(line_number=1, indented=False, module="os")
    assert str(import_obj) == ":1 import os"

    # Test basic import with alias
    import_obj = Import(line_number=2, indented=True, module="numpy", alias="np")
    assert str(import_obj) == ":2 indented import numpy as np"

    # Test from import without alias
    import_obj = Import(line_number=3, indented=False, module="sys", attribute="path")
    assert str(import_obj) == ":3 from sys import path"

    # Test from import with alias
    import_obj = Import(line_number=4, indented=True, module="collections", attribute="defaultdict", alias="dd")
    assert str(import_obj) == ":4 indented from collections import defaultdict as dd"

    # Test cimport
    import_obj = Import(line_number=5, indented=False, module="cython", cimport=True)
    assert str(import_obj) == ":5 cimport cython"

    # Test with file path
    import_obj = Import(line_number=6, indented=False, module="os", file_path=Path("/path/to/file.py"))
    assert str(import_obj) == "/path/to/file.py:6 import os"


# LLM-generated content at query #8
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="os")
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np")
    assert import_obj.statement() == "import numpy as np"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", alias="cy", cimport=True)
    assert import_obj.statement() == "cimport cython as cy"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path")
    assert import_obj.statement() == "from os import path"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", alias="p")
    assert import_obj.statement() == "from os import path as p"

    # Test from cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", cimport=True)
    assert import_obj.statement() == "from cython cimport view"

    # Test from cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", alias="v", cimport=True)
    assert import_obj.statement() == "from cython cimport view as v"


# LLM-generated content at query #9
#--------------------------

```python
def test_Import___str__():
    # Test basic import without alias
    import_obj = Import(line_number=1, indented=False, module="os")
    assert str(import_obj) == ":1 import os"

    # Test import with alias
    import_obj = Import(line_number=2, indented=True, module="numpy", alias="np")
    assert str(import_obj) == ":2 indented import numpy as np"

    # Test cimport
    import_obj = Import(line_number=3, indented=False, module="cython", cimport=True)
    assert str(import_obj) == ":3 cimport cython"

    # Test from import with attribute
    import_obj = Import(line_number=4, indented=False, module="sys", attribute="path")
    assert str(import_obj) == ":4 from sys import path"

    # Test from import with attribute and alias
    import_obj = Import(line_number=5, indented=True, module="collections", attribute="defaultdict", alias="dd")
    assert str(import_obj) == ":5 indented from collections import defaultdict as dd"

    # Test with file path
    import_obj = Import(line_number=6, indented=False, module="json", file_path=Path("/path/to/file.py"))
    assert str(import_obj) == "/path/to/file.py:6 import json"


# LLM-generated content at query #10
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="os", cimport=False)
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np", cimport=False)
    assert import_obj.statement() == "import numpy as np"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", alias="cy", cimport=True)
    assert import_obj.statement() == "cimport cython as cy"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", cimport=False)
    assert import_obj.statement() == "from os import path"

    # Test from cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="cfunc", cimport=True)
    assert import_obj.statement() == "from cython cimport cfunc"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", alias="p", cimport=False)
    assert import_obj.statement() == "from os import path as p"

    # Test from cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="cfunc", alias="cf", cimport=True)
    assert import_obj.statement() == "from cython cimport cfunc as cf"


# LLM-generated content at query #11
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os\n", "import sys\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[1].module == "sys"

    # Test import with alias
    input_stream = ["import numpy as np\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias == "np"

    # Test from import
    input_stream = ["from os import path\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"

    # Test from import with alias
    input_stream = ["from os import path as p\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"
    assert imports_list[0].alias == "p"

    # Test cimport
    input_stream = ["cimport numpy\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].cimport is True

    # Test multiline import
    input_stream = ["from os import (\n", "    path,\n", "    sys\n", ")\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"
    assert imports_list[1].module == "os"
    assert imports_list[1].attribute == "sys"

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy\n"]
    imports_list = list(imports(input_stream, config=config))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias is None

    # Test indented import
    input_stream = ["    import os\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].indented is True

    # Test top_only parameter
    input_stream = ["import os\n", "def foo():\n", "    import sys\n"]
    imports_list = list(imports(input_stream, top_only=True))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"

    # Test with file_path
    file_path = Path("/test/path")
    input_stream = ["import os\n"]
    imports_list = list(imports(input_stream, file_path=file_path))
    assert len(imports_list) == 1
    assert imports_list[0].file_path == file_path


# LLM-generated content at query #12
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="os")
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np")
    assert import_obj.statement() == "import numpy as np"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", alias="cy", cimport=True)
    assert import_obj.statement() == "cimport cython as cy"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path")
    assert import_obj.statement() == "from os import path"

    # Test from cimport without alias
    import_obj = Import(line_number=1, indented=False, module="libc", attribute="stdio", cimport=True)
    assert import_obj.statement() == "from libc cimport stdio"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", alias="p")
    assert import_obj.statement() == "from os import path as p"

    # Test from cimport with alias
    import_obj = Import(line_number=1, indented=False, module="libc", attribute="stdio", alias="s", cimport=True)
    assert import_obj.statement() == "from libc cimport stdio as s"


# LLM-generated content at query #13
#--------------------------

```python
def test_Import___str__():
    # Test basic import without alias
    import_obj = Import(line_number=1, indented=False, module="os", file_path=None)
    assert str(import_obj) == ":1 import os"

    # Test import with alias
    import_obj = Import(line_number=2, indented=True, module="numpy", alias="np", file_path=Path("test.py"))
    assert str(import_obj) == "test.py:2 indented import numpy as np"

    # Test from import with attribute
    import_obj = Import(line_number=3, indented=False, module="collections", attribute="defaultdict", file_path=None)
    assert str(import_obj) == ":3 from collections import defaultdict"

    # Test from import with attribute and alias
    import_obj = Import(line_number=4, indented=True, module="typing", attribute="List", alias="TList", file_path=Path("example.py"))
    assert str(import_obj) == "example.py:4 indented from typing import List as TList"

    # Test cimport
    import_obj = Import(line_number=5, indented=False, module="cython", cimport=True, file_path=None)
    assert str(import_obj) == ":5 cimport cython"

    # Test cimport with attribute
    import_obj = Import(line_number=6, indented=True, module="libc", attribute="stdio", cimport=True, file_path=Path("cy.py"))
    assert str(import_obj) == "cy.py:6 indented from libc cimport stdio"


# LLM-generated content at query #14
#--------------------------

```python
def test_Import___str__():
    # Test with all attributes
    import_obj = Import(
        line_number=10,
        indented=True,
        module="os",
        attribute="path",
        alias="osp",
        cimport=False,
        file_path=Path("/test/path.py")
    )
    assert str(import_obj) == "/test/path.py:10 indented from os cimport path as osp"

    # Test without alias
    import_obj = Import(
        line_number=5,
        indented=False,
        module="sys",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=None
    )
    assert str(import_obj) == ":5 cimport sys"

    # Test without attribute
    import_obj = Import(
        line_number=15,
        indented=True,
        module="collections",
        attribute=None,
        alias="coll",
        cimport=False,
        file_path=Path("example.py")
    )
    assert str(import_obj) == "example.py:15 indented import collections as coll"

    # Test without file_path
    import_obj = Import(
        line_number=20,
        indented=False,
        module="math",
        attribute="sqrt",
        alias=None,
        cimport=True,
        file_path=None
    )
    assert str(import_obj) == ":20 from math cimport sqrt"


# LLM-generated content at query #15
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os"]
    expected = [Import(1, False, "os")]
    assert list(imports(input_stream)) == expected

    # Test import with alias
    input_stream = ["import numpy as np"]
    expected = [Import(1, False, "numpy", alias="np")]
    assert list(imports(input_stream)) == expected

    # Test from import
    input_stream = ["from sys import argv"]
    expected = [Import(1, False, "sys", attribute="argv")]
    assert list(imports(input_stream)) == expected

    # Test cimport
    input_stream = ["cimport numpy"]
    expected = [Import(1, False, "numpy", cimport=True)]
    assert list(imports(input_stream)) == expected

    # Test multiple imports on one line
    input_stream = ["import os, sys"]
    expected = [Import(1, False, "os"), Import(1, False, "sys")]
    assert list(imports(input_stream)) == expected

    # Test indented import
    input_stream = ["    import os"]
    expected = [Import(1, True, "os")]
    assert list(imports(input_stream)) == expected

    # Test import with parentheses
    input_stream = ["from os import (\n    path,\n    sep\n)"]
    expected = [Import(1, False, "os", attribute="path"), Import(1, False, "os", attribute="sep")]
    assert list(imports(input_stream)) == expected

    # Test import with backslash
    input_stream = ["from os import path, \\\n    sep"]
    expected = [Import(1, False, "os", attribute="path"), Import(1, False, "os", attribute="sep")]
    assert list(imports(input_stream)) == expected

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy"]
    expected = [Import(1, False, "numpy")]
    assert list(imports(input_stream, config=config)) == expected

    # Test comment handling
    input_stream = ["import os # This is a comment"]
    expected = [Import(1, False, "os")]
    assert list(imports(input_stream)) == expected

    # Test empty input
    input_stream = []
    expected = []
    assert list(imports(input_stream)) == expected

    # Test non-import line
    input_stream = ["x = 1"]
    expected = []
    assert list(imports(input_stream)) == expected

    # Test top_only parameter
    input_stream = ["import os", "def foo():", "    import sys"]
    expected = [Import(1, False, "os")]
    assert list(imports(input_stream, top_only=True)) == expected


# LLM-generated content at query #16
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os\n", "import sys\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "os", None, None, False, None)
    assert imports_list[1] == Import(2, False, "sys", None, None, False, None)

    # Test from import
    input_stream = ["from os import path\n", "from sys import argv\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "os", "path", None, False, None)
    assert imports_list[1] == Import(2, False, "sys", "argv", None, False, None)

    # Test import with alias
    input_stream = ["import numpy as np\n", "import pandas as pd\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "numpy", None, "np", False, None)
    assert imports_list[1] == Import(2, False, "pandas", None, "pd", False, None)

    # Test from import with alias
    input_stream = ["from os import path as p\n", "from sys import argv as a\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "os", "path", "p", False, None)
    assert imports_list[1] == Import(2, False, "sys", "argv", "a", False, None)

    # Test cimport
    input_stream = ["cimport numpy as np\n", "cimport pandas as pd\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "numpy", None, "np", True, None)
    assert imports_list[1] == Import(2, False, "pandas", None, "pd", True, None)

    # Test multiline import
    input_stream = ["from os import (\n", "    path,\n", "    environ\n", ")\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "os", "path", None, False, None)
    assert imports_list[1] == Import(3, False, "os", "environ", None, False, None)

    # Test indented import
    input_stream = ["    import os\n", "    import sys\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, True, "os", None, None, False, None)
    assert imports_list[1] == Import(2, True, "sys", None, None, False, None)

    # Test with file path
    input_stream = ["import os\n", "import sys\n"]
    file_path = Path("/test/path")
    imports_list = list(imports(input_stream, file_path=file_path))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "os", None, None, False, file_path)
    assert imports_list[1] == Import(2, False, "sys", None, None, False, file_path)

    # Test with top_only
    input_stream = ["import os\n", "def func():\n", "    import sys\n"]
    imports_list = list(imports(input_stream, top_only=True))
    assert len(imports_list) == 1
    assert imports_list[0] == Import(1, False, "os", None, None, False, None)

    # Test with redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import os as os\n", "from sys import path as path\n"]
    imports_list = list(imports(input_stream, config=config))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "os", None, None, False, None)
    assert imports_list[1] == Import(2, False, "sys", "path", None, False, None)

    # Test with comments
    input_stream = ["import os  # comment\n", "import sys  # another comment\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "os", None, None, False, None)
    assert imports_list[1] == Import(2, False, "sys", None, None, False, None)

    # Test with semicolons
    input_stream = ["import os; import sys\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "os", None, None, False, None)
    assert imports_list[1] == Import(1, False, "sys", None, None, False, None)

    # Test with escaped newlines
    input_stream = ["import os \\\n", "    , sys\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "os", None, None, False, None)
    assert imports_list[1] == Import(2, False, "sys", None, None, False, None)

    # Test with parentheses in from import
    input_stream = ["from os import (path, environ)\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(1, False, "os", "path", None, False, None)
    assert imports_list[1] == Import(1, False, "os", "environ", None, False, None)

    # Test with cimport in from import
    input_stream = ["from os cimport path\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0] == Import(1, False, "os", "path", None, True, None)

    # Test with yield statement
    input_stream = ["yield\n", "import os\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0] == Import(2, False, "os", None, None, False, None)

    # Test with raise statement
    input_stream = ["raise\n", "import os\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0] == Import(2, False, "os", None, None, False, None)

    # Test with empty lines
    input_stream = ["\n", "import os\n", "\n", "import sys\n"]
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0] == Import(2, False, "os", None, None, False, None)
    assert imports_list[1] == Import(4, False, "sys", None, None, False, None)


# LLM-generated content at query #17
#--------------------------

```python
def test_Import___str__():
    # Test with all fields
    import_obj = Import(
        line_number=10,
        indented=True,
        module="os",
        attribute="path",
        alias="osp",
        cimport=False,
        file_path=Path("/tmp/test.py")
    )
    assert str(import_obj) == "/tmp/test.py:10 indented from os import path as osp"

    # Test without alias
    import_obj = Import(
        line_number=5,
        indented=False,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None
    )
    assert str(import_obj) == ":5 import sys"

    # Test with cimport
    import_obj = Import(
        line_number=15,
        indented=True,
        module="libc",
        attribute="stdio",
        alias=None,
        cimport=True,
        file_path=Path("test.py")
    )
    assert str(import_obj) == "test.py:15 indented from libc cimport stdio"

    # Test without attribute (straight import)
    import_obj = Import(
        line_number=20,
        indented=False,
        module="collections",
        attribute=None,
        alias="coll",
        cimport=False,
        file_path=Path("/path/to/file.py")
    )
    assert str(import_obj) == "/path/to/file.py:20 import collections as coll"


# LLM-generated content at query #18
#--------------------------

```python
def test_Import___str__():
    # Test basic import without alias
    import_obj = Import(line_number=1, indented=False, module="os", file_path=None)
    assert str(import_obj) == ":1 import os"

    # Test import with alias
    import_obj = Import(line_number=2, indented=True, module="numpy", alias="np", file_path=None)
    assert str(import_obj) == ":2 indented import numpy as np"

    # Test cimport
    import_obj = Import(line_number=3, indented=False, module="libc", cimport=True, file_path=None)
    assert str(import_obj) == ":3 cimport libc"

    # Test from import with attribute
    import_obj = Import(line_number=4, indented=False, module="collections", attribute="defaultdict", file_path=None)
    assert str(import_obj) == ":4 from collections import defaultdict"

    # Test from cimport with attribute and alias
    import_obj = Import(line_number=5, indented=True, module="libc", attribute="stdio", alias="cstdio", cimport=True, file_path=None)
    assert str(import_obj) == ":5 indented from libc cimport stdio as cstdio"

    # Test with file path
    import_obj = Import(line_number=6, indented=False, module="sys", file_path=Path("/path/to/file.py"))
    assert str(import_obj) == "/path/to/file.py:6 import sys"


# LLM-generated content at query #19
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os\n", "import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test import with alias
    input_stream = ["import numpy as np\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"

    # Test from import
    input_stream = ["from collections import defaultdict\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "defaultdict"

    # Test from import with alias
    input_stream = ["from pathlib import Path as P\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias == "P"

    # Test cimport
    input_stream = ["cimport numpy\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True

    # Test multiline import
    input_stream = ["from typing import (\n", "    List,\n", "    Dict,\n", ")\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "typing"
    assert result[0].attribute == "List"
    assert result[1].module == "typing"
    assert result[1].attribute == "Dict"

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import numpy as numpy\n"]
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias is None

    # Test indented import
    input_stream = ["    import os\n"]
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True

    # Test with file path
    file_path = Path("/test/path.py")
    input_stream = ["import os\n"]
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    # Test top_only flag
    input_stream = ["import os\n", "def foo():\n", "    import sys\n"]
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"


# LLM-generated content at query #20
#--------------------------

```python
def test_Import___str__():
    # Test case 1: Basic import without alias
    import_obj = Import(line_number=1, indented=False, module="os", file_path=None)
    assert str(import_obj) == ":1 import os"

    # Test case 2: Import with alias
    import_obj = Import(line_number=2, indented=False, module="numpy", alias="np", file_path=None)
    assert str(import_obj) == ":2 import numpy as np"

    # Test case 3: Indented import
    import_obj = Import(line_number=3, indented=True, module="sys", file_path=None)
    assert str(import_obj) == ":3 indented import sys"

    # Test case 4: Import with file path
    import_obj = Import(line_number=4, indented=False, module="pandas", file_path=Path("test.py"))
    assert str(import_obj) == "test.py:4 import pandas"

    # Test case 5: Cimport
    import_obj = Import(line_number=5, indented=False, module="cython", cimport=True, file_path=None)
    assert str(import_obj) == ":5 cimport cython"

    # Test case 6: From import with attribute
    import_obj = Import(line_number=6, indented=False, module="collections", attribute="defaultdict", file_path=None)
    assert str(import_obj) == ":6 from collections import defaultdict"

    # Test case 7: From import with attribute and alias
    import_obj = Import(line_number=7, indented=False, module="collections", attribute="defaultdict", alias="dd", file_path=None)
    assert str(import_obj) == ":7 from collections import defaultdict as dd"

    # Test case 8: Indented from import with file path
    import_obj = Import(line_number=8, indented=True, module="os.path", attribute="join", file_path=Path("script.py"))
    assert str(import_obj) == "script.py:8 indented from os.path import join"


# LLM-generated content at query #21
#--------------------------

```python
def test_imports():
    # Test basic import
    input_stream = ["import os\n", "import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test from import
    input_stream = ["from os import path\n", "from sys import argv\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "sys"
    assert result[1].attribute == "argv"

    # Test import with alias
    input_stream = ["import numpy as np\n", "import pandas as pd\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[1].module == "pandas"
    assert result[1].alias == "pd"

    # Test from import with alias
    input_stream = ["from os import path as p\n", "from sys import argv as a\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    assert result[1].module == "sys"
    assert result[1].attribute == "argv"
    assert result[1].alias == "a"

    # Test cimport
    input_stream = ["cimport numpy as np\n", "cimport pandas as pd\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[0].cimport == True
    assert result[1].module == "pandas"
    assert result[1].alias == "pd"
    assert result[1].cimport == True

    # Test multiline import
    input_stream = ["from os import (\n", "    path,\n", "    environ\n", ")\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"

    # Test indented import
    input_stream = ["    import os\n", "    import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].indented == True
    assert result[1].indented == True

    # Test comment handling
    input_stream = ["import os  # comment\n", "import sys  # comment\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test empty input
    input_stream = []
    result = list(imports(input_stream))
    assert len(result) == 0

    # Test non-import lines
    input_stream = ["x = 1\n", "y = 2\n"]
    result = list(imports(input_stream))
    assert len(result) == 0

    # Test mixed lines
    input_stream = ["import os\n", "x = 1\n", "import sys\n"]
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test top_only parameter
    input_stream = ["import os\n", "def foo():\n", "    import sys\n"]
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test file_path parameter
    input_stream = ["import os\n", "import sys\n"]
    file_path = Path("/path/to/file.py")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 2
    assert result[0].file_path == file_path
    assert result[1].file_path == file_path

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = ["import os as os\n", "import sys as sys\n"]
    result = list(imports(input_stream, config=config))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].alias is None
    assert result[1].module == "sys"
    assert result[1].alias is None


# LLM-generated content at query #22
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="os")
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np")
    assert import_obj.statement() == "import numpy as np"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path")
    assert import_obj.statement() == "from os import path"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", alias="p")
    assert import_obj.statement() == "from os import path as p"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", alias="cy", cimport=True)
    assert import_obj.statement() == "cimport cython as cy"

    # Test from cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", cimport=True)
    assert import_obj.statement() == "from cython cimport view"

    # Test from cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", alias="v", cimport=True)
    assert import_obj.statement() == "from cython cimport view as v"


# LLM-generated content at query #23
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="os")
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np")
    assert import_obj.statement() == "import numpy as np"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True, alias="cy")
    assert import_obj.statement() == "cimport cython as cy"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path")
    assert import_obj.statement() == "from os import path"

    # Test from cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", cimport=True)
    assert import_obj.statement() == "from cython cimport view"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", alias="p")
    assert import_obj.statement() == "from os import path as p"

    # Test from cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", cimport=True, alias="v")
    assert import_obj.statement() == "from cython cimport view as v"


# LLM-generated content at query #24
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(1, False, "module")
    assert import_obj.statement() == "import module"

    # Test straight import with alias
    import_obj = Import(1, False, "module", alias="alias")
    assert import_obj.statement() == "import module as alias"

    # Test cimport without alias
    import_obj = Import(1, False, "module", cimport=True)
    assert import_obj.statement() == "cimport module"

    # Test cimport with alias
    import_obj = Import(1, False, "module", alias="alias", cimport=True)
    assert import_obj.statement() == "cimport module as alias"

    # Test from import without alias
    import_obj = Import(1, False, "module", attribute="attribute")
    assert import_obj.statement() == "from module import attribute"

    # Test from cimport without alias
    import_obj = Import(1, False, "module", attribute="attribute", cimport=True)
    assert import_obj.statement() == "from module cimport attribute"

    # Test from import with alias
    import_obj = Import(1, False, "module", attribute="attribute", alias="alias")
    assert import_obj.statement() == "from module import attribute as alias"

    # Test from cimport with alias
    import_obj = Import(1, False, "module", attribute="attribute", alias="alias", cimport=True)
    assert import_obj.statement() == "from module cimport attribute as alias"


# LLM-generated content at query #25
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(line_number=1, indented=False, module="os", file_path=None)
    assert import_obj.statement() == "import os"

    # Test straight import with alias
    import_obj = Import(line_number=1, indented=False, module="numpy", alias="np", file_path=None)
    assert import_obj.statement() == "import numpy as np"

    # Test from import without alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", file_path=None)
    assert import_obj.statement() == "from os import path"

    # Test from import with alias
    import_obj = Import(line_number=1, indented=False, module="os", attribute="path", alias="p", file_path=None)
    assert import_obj.statement() == "from os import path as p"

    # Test cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", cimport=True, file_path=None)
    assert import_obj.statement() == "cimport cython"

    # Test cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", alias="cy", cimport=True, file_path=None)
    assert import_obj.statement() == "cimport cython as cy"

    # Test from cimport without alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", cimport=True, file_path=None)
    assert import_obj.statement() == "from cython cimport view"

    # Test from cimport with alias
    import_obj = Import(line_number=1, indented=False, module="cython", attribute="view", alias="v", cimport=True, file_path=None)
    assert import_obj.statement() == "from cython cimport view as v"


