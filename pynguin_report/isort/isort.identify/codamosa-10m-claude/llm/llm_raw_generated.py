####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Import___str__():
    """Test the __str__ method of the Import class."""
    # Test basic import without file path
    import_obj = Import(
        line_number=10,
        indented=False,
        module="os",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert import_obj.__str__() == ":10 import os"

    # Test import with file path
    import_obj = Import(
        line_number=5,
        indented=False,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("test.py"),
    )
    assert import_obj.__str__() == "test.py:5 import sys"

    # Test indented import
    import_obj = Import(
        line_number=15,
        indented=True,
        module="json",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("main.py"),
    )
    assert import_obj.__str__() == "main.py:15 indented import json"

    # Test from import with attribute
    import_obj = Import(
        line_number=20,
        indented=False,
        module="collections",
        attribute="defaultdict",
        alias=None,
        cimport=False,
        file_path=Path("app.py"),
    )
    assert import_obj.__str__() == "app.py:20 from collections import defaultdict"

    # Test from import with attribute and alias
    import_obj = Import(
        line_number=25,
        indented=False,
        module="numpy",
        attribute="array",
        alias="arr",
        cimport=False,
        file_path=Path("script.py"),
    )
    assert import_obj.__str__() == "script.py:25 from numpy import array as arr"

    # Test import with alias
    import_obj = Import(
        line_number=30,
        indented=False,
        module="pandas",
        attribute=None,
        alias="pd",
        cimport=False,
        file_path=Path("analysis.py"),
    )
    assert import_obj.__str__() == "analysis.py:30 import pandas as pd"

    # Test cimport
    import_obj = Import(
        line_number=35,
        indented=False,
        module="libc.stdlib",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=Path("cython_file.pyx"),
    )
    assert import_obj.__str__() == "cython_file.pyx:35 cimport libc.stdlib"

    # Test from cimport with attribute
    import_obj = Import(
        line_number=40,
        indented=True,
        module="libc.math",
        attribute="sin",
        alias=None,
        cimport=True,
        file_path=Path("math_module.pyx"),
    )
    assert import_obj.__str__() == "math_module.pyx:40 indented from libc.math cimport sin"

    # Test from cimport with attribute and alias
    import_obj = Import(
        line_number=45,
        indented=False,
        module="libc.stdio",
        attribute="printf",
        alias="print_func",
        cimport=True,
        file_path=Path("io.pyx"),
    )
    assert import_obj.__str__() == "io.pyx:45 from libc.stdio cimport printf as print_func"


# LLM-generated content at query #2
#--------------------------

```python
def test_Import___str__():
    """Test the __str__ method of Import class."""
    # Test basic import without file path
    import_obj = Import(
        line_number=1,
        indented=False,
        module="os",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert str(import_obj) == ":1 import os"

    # Test import with file path
    import_obj = Import(
        line_number=5,
        indented=False,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("/path/to/file.py"),
    )
    assert str(import_obj) == "/path/to/file.py:5 import sys"

    # Test indented import
    import_obj = Import(
        line_number=10,
        indented=True,
        module="json",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert str(import_obj) == ":10 indented import json"

    # Test from import with attribute
    import_obj = Import(
        line_number=3,
        indented=False,
        module="collections",
        attribute="defaultdict",
        alias=None,
        cimport=False,
        file_path=Path("test.py"),
    )
    assert str(import_obj) == "test.py:3 from collections import defaultdict"

    # Test from import with alias
    import_obj = Import(
        line_number=7,
        indented=False,
        module="numpy",
        attribute="array",
        alias="arr",
        cimport=False,
        file_path=Path("script.py"),
    )
    assert str(import_obj) == "script.py:7 from numpy import array as arr"

    # Test cimport
    import_obj = Import(
        line_number=2,
        indented=False,
        module="libc.math",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=None,
    )
    assert str(import_obj) == ":2 cimport libc.math"

    # Test from cimport with attribute
    import_obj = Import(
        line_number=4,
        indented=False,
        module="libc.stdlib",
        attribute="malloc",
        alias=None,
        cimport=True,
        file_path=Path("cython_file.pyx"),
    )
    assert str(import_obj) == "cython_file.pyx:4 from libc.stdlib cimport malloc"

    # Test indented from import with alias and file path
    import_obj = Import(
        line_number=15,
        indented=True,
        module="typing",
        attribute="Optional",
        alias="Opt",
        cimport=False,
        file_path=Path("module.py"),
    )
    assert str(import_obj) == "module.py:15 indented from typing import Optional as Opt"

    # Test straight import with alias
    import_obj = Import(
        line_number=6,
        indented=False,
        module="pandas",
        attribute=None,
        alias="pd",
        cimport=False,
        file_path=Path("data.py"),
    )
    assert str(import_obj) == "data.py:6 import pandas as pd"


# LLM-generated content at query #3
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].line_number == 1
    
    # Test from import
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports in one line
    input_stream = StringIO("import os, sys, re\n")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert result[0].module == "os"
    assert result[1].module == "sys"
    assert result[2].module == "re"
    
    # Test from import with multiple attributes
    input_stream = StringIO("from os import path, getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "getcwd"
    
    # Test indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "getcwd"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, \\\n    getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test with comments
    input_stream = StringIO("import os  # comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    test_path = Path("test.py")
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, file_path=test_path))
    assert len(result) == 1
    assert result[0].file_path == test_path
    
    # Test top_only parameter
    input_stream = StringIO("import os\ndef func():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skip comments and blank lines
    input_stream = StringIO("# comment\n\nimport os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test multiple statements on one line
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test nested module imports
    input_stream = StringIO("import os.path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os.path"


# LLM-generated content at query #4
#--------------------------

```python
def test_Import___str__():
    # Test with all fields populated
    imp = Import(
        line_number=42,
        indented=True,
        module="os.path",
        attribute="join",
        alias="path_join",
        cimport=False,
        file_path=Path("/home/user/test.py")
    )
    assert str(imp) == "/home/user/test.py:42 indented from os.path import join as path_join"

    # Test without file_path
    imp = Import(
        line_number=10,
        indented=False,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None
    )
    assert str(imp) == ":10 import sys"

    # Test with cimport
    imp = Import(
        line_number=5,
        indented=True,
        module="numpy",
        attribute="array",
        alias=None,
        cimport=True,
        file_path=Path("/tmp/module.pyx")
    )
    assert str(imp) == "/tmp/module.pyx:5 indented from numpy cimport array"

    # Test straight import with alias and cimport
    imp = Import(
        line_number=1,
        indented=False,
        module="collections",
        attribute=None,
        alias="col",
        cimport=False,
        file_path=Path("./src/main.py")
    )
    assert str(imp) == "./src/main.py:1 import collections as col"

    # Test with cimport, alias, and file_path
    imp = Import(
        line_number=99,
        indented=True,
        module="libc.stdlib",
        attribute="malloc",
        alias="mem_alloc",
        cimport=True,
        file_path=Path("/project/ext.pyx")
    )
    assert str(imp) == "/project/ext.pyx:99 indented from libc.stdlib cimport malloc as mem_alloc"

    # Test not indented
    imp = Import(
        line_number=7,
        indented=False,
        module="json",
        attribute="dumps",
        alias=None,
        cimport=False,
        file_path=Path("test.py")
    )
    assert str(imp) == "test.py:7 from json import dumps"


# LLM-generated content at query #5
#--------------------------

```python
def test_Import_statement():
    """Test the statement method of Import class."""
    # Test simple import
    import_obj = Import(
        line_number=1,
        indented=False,
        module="os",
    )
    assert import_obj.statement() == "import os"

    # Test import with alias
    import_obj = Import(
        line_number=2,
        indented=False,
        module="numpy",
        alias="np",
    )
    assert import_obj.statement() == "import numpy as np"

    # Test from import
    import_obj = Import(
        line_number=3,
        indented=False,
        module="os",
        attribute="path",
    )
    assert import_obj.statement() == "from os import path"

    # Test from import with alias
    import_obj = Import(
        line_number=4,
        indented=False,
        module="os",
        attribute="path",
        alias="ospath",
    )
    assert import_obj.statement() == "from os import path as ospath"

    # Test cimport
    import_obj = Import(
        line_number=5,
        indented=False,
        module="libc.stdlib",
        cimport=True,
    )
    assert import_obj.statement() == "cimport libc.stdlib"

    # Test from cimport
    import_obj = Import(
        line_number=6,
        indented=False,
        module="libc.stdlib",
        attribute="malloc",
        cimport=True,
    )
    assert import_obj.statement() == "from libc.stdlib cimport malloc"

    # Test from cimport with alias
    import_obj = Import(
        line_number=7,
        indented=False,
        module="libc.stdlib",
        attribute="malloc",
        alias="my_malloc",
        cimport=True,
    )
    assert import_obj.statement() == "from libc.stdlib cimport malloc as my_malloc"

    # Test cimport with alias
    import_obj = Import(
        line_number=8,
        indented=False,
        module="libc.stdlib",
        alias="stdlib",
        cimport=True,
    )
    assert import_obj.statement() == "cimport libc.stdlib as stdlib"


# LLM-generated content at query #6
#--------------------------

```python
def test_Import___str__():
    """Test the __str__ method of Import class."""
    # Test basic import without file_path
    imp = Import(
        line_number=10,
        indented=False,
        module="os",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert str(imp) == ":10 import os"

    # Test indented import without file_path
    imp = Import(
        line_number=5,
        indented=True,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert str(imp) == ":5 indented import sys"

    # Test from import with attribute
    imp = Import(
        line_number=15,
        indented=False,
        module="os.path",
        attribute="join",
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert str(imp) == ":15 from os.path import join"

    # Test from import with attribute and alias
    imp = Import(
        line_number=20,
        indented=False,
        module="collections",
        attribute="defaultdict",
        alias="dd",
        cimport=False,
        file_path=None,
    )
    assert str(imp) == ":20 from collections import defaultdict as dd"

    # Test import with alias
    imp = Import(
        line_number=25,
        indented=False,
        module="numpy",
        attribute=None,
        alias="np",
        cimport=False,
        file_path=None,
    )
    assert str(imp) == ":25 import numpy as np"

    # Test cimport
    imp = Import(
        line_number=30,
        indented=False,
        module="cython",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=None,
    )
    assert str(imp) == ":30 cimport cython"

    # Test with file_path
    imp = Import(
        line_number=35,
        indented=False,
        module="json",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("test.py"),
    )
    assert str(imp) == "test.py:35 import json"

    # Test indented with file_path and from import
    imp = Import(
        line_number=40,
        indented=True,
        module="typing",
        attribute="List",
        alias="L",
        cimport=False,
        file_path=Path("/home/user/module.py"),
    )
    assert str(imp) == "/home/user/module.py:40 indented from typing import List as L"

    # Test cimport with attribute
    imp = Import(
        line_number=45,
        indented=False,
        module="libc.stdlib",
        attribute="malloc",
        alias=None,
        cimport=True,
        file_path=Path("cython_module.pyx"),
    )
    assert str(imp) == "cython_module.pyx:45 from libc.stdlib cimport malloc"

    # Test indented cimport with alias
    imp = Import(
        line_number=50,
        indented=True,
        module="numpy",
        attribute="ndarray",
        alias="arr",
        cimport=True,
        file_path=Path("fast.pyx"),
    )
    assert str(imp) == "fast.pyx:50 indented from numpy cimport ndarray as arr"


# LLM-generated content at query #7
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    
    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias is None
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from collections import OrderedDict as OD\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"
    
    # Test multiple imports on one line
    input_stream = StringIO("from os import path, sep\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "sep"
    
    # Test indented import
    input_stream = StringIO("    import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    sep\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert all(r.module == "os" for r in result)
    assert result[0].attribute == "path"
    assert result[1].attribute == "sep"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    sep\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert all(r.module == "os" for r in result)
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test with comments
    input_stream = StringIO("import os  # operating system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test multiple statements on one line with semicolon
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test top_only parameter
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test file_path parameter
    test_path = Path("test.py")
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, file_path=test_path))
    assert result[0].file_path == test_path
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 1\nimport os\ny = 2\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test line numbers are correct
    input_stream = StringIO("# comment\nimport os\nimport sys\n")
    result = list(imports(input_stream))
    assert result[0].line_number == 2
    assert result[1].line_number == 3


# LLM-generated content at query #8
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "sys"
    assert result[1].line_number == 2
    
    # Test from import
    input_stream = StringIO("from os import path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports on one line
    input_stream = StringIO("from os import path, sep")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "sep"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    sep\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, \\\n    sep")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test indented import
    input_stream = StringIO("    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test cimport
    input_stream = StringIO("cimport numpy")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test with comments
    input_stream = StringIO("import os  # comment")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 1\nimport os\ny = 2")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only parameter
    input_stream = StringIO("import os\n\ndef func():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    input_stream = StringIO("import os")
    result = list(imports(input_stream, file_path=Path("test.py")))
    assert result[0].file_path == Path("test.py")
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test semicolon-separated statements
    input_stream = StringIO("import os; import sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test line number tracking
    input_stream = StringIO("# comment\nimport os\n\nimport sys")
    result = list(imports(input_stream))
    assert result[0].line_number == 2
    assert result[1].line_number == 4


# LLM-generated content at query #9
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test simple import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[1].module == "sys"
    assert result[1].line_number == 2
    
    # Test from import
    input_stream = StringIO("from os import path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports from same module
    input_stream = StringIO("from os import path, getcwd")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "getcwd"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test line continuation with backslash
    input_stream = StringIO("import os, \\\n    sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test indented import
    input_stream = StringIO("    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test cimport
    input_stream = StringIO("cimport numpy")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test top_only parameter stops at function definition
    input_stream = StringIO("import os\n\ndef foo():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    input_stream = StringIO("import os")
    result = list(imports(input_stream, file_path=Path("test.py")))
    assert len(result) == 1
    assert result[0].file_path == Path("test.py")
    
    # Test with comments
    input_stream = StringIO("import os  # comment")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test semicolon-separated statements
    input_stream = StringIO("import os; import sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test non-redundant alias kept
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as operating_system")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias == "operating_system"


# LLM-generated content at query #10
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    
    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias is None
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from collections import OrderedDict as OD\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"
    
    # Test multiple imports
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test from import with multiple attributes
    input_stream = StringIO("from os import path, sep\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "sep"
    
    # Test indented import
    input_stream = StringIO("    import json\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    sep\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].attribute == "sep"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    sep\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "sep"
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    assert result[0].cimport is True
    
    # Test with comments
    input_stream = StringIO("import os  # operating system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skip lines (quotes)
    input_stream = StringIO('"""\nfrom fake import module\n"""\nimport real\n')
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "real"
    
    # Test top_only parameter
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test file_path parameter
    input_stream = StringIO("import os\n")
    test_path = Path("test.py")
    result = list(imports(input_stream, file_path=test_path))
    assert result[0].file_path == test_path
    
    # Test multiple statements on one line
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test redundant alias removal
    input_stream = StringIO("import os as os\n")
    config = Config(remove_redundant_aliases=True)
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test with config
    config = Config()
    input_stream = StringIO("import asyncio\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1


# LLM-generated content at query #11
#--------------------------

```python
def test_Import___str__():
    """Test the __str__ method of Import class."""
    # Test basic import without file_path
    import_obj = Import(
        line_number=10,
        indented=False,
        module="os",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert import_obj.__str__() == ":10 import os"

    # Test import with file_path
    import_obj = Import(
        line_number=5,
        indented=False,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("/path/to/file.py"),
    )
    assert import_obj.__str__() == "/path/to/file.py:5 import sys"

    # Test indented import
    import_obj = Import(
        line_number=15,
        indented=True,
        module="json",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert import_obj.__str__() == ":15 indented import json"

    # Test from import with attribute
    import_obj = Import(
        line_number=20,
        indented=False,
        module="collections",
        attribute="defaultdict",
        alias=None,
        cimport=False,
        file_path=Path("test.py"),
    )
    assert import_obj.__str__() == "test.py:20 from collections import defaultdict"

    # Test from import with attribute and alias
    import_obj = Import(
        line_number=25,
        indented=False,
        module="typing",
        attribute="Dict",
        alias="DictType",
        cimport=False,
        file_path=None,
    )
    assert import_obj.__str__() == ":25 from typing import Dict as DictType"

    # Test cimport without attribute
    import_obj = Import(
        line_number=30,
        indented=False,
        module="numpy",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=None,
    )
    assert import_obj.__str__() == ":30 cimport numpy"

    # Test cimport with attribute
    import_obj = Import(
        line_number=35,
        indented=False,
        module="libc.stdlib",
        attribute="malloc",
        alias=None,
        cimport=True,
        file_path=Path("cython_file.pyx"),
    )
    assert import_obj.__str__() == "cython_file.pyx:35 from libc.stdlib cimport malloc"

    # Test indented import with alias
    import_obj = Import(
        line_number=40,
        indented=True,
        module="pandas",
        attribute=None,
        alias="pd",
        cimport=False,
        file_path=Path("analysis.py"),
    )
    assert import_obj.__str__() == "analysis.py:40 indented import pandas as pd"

    # Test indented from import with attribute and alias
    import_obj = Import(
        line_number=45,
        indented=True,
        module="matplotlib.pyplot",
        attribute="plot",
        alias="plt_plot",
        cimport=False,
        file_path=None,
    )
    assert import_obj.__str__() == ":45 indented from matplotlib.pyplot import plot as plt_plot"


# LLM-generated content at query #12
#--------------------------

```python
def test_Import___str__():
    """Test the __str__ method of Import class."""
    
    # Test basic import without file path
    import_obj = Import(
        line_number=10,
        indented=False,
        module="os",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None
    )
    assert import_obj.__str__() == ":10 import os"
    
    # Test indented import
    import_obj = Import(
        line_number=5,
        indented=True,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None
    )
    assert import_obj.__str__() == ":5 indented import sys"
    
    # Test from import with attribute
    import_obj = Import(
        line_number=15,
        indented=False,
        module="os.path",
        attribute="join",
        alias=None,
        cimport=False,
        file_path=None
    )
    assert import_obj.__str__() == ":15 from os.path import join"
    
    # Test from import with alias
    import_obj = Import(
        line_number=20,
        indented=False,
        module="numpy",
        attribute="array",
        alias="arr",
        cimport=False,
        file_path=None
    )
    assert import_obj.__str__() == ":20 from numpy import array as arr"
    
    # Test cimport
    import_obj = Import(
        line_number=25,
        indented=False,
        module="libc.stdlib",
        attribute="malloc",
        alias=None,
        cimport=True,
        file_path=None
    )
    assert import_obj.__str__() == ":25 from libc.stdlib cimport malloc"
    
    # Test import with file path
    import_obj = Import(
        line_number=30,
        indented=False,
        module="json",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("/path/to/file.py")
    )
    assert import_obj.__str__() == "/path/to/file.py:30 import json"
    
    # Test indented import with file path and alias
    import_obj = Import(
        line_number=8,
        indented=True,
        module="collections",
        attribute="defaultdict",
        alias="dd",
        cimport=False,
        file_path=Path("test.py")
    )
    assert import_obj.__str__() == "test.py:8 indented from collections import defaultdict as dd"
    
    # Test cimport with file path and alias
    import_obj = Import(
        line_number=12,
        indented=True,
        module="cython",
        attribute="parallel",
        alias="par",
        cimport=True,
        file_path=Path("cython_file.pyx")
    )
    assert import_obj.__str__() == "cython_file.pyx:12 indented from cython cimport parallel as par"


# LLM-generated content at query #13
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "sys"
    assert result[1].line_number == 2
    
    # Test from import
    input_stream = StringIO("from os import path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports from same module
    input_stream = StringIO("from os import path, environ")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test indented import
    input_stream = StringIO("    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    environ\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "environ"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    environ")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport numpy")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    assert result[0].cimport is True
    
    # Test with comments
    input_stream = StringIO("import os  # comment")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test file_path parameter
    test_path = Path("test.py")
    input_stream = StringIO("import os")
    result = list(imports(input_stream, file_path=test_path))
    assert result[0].file_path == test_path
    
    # Test top_only parameter
    input_stream = StringIO("import os\n\ndef func():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test empty file
    input_stream = StringIO("")
    result = list(imports(input_stream))
    assert len(result) == 0
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 5\nimport os\ny = 10")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test multiple statements on one line
    input_stream = StringIO("import os; import sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test statement() method
    imp = Import(1, False, "os", None, None, False, None)
    assert imp.statement() == "import os"
    
    imp = Import(1, False, "os", "path", None, False, None)
    assert imp.statement() == "from os import path"
    
    imp = Import(1, False, "os", "path", "p", False, None)
    assert imp.statement() == "from os import path as p"
    
    imp = Import(1, False, "numpy", None, "np", False, None)
    assert imp.statement() == "import numpy as np"
    
    imp = Import(1, False, "libc.stdlib", "malloc", None, True, None)
    assert imp.statement() == "from libc.stdlib cimport malloc"


# LLM-generated content at query #14
#--------------------------

```python
def test_Import_statement():
    """Test the statement method of the Import class."""
    
    # Test simple import
    import_obj = Import(
        line_number=1,
        indented=False,
        module="os",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert import_obj.statement() == "import os"
    
    # Test import with alias
    import_obj = Import(
        line_number=2,
        indented=False,
        module="numpy",
        attribute=None,
        alias="np",
        cimport=False,
        file_path=None,
    )
    assert import_obj.statement() == "import numpy as np"
    
    # Test from import
    import_obj = Import(
        line_number=3,
        indented=False,
        module="os",
        attribute="path",
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert import_obj.statement() == "from os import path"
    
    # Test from import with alias
    import_obj = Import(
        line_number=4,
        indented=False,
        module="os",
        attribute="path",
        alias="p",
        cimport=False,
        file_path=None,
    )
    assert import_obj.statement() == "from os import path as p"
    
    # Test cimport
    import_obj = Import(
        line_number=5,
        indented=False,
        module="libc.stdlib",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=None,
    )
    assert import_obj.statement() == "cimport libc.stdlib"
    
    # Test from cimport
    import_obj = Import(
        line_number=6,
        indented=False,
        module="libc.stdlib",
        attribute="malloc",
        alias=None,
        cimport=True,
        file_path=None,
    )
    assert import_obj.statement() == "from libc.stdlib cimport malloc"
    
    # Test from cimport with alias
    import_obj = Import(
        line_number=7,
        indented=False,
        module="libc.stdlib",
        attribute="malloc",
        alias="my_malloc",
        cimport=True,
        file_path=None,
    )
    assert import_obj.statement() == "from libc.stdlib cimport malloc as my_malloc"


# LLM-generated content at query #15
#--------------------------

```python
def test_Import_statement():
    """Test the statement method of Import class."""
    # Test simple import
    imp = Import(
        line_number=1,
        indented=False,
        module="os",
    )
    assert imp.statement() == "import os"

    # Test import with alias
    imp = Import(
        line_number=2,
        indented=False,
        module="numpy",
        alias="np",
    )
    assert imp.statement() == "import numpy as np"

    # Test from import
    imp = Import(
        line_number=3,
        indented=False,
        module="collections",
        attribute="defaultdict",
    )
    assert imp.statement() == "from collections import defaultdict"

    # Test from import with alias
    imp = Import(
        line_number=4,
        indented=False,
        module="collections",
        attribute="defaultdict",
        alias="dd",
    )
    assert imp.statement() == "from collections import defaultdict as dd"

    # Test cimport
    imp = Import(
        line_number=5,
        indented=False,
        module="libc.stdlib",
        cimport=True,
    )
    assert imp.statement() == "cimport libc.stdlib"

    # Test from cimport
    imp = Import(
        line_number=6,
        indented=False,
        module="libc.stdlib",
        attribute="malloc",
        cimport=True,
    )
    assert imp.statement() == "from libc.stdlib cimport malloc"

    # Test from cimport with alias
    imp = Import(
        line_number=7,
        indented=False,
        module="libc.stdlib",
        attribute="malloc",
        alias="my_malloc",
        cimport=True,
    )
    assert imp.statement() == "from libc.stdlib cimport malloc as my_malloc"

    # Test cimport with alias
    imp = Import(
        line_number=8,
        indented=False,
        module="libc.math",
        alias="math",
        cimport=True,
    )
    assert imp.statement() == "cimport libc.math as math"


# LLM-generated content at query #16
#--------------------------

```python
def test_Import___str__():
    """Test the __str__ method of Import class."""
    # Test basic import without file_path
    imp = Import(
        line_number=10,
        indented=False,
        module="os",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert str(imp) == ":10 import os"

    # Test import with file_path
    imp = Import(
        line_number=5,
        indented=False,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("/path/to/file.py"),
    )
    assert str(imp) == "/path/to/file.py:5 import sys"

    # Test indented import
    imp = Import(
        line_number=15,
        indented=True,
        module="json",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert str(imp) == ":15 indented import json"

    # Test from import with attribute
    imp = Import(
        line_number=20,
        indented=False,
        module="collections",
        attribute="defaultdict",
        alias=None,
        cimport=False,
        file_path=Path("test.py"),
    )
    assert str(imp) == "test.py:20 from collections import defaultdict"

    # Test from import with attribute and alias
    imp = Import(
        line_number=25,
        indented=False,
        module="typing",
        attribute="List",
        alias="ListType",
        cimport=False,
        file_path=None,
    )
    assert str(imp) == ":25 from typing import List as ListType"

    # Test import with alias
    imp = Import(
        line_number=30,
        indented=False,
        module="numpy",
        attribute=None,
        alias="np",
        cimport=False,
        file_path=Path("script.py"),
    )
    assert str(imp) == "script.py:30 import numpy as np"

    # Test cimport
    imp = Import(
        line_number=35,
        indented=False,
        module="cython",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=None,
    )
    assert str(imp) == ":35 cimport cython"

    # Test indented cimport with attribute and alias
    imp = Import(
        line_number=40,
        indented=True,
        module="libc.stdlib",
        attribute="malloc",
        alias="memory_alloc",
        cimport=True,
        file_path=Path("cython_file.pyx"),
    )
    assert str(imp) == "cython_file.pyx:40 indented from libc.stdlib cimport malloc as memory_alloc"


# LLM-generated content at query #17
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    
    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from collections import OrderedDict as OD\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"
    
    # Test multiple imports on one line
    input_stream = StringIO("from os import path, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from typing import (\n    List,\n    Dict,\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "List"
    assert result[1].attribute == "Dict"
    
    # Test backslash continuation
    input_stream = StringIO("from os import \\\n    path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test indented import
    input_stream = StringIO("    import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].attribute == "malloc"
    
    # Test with comments
    input_stream = StringIO("import os  # system module\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test multiple statements on one line
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test top_only parameter
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    test_path = Path("test.py")
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, file_path=test_path))
    assert result[0].file_path == test_path
    
    # Test skipping lines with triple quotes
    input_stream = StringIO('"""\nimport os\n"""\nimport sys\n')
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "sys"
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test nested module import
    input_stream = StringIO("import os.path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os.path"
    
    # Test star import
    input_stream = StringIO("from os import *\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].attribute == "*"


# LLM-generated content at query #18
#--------------------------

```python
def test_Import___str__():
    # Test with all parameters
    import_obj = Import(
        line_number=10,
        indented=True,
        module="os.path",
        attribute="join",
        alias="path_join",
        cimport=False,
        file_path=Path("/home/user/script.py")
    )
    assert str(import_obj) == "/home/user/script.py:10 indented from os.path import join as path_join"

    # Test without file_path
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

    # Test with indented=True but no file_path
    import_obj = Import(
        line_number=15,
        indented=True,
        module="collections",
        attribute="defaultdict",
        alias=None,
        cimport=False,
        file_path=None
    )
    assert str(import_obj) == ":15 indented from collections import defaultdict"

    # Test with cimport=True
    import_obj = Import(
        line_number=3,
        indented=False,
        module="numpy",
        attribute="array",
        alias="np_array",
        cimport=True,
        file_path=Path("/tmp/test.pyx")
    )
    assert str(import_obj) == "/tmp/test.pyx:3 from numpy cimport array as np_array"

    # Test with straight cimport (no attribute)
    import_obj = Import(
        line_number=1,
        indented=False,
        module="libc.stdlib",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=Path("/code/module.pyx")
    )
    assert str(import_obj) == "/code/module.pyx:1 cimport libc.stdlib"

    # Test with indented=False explicitly
    import_obj = Import(
        line_number=20,
        indented=False,
        module="json",
        attribute="loads",
        alias=None,
        cimport=False,
        file_path=Path("/app/main.py")
    )
    assert str(import_obj) == "/app/main.py:20 from json import loads"


# LLM-generated content at query #19
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test simple import
    input_stream = StringIO("import os\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].attribute is None
    assert result[1].module == "sys"
    
    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from collections import OrderedDict as OD\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"
    
    # Test multiple imports from same module
    input_stream = StringIO("from os import path, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test indented import
    input_stream = StringIO("    import json\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from typing import (\n    List,\n    Dict\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "List"
    assert result[1].attribute == "Dict"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test with comments
    input_stream = StringIO("import os  # operating system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only parameter stops at function definition
    input_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    input_stream = StringIO("import os\n")
    test_path = Path("test_file.py")
    result = list(imports(input_stream, file_path=test_path))
    assert len(result) == 1
    assert result[0].file_path == test_path
    
    # Test multiple statements on one line separated by semicolon
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test skip lines (docstrings)
    input_stream = StringIO('"""\nModule docstring\n"""\nimport os\n')
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test line numbers with multiple imports
    input_stream = StringIO("import os\n\nimport sys\n")
    result = list(imports(input_stream))
    assert result[0].line_number == 1
    assert result[1].line_number == 3
    
    # Test empty input
    input_stream = StringIO("")
    result = list(imports(input_stream))
    assert len(result) == 0


# LLM-generated content at query #20
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    
    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].line_number == 1
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from collections import OrderedDict as OD\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"
    
    # Test multiple imports on one line
    input_stream = StringIO("from os import path, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test indented import
    input_stream = StringIO("    import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    environ\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "environ"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test with comments
    input_stream = StringIO("import os  # comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only parameter - should stop at function definition
    input_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test file_path parameter
    input_stream = StringIO("import os\n")
    test_path = Path("test.py")
    result = list(imports(input_stream, file_path=test_path))
    assert result[0].file_path == test_path
    
    # Test multiple statements on one line
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 1\nimport os\ny = 2\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with triple-quoted strings
    input_stream = StringIO('"""\nModule docstring\nimport fake\n"""\nimport os\n')
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"


# LLM-generated content at query #21
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "sys"
    
    # Test from import
    input_stream = StringIO("from pathlib import Path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from collections import OrderedDict as OD")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"
    
    # Test multiple imports in one line
    input_stream = StringIO("from os import path, environ")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test indented import
    input_stream = StringIO("    import json")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    environ\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    environ")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport cython")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "cython"
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    assert result[0].cimport is True
    
    # Test with file_path
    file_path = Path("test.py")
    input_stream = StringIO("import os")
    result = list(imports(input_stream, file_path=file_path))
    assert result[0].file_path == file_path
    
    # Test top_only flag stops at function definition
    input_stream = StringIO("import os\ndef foo():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with comments
    input_stream = StringIO("import os  # comment")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 5\nimport os\ny = 10")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test multiple statements on one line
    input_stream = StringIO("import os; import sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test redundant alias NOT removed
    config = Config(remove_redundant_aliases=False)
    input_stream = StringIO("import os as os")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias == "os"


# LLM-generated content at query #22
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic straight imports
    input_stream = StringIO("import os\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].attribute is None
    assert result[1].module == "sys"
    
    # Test from imports
    input_stream = StringIO("from os import path\nfrom sys import argv\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "sys"
    assert result[1].attribute == "argv"
    
    # Test imports with aliases
    input_stream = StringIO("import numpy as np\nfrom os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[1].module == "os"
    assert result[1].attribute == "path"
    assert result[1].alias == "p"
    
    # Test multiline imports with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "getcwd"
    
    # Test multiline imports with backslash
    input_stream = StringIO("from os import path, \\\n    getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test indented imports
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test cimport
    input_stream = StringIO("cimport numpy\nfrom libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    assert result[1].cimport is True
    assert result[1].module == "libc.stdlib"
    assert result[1].attribute == "malloc"
    
    # Test multiple imports on one line
    input_stream = StringIO("import os, sys, json\n")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert result[0].module == "os"
    assert result[1].module == "sys"
    assert result[2].module == "json"
    
    # Test with comments
    input_stream = StringIO("import os  # comment\nfrom sys import argv  # another comment\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test top_only stops at function definitions
    input_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    input_stream = StringIO("import os\n")
    file_path = Path("test.py")
    result = list(imports(input_stream, file_path=file_path))
    assert result[0].file_path == file_path
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\nfrom sys import argv as argv\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 2
    assert result[0].alias is None
    assert result[1].alias is None
    
    # Test from import with multiple attributes
    input_stream = StringIO("from os import path, getcwd, listdir\n")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert [r.attribute for r in result] == ["path", "getcwd", "listdir"]
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 1\nimport os\ny = 2\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test empty input
    input_stream = StringIO("")
    result = list(imports(input_stream))
    assert len(result) == 0


# LLM-generated content at query #23
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test simple import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].line_number == 1
    
    # Test from import
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports on one line
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test from import with multiple attributes
    input_stream = StringIO("from os import path, getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test skip_line with comments
    input_stream = StringIO("import os  # comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test multiple statements on one line separated by semicolon
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test top_only parameter stops at function definition
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only parameter with class definition
    input_stream = StringIO("import os\nclass MyClass:\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    
    # Test file_path parameter
    input_stream = StringIO("import os\n")
    test_path = Path("test_file.py")
    result = list(imports(input_stream, file_path=test_path))
    assert len(result) == 1
    assert result[0].file_path == test_path
    
    # Test line number tracking
    input_stream = StringIO("import os\n\nimport sys\n")
    result = list(imports(input_stream))
    assert result[0].line_number == 1
    assert result[1].line_number == 3
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test from import redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("from os import path as path\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test skip lines with yield
    input_stream = StringIO("yield\n    x\nimport os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test empty input
    input_stream = StringIO("")
    result = list(imports(input_stream))
    assert len(result) == 0


# LLM-generated content at query #24
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    from pathlib import Path
    
    # Test basic import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[0].cimport is False

    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias is None

    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"

    # Test from import with alias
    input_stream = StringIO("from os.path import join as path_join\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os.path"
    assert result[0].attribute == "join"
    assert result[0].alias == "path_join"

    # Test multiple imports on one line
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test from import with multiple attributes
    input_stream = StringIO("from os import path, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"

    # Test indented import
    input_stream = StringIO("    import json\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True

    # Test multiline import with parentheses
    input_stream = StringIO("from module import (\n    func1,\n    func2\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "module"
    assert result[0].attribute == "func1"
    assert result[1].module == "module"
    assert result[1].attribute == "func2"

    # Test multiline import with backslash
    input_stream = StringIO("from module import \\\n    func1, func2\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "func1"
    assert result[1].attribute == "func2"

    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"

    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"

    # Test with file_path
    input_stream = StringIO("import os\n")
    file_path = Path("test.py")
    result = list(imports(input_stream, file_path=file_path))
    assert result[0].file_path == file_path

    # Test top_only stops at function definition
    input_stream = StringIO("import os\n\ndef func():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test with comments
    input_stream = StringIO("import os  # this is a comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test multiple statements on one line
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test line number tracking
    input_stream = StringIO("# comment\nimport os\nimport sys\n")
    result = list(imports(input_stream))
    assert result[0].line_number == 2
    assert result[1].line_number == 3

    # Test skip lines with yield
    input_stream = StringIO("yield\nimport os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None

    # Test redundant alias in from import
    input_stream = StringIO("from os import path as path\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None


# LLM-generated content at query #25
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[1].module == "sys"
    assert result[1].line_number == 2
    
    # Test from import
    input_stream = StringIO("from os import path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports on one line
    input_stream = StringIO("import os, sys, json")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert result[0].module == "os"
    assert result[1].module == "sys"
    assert result[2].module == "json"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "getcwd"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    getcwd")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test indented import
    input_stream = StringIO("    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test cimport
    input_stream = StringIO("cimport numpy")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].attribute == "malloc"
    
    # Test with file_path
    input_stream = StringIO("import os")
    file_path = Path("test.py")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path
    
    # Test top_only flag
    input_stream = StringIO("import os\n\ndef foo():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with comments
    input_stream = StringIO("import os  # comment")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test multiple statements on one line
    input_stream = StringIO("import os; import sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test skip lines with triple quotes
    input_stream = StringIO('"""\nimport fake\n"""\nimport os')
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test non-redundant alias kept
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as operating_system")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias == "operating_system"


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pathlib import Path
from io import StringIO
from isort.stdstream import imports, Import
from isort.settings import Config


def test_imports():
    """Test the imports function with various import scenarios."""
    
    # Test simple import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[0].cimport is False

    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias is None

    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"

    # Test from import with alias
    input_stream = StringIO("from collections import OrderedDict as OD\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"

    # Test multiple imports on one line
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test multiple from imports
    input_stream = StringIO("from os import path, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"

    # Test indented import
    input_stream = StringIO("    import json\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    assert result[0].module == "json"

    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    environ\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"

    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "environ"

    # Test import with comment
    input_stream = StringIO("import os  # operating system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"

    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"

    # Test file_path parameter
    input_stream = StringIO("import sys\n")
    file_path = Path("/test/file.py")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    # Test top_only parameter stops at statement declarations
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test skipping non-import lines
    input_stream = StringIO("x = 5\nimport os\ny = 10\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test line numbers are 1-indexed
    input_stream = StringIO("\nimport os\n")
    result = list(imports(input_stream))
    assert result[0].line_number == 2

    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None

    # Test multiple imports with mixed aliases
    input_stream = StringIO("import os as o, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].alias == "o"
    assert result[1].module == "sys"
    assert result[1].alias is None

    # Test statement method
    import_obj = Import(
        line_number=1,
        indented=False,
        module="os",
        attribute=None,
        alias=None,
        cimport=False
    )
    assert import_obj.statement() == "import os"

    import_obj = Import(
        line_number=1,
        indented=False,
        module="collections",
        attribute="OrderedDict",
        alias="OD",
        cimport=False
    )
    assert import_obj.statement() == "from collections import OrderedDict as OD"

    # Test __str__ method
    import_obj = Import(
        line_number=5,
        indented=True,
        module="os",
        file_path=Path("test.py")
    )
    assert "test.py:5" in str(import_obj)
    assert "indented" in str(import_obj)
    assert "import os" in str(import_obj)


# LLM-generated content at query #2
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test simple import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "sys"
    assert result[1].line_number == 2
    
    # Test from import
    input_stream = StringIO("from os import path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports on one line
    input_stream = StringIO("from os import path, sep")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "sep"
    
    # Test indented import
    input_stream = StringIO("    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    sep\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "sep"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, \\\n    sep")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport numpy")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    assert result[0].cimport is True
    
    # Test with comments
    input_stream = StringIO("import os  # comment\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    
    # Test skip line with quotes
    input_stream = StringIO('"""\nimport os\n"""\nimport sys')
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "sys"
    
    # Test top_only parameter
    input_stream = StringIO("import os\n\ndef foo():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    input_stream = StringIO("import os")
    file_path = Path("test.py")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path
    
    # Test empty input
    input_stream = StringIO("")
    result = list(imports(input_stream))
    assert len(result) == 0
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test multiple imports with aliases
    input_stream = StringIO("from os import path as p, sep as s")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    assert result[1].attribute == "sep"
    assert result[1].alias == "s"
    
    # Test statement method
    import_obj = Import(
        line_number=1,
        indented=False,
        module="os",
        attribute="path",
        alias="p"
    )
    assert import_obj.statement() == "from os import path as p"
    
    # Test __str__ method
    import_obj = Import(
        line_number=1,
        indented=False,
        module="os",
        file_path=Path("test.py")
    )
    assert "test.py:1" in str(import_obj)
    assert "import os" in str(import_obj)


# LLM-generated content at query #3
#--------------------------

```python
def test_Import___str__():
    """Test the __str__ method of the Import class."""
    # Test with all parameters
    import_obj = Import(
        line_number=10,
        indented=True,
        module="os.path",
        attribute="join",
        alias="path_join",
        cimport=False,
        file_path=Path("/home/user/test.py")
    )
    assert str(import_obj) == "/home/user/test.py:10 indented from os.path import join as path_join"

    # Test without indentation
    import_obj = Import(
        line_number=5,
        indented=False,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("/home/user/main.py")
    )
    assert str(import_obj) == "/home/user/main.py:5 import sys"

    # Test with cimport
    import_obj = Import(
        line_number=3,
        indented=False,
        module="numpy",
        attribute="array",
        alias=None,
        cimport=True,
        file_path=Path("/home/user/cython_module.pyx")
    )
    assert str(import_obj) == "/home/user/cython_module.pyx:3 from numpy cimport array"

    # Test without file_path
    import_obj = Import(
        line_number=15,
        indented=True,
        module="collections",
        attribute="defaultdict",
        alias="dd",
        cimport=False,
        file_path=None
    )
    assert str(import_obj) == ":15 indented from collections import defaultdict as dd"

    # Test with indented and cimport
    import_obj = Import(
        line_number=20,
        indented=True,
        module="mymodule",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=Path("/test/file.pyx")
    )
    assert str(import_obj) == "/test/file.pyx:20 indented cimport mymodule"

    # Test straight import with alias and file_path
    import_obj = Import(
        line_number=1,
        indented=False,
        module="numpy",
        attribute=None,
        alias="np",
        cimport=False,
        file_path=Path("/project/main.py")
    )
    assert str(import_obj) == "/project/main.py:1 import numpy as np"


# LLM-generated content at query #4
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].attribute is None
    assert result[1].module == "sys"
    
    # Test from import
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports from same module
    input_stream = StringIO("from os import path, sep\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "sep"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    sep\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "sep"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, \\\n    sep\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].attribute == "malloc"
    
    # Test with file_path
    input_stream = StringIO("import os\n")
    file_path = Path("test.py")
    result = list(imports(input_stream, file_path=file_path))
    assert result[0].file_path == file_path
    
    # Test top_only stops at function definition
    input_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with comments
    input_stream = StringIO("import os  # operating system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test semicolon separated statements
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("from os import path as path\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test redundant alias not removed when config is False
    config = Config(remove_redundant_aliases=False)
    input_stream = StringIO("from os import path as path\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias == "path"
    
    # Test nested module import
    input_stream = StringIO("import os.path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os.path"
    
    # Test empty and comment-only lines
    input_stream = StringIO("\n# comment\nimport os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"


# LLM-generated content at query #5
#--------------------------

```python
def test_Import_statement():
    # Test straight import without alias
    import_obj = Import(1, False, "os")
    assert import_obj.statement() == "import os"
    
    # Test straight import with alias
    import_obj = Import(1, False, "os", alias="operating_system")
    assert import_obj.statement() == "import os as operating_system"
    
    # Test from import without alias
    import_obj = Import(1, False, "os", attribute="path")
    assert import_obj.statement() == "from os import path"
    
    # Test from import with alias
    import_obj = Import(1, False, "os", attribute="path", alias="p")
    assert import_obj.statement() == "from os import path as p"
    
    # Test cimport without attribute
    import_obj = Import(1, False, "numpy", cimport=True)
    assert import_obj.statement() == "cimport numpy"
    
    # Test cimport with alias
    import_obj = Import(1, False, "numpy", alias="np", cimport=True)
    assert import_obj.statement() == "cimport numpy as np"
    
    # Test cimport from import
    import_obj = Import(1, False, "libc.stdlib", attribute="malloc", cimport=True)
    assert import_obj.statement() == "from libc.stdlib cimport malloc"
    
    # Test cimport from import with alias
    import_obj = Import(1, False, "libc.stdlib", attribute="malloc", alias="mem_alloc", cimport=True)
    assert import_obj.statement() == "from libc.stdlib cimport malloc as mem_alloc"
    
    # Test indented import (indented flag doesn't affect statement)
    import_obj = Import(5, True, "sys")
    assert import_obj.statement() == "import sys"
    
    # Test with complex module name
    import_obj = Import(1, False, "package.subpackage.module")
    assert import_obj.statement() == "import package.subpackage.module"
    
    # Test from import with complex module name
    import_obj = Import(1, False, "package.subpackage.module", attribute="MyClass")
    assert import_obj.statement() == "from package.subpackage.module import MyClass"


# LLM-generated content at query #6
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "sys"
    assert result[1].line_number == 2
    
    # Test from import
    input_stream = StringIO("from pathlib import Path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os.path import join as path_join")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os.path"
    assert result[0].attribute == "join"
    assert result[0].alias == "path_join"
    
    # Test multiple imports in one statement
    input_stream = StringIO("from collections import Counter, defaultdict")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "collections"
    assert result[0].attribute == "Counter"
    assert result[1].module == "collections"
    assert result[1].attribute == "defaultdict"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from typing import (\n    Dict,\n    List\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "Dict"
    assert result[1].attribute == "List"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, \\\n    environ")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "environ"
    
    # Test indented import
    input_stream = StringIO("    import json")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test import with comment
    input_stream = StringIO("import re  # regular expressions")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "re"
    
    # Test cimport
    input_stream = StringIO("cimport numpy")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test skip comments and blank lines
    input_stream = StringIO("# comment\nimport os\n\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test top_only parameter
    input_stream = StringIO("import os\n\ndef foo():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    input_stream = StringIO("import os")
    file_path = Path("test.py")
    result = list(imports(input_stream, file_path=file_path))
    assert result[0].file_path == file_path
    
    # Test statement method
    input_stream = StringIO("import os")
    result = list(imports(input_stream))
    assert result[0].statement() == "import os"
    
    input_stream = StringIO("from pathlib import Path")
    result = list(imports(input_stream))
    assert result[0].statement() == "from pathlib import Path"
    
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert result[0].statement() == "import numpy as np"
    
    # Test __str__ method
    input_stream = StringIO("import os")
    result = list(imports(input_stream))
    str_repr = str(result[0])
    assert "import os" in str_repr
    assert ":1" in str_repr


# LLM-generated content at query #7
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic straight imports
    input_stream = StringIO("import os\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].attribute is None
    assert result[1].module == "sys"
    
    # Test from imports
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    
    # Test imports with aliases
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from imports with aliases
    input_stream = StringIO("from collections import OrderedDict as OD\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"
    
    # Test multiple imports from same module
    input_stream = StringIO("from os import path, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test multiline imports with parentheses
    input_stream = StringIO("from os import (\n    path,\n    environ\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test indented imports
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test with comments
    input_stream = StringIO("import os  # comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test cimports
    input_stream = StringIO("cimport cython\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "cython"
    
    # Test from cimports
    input_stream = StringIO("from libc.stdio cimport printf\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdio"
    assert result[0].attribute == "printf"
    
    # Test top_only parameter
    input_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    test_path = Path("test.py")
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, file_path=test_path))
    assert result[0].file_path == test_path
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 1\nimport os\ny = 2\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test line continuation with backslash
    input_stream = StringIO("import os, \\\n    sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test multiple statements on one line
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test nested module imports
    input_stream = StringIO("from os.path import join\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os.path"
    assert result[0].attribute == "join"


# LLM-generated content at query #8
#--------------------------

```python
def test_Import___str__():
    """Test the __str__ method of Import class."""
    # Test basic import without file path
    import_obj = Import(
        line_number=10,
        indented=False,
        module="os",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert import_obj.__str__() == ":10 import os"

    # Test import with file path
    import_obj = Import(
        line_number=5,
        indented=False,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("test.py"),
    )
    assert import_obj.__str__() == "test.py:5 import sys"

    # Test indented import
    import_obj = Import(
        line_number=15,
        indented=True,
        module="json",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None,
    )
    assert import_obj.__str__() == ":15 indented import json"

    # Test from import with attribute
    import_obj = Import(
        line_number=20,
        indented=False,
        module="collections",
        attribute="defaultdict",
        alias=None,
        cimport=False,
        file_path=Path("app.py"),
    )
    assert import_obj.__str__() == "app.py:20 from collections import defaultdict"

    # Test from import with attribute and alias
    import_obj = Import(
        line_number=25,
        indented=False,
        module="numpy",
        attribute="array",
        alias="arr",
        cimport=False,
        file_path=Path("script.py"),
    )
    assert import_obj.__str__() == "script.py:25 from numpy import array as arr"

    # Test cimport
    import_obj = Import(
        line_number=30,
        indented=False,
        module="libc.stdlib",
        attribute=None,
        alias=None,
        cimport=True,
        file_path=Path("cython_module.pyx"),
    )
    assert import_obj.__str__() == "cython_module.pyx:30 cimport libc.stdlib"

    # Test cimport with attribute
    import_obj = Import(
        line_number=35,
        indented=True,
        module="libc.math",
        attribute="sin",
        alias="sine",
        cimport=True,
        file_path=Path("math.pyx"),
    )
    assert import_obj.__str__() == "math.pyx:35 indented from libc.math cimport sin as sine"

    # Test import with alias (straight import)
    import_obj = Import(
        line_number=40,
        indented=False,
        module="pandas",
        attribute=None,
        alias="pd",
        cimport=False,
        file_path=None,
    )
    assert import_obj.__str__() == ":40 import pandas as pd"


# LLM-generated content at query #9
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    
    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os.path import join as path_join\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os.path"
    assert result[0].attribute == "join"
    assert result[0].alias == "path_join"
    
    # Test multiple imports
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test multiple from imports
    input_stream = StringIO("from pathlib import Path, PurePath\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[1].module == "pathlib"
    assert result[1].attribute == "PurePath"
    
    # Test indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from pathlib import (\n    Path,\n    PurePath\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "Path"
    assert result[1].attribute == "PurePath"
    
    # Test multiline import with backslash
    input_stream = StringIO("from pathlib import \\\n    Path, \\\n    PurePath\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test with comments
    input_stream = StringIO("import os  # system module\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 1\nimport os\nprint('hello')\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only parameter
    input_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    file_path = Path("test.py")
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path
    
    # Test statement method
    imp = Import(1, False, "os")
    assert imp.statement() == "import os"
    
    imp = Import(1, False, "pathlib", "Path")
    assert imp.statement() == "from pathlib import Path"
    
    imp = Import(1, False, "numpy", alias="np")
    assert imp.statement() == "import numpy as np"
    
    imp = Import(1, False, "libc.stdlib", "malloc", cimport=True)
    assert imp.statement() == "from libc.stdlib cimport malloc"
    
    # Test __str__ method
    imp = Import(5, True, "os", file_path=Path("test.py"))
    str_repr = str(imp)
    assert "test.py:5" in str_repr
    assert "indented" in str_repr
    assert "import os" in str_repr


# LLM-generated content at query #10
#--------------------------

```python
def test_Import___str__():
    """Test the __str__ method of the Import class."""
    # Test with all fields populated
    import_obj = Import(
        line_number=10,
        indented=False,
        module="os",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("/path/to/file.py")
    )
    assert str(import_obj) == "/path/to/file.py:10 import os"

    # Test with indented import
    import_obj = Import(
        line_number=5,
        indented=True,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("/path/to/file.py")
    )
    assert str(import_obj) == "/path/to/file.py:5 indented import sys"

    # Test with from import
    import_obj = Import(
        line_number=15,
        indented=False,
        module="os.path",
        attribute="join",
        alias=None,
        cimport=False,
        file_path=Path("/path/to/file.py")
    )
    assert str(import_obj) == "/path/to/file.py:15 from os.path import join"

    # Test with alias
    import_obj = Import(
        line_number=20,
        indented=False,
        module="numpy",
        attribute=None,
        alias="np",
        cimport=False,
        file_path=Path("/path/to/file.py")
    )
    assert str(import_obj) == "/path/to/file.py:20 import numpy as np"

    # Test with from import and alias
    import_obj = Import(
        line_number=25,
        indented=False,
        module="collections",
        attribute="defaultdict",
        alias="dd",
        cimport=False,
        file_path=Path("/path/to/file.py")
    )
    assert str(import_obj) == "/path/to/file.py:25 from collections import defaultdict as dd"

    # Test with cimport
    import_obj = Import(
        line_number=30,
        indented=False,
        module="numpy",
        attribute="ndarray",
        alias=None,
        cimport=True,
        file_path=Path("/path/to/file.py")
    )
    assert str(import_obj) == "/path/to/file.py:30 from numpy cimport ndarray"

    # Test with no file_path
    import_obj = Import(
        line_number=35,
        indented=False,
        module="json",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None
    )
    assert str(import_obj) == ":35 import json"

    # Test with indented and no file_path
    import_obj = Import(
        line_number=40,
        indented=True,
        module="re",
        attribute=None,
        alias="regex",
        cimport=False,
        file_path=None
    )
    assert str(import_obj) == ":40 indented import re as regex"

    # Test with cimport and alias
    import_obj = Import(
        line_number=45,
        indented=True,
        module="libc.stdlib",
        attribute="malloc",
        alias="mem_alloc",
        cimport=True,
        file_path=Path("cython_file.pyx")
    )
    assert str(import_obj) == "cython_file.pyx:45 indented from libc.stdlib cimport malloc as mem_alloc"


# LLM-generated content at query #11
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    
    # Test from import
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].line_number == 1
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports on one line
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test from import with multiple attributes
    input_stream = StringIO("from os import path, sep\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "sep"
    
    # Test indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    sep\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "sep"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, \\\n    sep\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "sep"
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc cimport math\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "libc"
    assert result[0].attribute == "math"
    assert result[0].cimport is True
    
    # Test with comments
    input_stream = StringIO("import os  # comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skipped lines (docstring)
    input_stream = StringIO('"""\nModule docstring\n"""\nimport os\n')
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].line_number == 4
    
    # Test top_only parameter
    input_stream = StringIO("import os\n\ndef func():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    input_stream = StringIO("import os\n")
    test_path = Path("test.py")
    result = list(imports(input_stream, file_path=test_path))
    assert len(result) == 1
    assert result[0].file_path == test_path
    
    # Test with semicolon-separated statements
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test empty input
    input_stream = StringIO("")
    result = list(imports(input_stream))
    assert len(result) == 0
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test redundant from import alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("from os import path as path\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None


# LLM-generated content at query #12
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    
    # Test from import
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports on single line
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test multiple from imports
    input_stream = StringIO("from os import path, sep\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "sep"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    sep\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "sep"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, \\\n    sep\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert result[0].indented is True
    
    # Test import with comment
    input_stream = StringIO("import os  # comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skip comments and blank lines
    input_stream = StringIO("# comment\nimport os\n\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test top_only stops at function definition
    input_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only stops at class definition
    input_stream = StringIO("import os\n\nclass Foo:\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    
    # Test with file_path
    input_stream = StringIO("import os\n")
    file_path = Path("test.py")
    result = list(imports(input_stream, file_path=file_path))
    assert result[0].file_path == file_path
    
    # Test redundant alias removal with config
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test from import redundant alias
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("from os import path as path\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test multiple statements separated by semicolon
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test nested module import
    input_stream = StringIO("import os.path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os.path"
    
    # Test line number tracking
    input_stream = StringIO("import os\nimport sys\nimport json\n")
    result = list(imports(input_stream))
    assert result[0].line_number == 1
    assert result[1].line_number == 2
    assert result[2].line_number == 3


# LLM-generated content at query #13
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    
    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias is None
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from collections import OrderedDict as OD\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"
    
    # Test multiple imports from same module
    input_stream = StringIO("from os import path, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test indented import
    input_stream = StringIO("    import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    environ\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "environ"
    
    # Test import with comment
    input_stream = StringIO("import os  # operating system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    assert result[0].cimport is True
    
    # Test top_only parameter stops at function definition
    input_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test file_path parameter
    test_path = Path("test.py")
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, file_path=test_path))
    assert result[0].file_path == test_path
    
    # Test multiple imports on one line separated by semicolon
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test skipping lines with quotes
    input_stream = StringIO('"""\nimport fake\n"""\nimport real\n')
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "real"
    
    # Test Import.statement() method
    imp = Import(
        line_number=1,
        indented=False,
        module="os",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=None
    )
    assert imp.statement() == "import os"
    
    imp = Import(
        line_number=1,
        indented=False,
        module="os",
        attribute="path",
        alias="p",
        cimport=False,
        file_path=None
    )
    assert imp.statement() == "from os import path as p"
    
    # Test Import.__str__() method
    imp = Import(
        line_number=5,
        indented=True,
        module="sys",
        attribute=None,
        alias=None,
        cimport=False,
        file_path=Path("test.py")
    )
    assert "test.py:5" in str(imp)
    assert "indented" in str(imp)
    assert "import sys" in str(imp)


# LLM-generated content at query #14
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test basic straight imports
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].attribute is None
    assert result[1].module == "sys"
    
    # Test from imports
    input_stream = StringIO("from pathlib import Path\nfrom typing import List")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[1].module == "typing"
    assert result[1].attribute == "List"
    
    # Test imports with aliases
    input_stream = StringIO("import numpy as np\nfrom collections import defaultdict as dd")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[1].module == "collections"
    assert result[1].attribute == "defaultdict"
    assert result[1].alias == "dd"
    
    # Test multiline imports with parentheses
    input_stream = StringIO("from os import (\n    path,\n    environ\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test multiline imports with backslash
    input_stream = StringIO("from sys import \\\n    argv, \\\n    exit")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "sys"
    assert result[0].attribute == "argv"
    assert result[1].module == "sys"
    assert result[1].attribute == "exit"
    
    # Test indented imports
    input_stream = StringIO("if True:\n    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    assert result[0].module == "os"
    
    # Test multiple imports on one line
    input_stream = StringIO("import os; import sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test import with comment
    input_stream = StringIO("import os  # important module")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test from import with multiple attributes
    input_stream = StringIO("from os import path, environ, getcwd")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert all(r.module == "os" for r in result)
    assert [r.attribute for r in result] == ["path", "environ", "getcwd"]
    
    # Test top_only parameter
    input_stream = StringIO("import os\n\ndef func():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test cimport
    input_stream = StringIO("cimport numpy\nfrom libc.stdlib cimport malloc")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    assert result[1].cimport is True
    assert result[1].module == "libc.stdlib"
    assert result[1].attribute == "malloc"
    
    # Test file_path parameter
    test_path = Path("test.py")
    input_stream = StringIO("import os")
    result = list(imports(input_stream, file_path=test_path))
    assert len(result) == 1
    assert result[0].file_path == test_path
    
    # Test skipping yield statements
    input_stream = StringIO("yield\nimport os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test empty file
    input_stream = StringIO("")
    result = list(imports(input_stream))
    assert len(result) == 0
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\nfrom sys import argv as argv")
    result = list(imports(input_stream, config=config))
    assert len(result) == 2
    assert result[0].alias is None
    assert result[1].alias is None
    
    # Test non-redundant alias preservation
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as operating_system")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias == "operating_system"


# LLM-generated content at query #15
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test simple import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    
    # Test from import
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports
    input_stream = StringIO("import os, sys, json\n")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert result[0].module == "os"
    assert result[1].module == "sys"
    assert result[2].module == "json"
    
    # Test multiple from imports
    input_stream = StringIO("from os import path, getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "getcwd"
    
    # Test indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "getcwd"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    assert result[0].cimport is True
    
    # Test with comments
    input_stream = StringIO("import os  # comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 5\nimport os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only parameter
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test file_path parameter
    test_path = Path("test.py")
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, file_path=test_path))
    assert len(result) == 1
    assert result[0].file_path == test_path
    
    # Test with semicolon separated statements
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test redundant from alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("from os import path as path\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None


# LLM-generated content at query #16
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test simple import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[0].cimport is False
    
    # Test from import
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None
    
    # Test import with alias
    input_stream = StringIO("import os as operating_system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias == "operating_system"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports from same module
    input_stream = StringIO("from os import path, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test indented import
    input_stream = StringIO("    import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    environ\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "environ"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, \\\n    environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "environ"
    
    # Test multiple imports on one line
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc cimport stdlib\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "libc"
    assert result[0].attribute == "stdlib"
    assert result[0].cimport is True
    
    # Test with comments
    input_stream = StringIO("import os  # this is a comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test line numbers
    input_stream = StringIO("import os\n\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].line_number == 1
    assert result[1].line_number == 3
    
    # Test top_only parameter
    input_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    test_path = Path("test.py")
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, file_path=test_path))
    assert len(result) == 1
    assert result[0].file_path == test_path
    
    # Test empty file
    input_stream = StringIO("")
    result = list(imports(input_stream))
    assert len(result) == 0
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 5\nimport os\nprint(x)\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"


# LLM-generated content at query #17
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].attribute is None
    assert result[1].module == "sys"
    assert result[1].line_number == 2
    
    # Test from import
    input_stream = StringIO("from pathlib import Path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias is None
    
    # Test import with alias
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from collections import OrderedDict as OD")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"
    
    # Test multiple imports in one line
    input_stream = StringIO("from os import path, environ")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    environ\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "environ"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].attribute == "path"
    
    # Test indented import
    input_stream = StringIO("    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test cimport
    input_stream = StringIO("cimport numpy")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test skipping comments
    input_stream = StringIO("import os  # this is a comment")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only parameter
    input_stream = StringIO("import os\n\ndef foo():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test file_path parameter
    test_path = Path("test.py")
    input_stream = StringIO("import os")
    result = list(imports(input_stream, file_path=test_path))
    assert len(result) == 1
    assert result[0].file_path == test_path
    
    # Test with semicolon-separated statements
    input_stream = StringIO("import os; import sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test redundant alias removal with config
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test non-redundant alias is kept
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as operating_system")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias == "operating_system"


# LLM-generated content at query #18
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "sys"
    assert result[1].line_number == 2
    
    # Test from import
    input_stream = StringIO("from pathlib import Path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    
    # Test import with alias
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from collections import OrderedDict as OD")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"
    
    # Test multiple imports from same module
    input_stream = StringIO("from os import path, environ, getcwd")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert all(r.module == "os" for r in result)
    assert [r.attribute for r in result] == ["path", "environ", "getcwd"]
    
    # Test indented import
    input_stream = StringIO("    import json")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from collections import (\n    namedtuple,\n    defaultdict\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "namedtuple"
    assert result[1].attribute == "defaultdict"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, \\\n    environ")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test import with comment
    input_stream = StringIO("import os  # for operating system")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test cimport
    input_stream = StringIO("cimport numpy")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc, free")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "libc.stdlib"
    assert result[0].cimport is True
    
    # Test with file_path
    input_stream = StringIO("import os")
    file_path = Path("/test/file.py")
    result = list(imports(input_stream, file_path=file_path))
    assert result[0].file_path == file_path
    
    # Test top_only parameter stops at function definition
    input_stream = StringIO("import os\ndef foo():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skip of non-import lines
    input_stream = StringIO("x = 5\nimport os\ny = 10")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test semicolon separated statements
    input_stream = StringIO("import os; import sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test redundant alias removal with config
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None


# LLM-generated content at query #19
#--------------------------

```python
def test_imports():
    """Test the imports function for parsing various import statements."""
    from io import StringIO
    
    # Test simple import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    
    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias is None
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from collections import OrderedDict as OD\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "OrderedDict"
    assert result[0].alias == "OD"
    
    # Test multiple imports on one line
    input_stream = StringIO("import os, sys, json\n")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert result[0].module == "os"
    assert result[1].module == "sys"
    assert result[2].module == "json"
    
    # Test from import with multiple attributes
    input_stream = StringIO("from os import path, getcwd, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert all(r.module == "os" for r in result)
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    assert result[2].attribute == "environ"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test multiline import with backslash
    input_stream = StringIO("import os, \\\n    sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test indented import
    input_stream = StringIO("    import json\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    assert result[0].cimport is True
    
    # Test with comments
    input_stream = StringIO("import os  # comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only flag stops at function definition
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    input_stream = StringIO("import os\n")
    file_path = Path("test.py")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 1\nimport os\ny = 2\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test semicolon separated statements
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test redundant alias removal
    input_stream = StringIO("import os as os\n")
    config = Config(remove_redundant_aliases=True)
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None


# LLM-generated content at query #20
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test simple import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].attribute is None
    assert result[1].module == "sys"
    assert result[1].line_number == 2
    
    # Test from import
    input_stream = StringIO("from os import path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].line_number == 1
    
    # Test import with alias
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports in one line
    input_stream = StringIO("from os import path, environ")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "environ"
    
    # Test indented import
    input_stream = StringIO("    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    environ\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "environ"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, \\\n    environ")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport numpy")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc cimport stdlib")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc"
    assert result[0].attribute == "stdlib"
    
    # Test with comments
    input_stream = StringIO("import os  # this is a comment\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test top_only parameter
    input_stream = StringIO("import os\ndef foo():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path parameter
    test_path = Path("test.py")
    input_stream = StringIO("import os")
    result = list(imports(input_stream, file_path=test_path))
    assert len(result) == 1
    assert result[0].file_path == test_path
    
    # Test skip lines with triple quotes
    input_stream = StringIO('"""\nimport os\n"""\nimport sys')
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "sys"
    
    # Test semicolon separated statements
    input_stream = StringIO("import os; import sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test statement method
    imp = Import(1, False, "os", attribute="path", alias="p")
    assert imp.statement() == "from os import path as p"
    
    # Test statement method for straight import
    imp = Import(1, False, "numpy", alias="np")
    assert imp.statement() == "import numpy as np"
    
    # Test __str__ method
    imp = Import(1, False, "os", file_path=Path("test.py"))
    assert "test.py:1" in str(imp)
    assert "import os" in str(imp)


# LLM-generated content at query #21
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test simple import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    
    # Test from import
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports on one line
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test multiple from imports
    input_stream = StringIO("from os import path, getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "getcwd"
    
    # Test indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path,\\\n    getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test with comments
    input_stream = StringIO("import os  # comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skip lines with quotes
    input_stream = StringIO('"""\nimport fake\n"""\nimport os\n')
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only parameter
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    input_stream = StringIO("import os\n")
    file_path = Path("test.py")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path
    
    # Test line numbers
    input_stream = StringIO("import os\n\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].line_number == 1
    assert result[1].line_number == 3
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias is None
    
    # Test redundant from alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("from os import path as path\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].attribute == "path"
    assert result[0].alias is None


# LLM-generated content at query #22
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].line_number == 1
    assert result[0].attribute is None
    assert result[1].module == "sys"
    assert result[1].line_number == 2
    
    # Test from import
    input_stream = StringIO("from os import path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].line_number == 1
    
    # Test import with alias
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports on one line
    input_stream = StringIO("from os import path, getcwd")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test indented import
    input_stream = StringIO("    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import \\\n    path, \\\n    getcwd")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport numpy")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test with comments
    input_stream = StringIO("import os  # comment")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only parameter stops at function definition
    input_stream = StringIO("import os\n\ndef foo():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    file_path = Path("test.py")
    input_stream = StringIO("import os")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path
    
    # Test skipping yield statements
    input_stream = StringIO("yield\nimport os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test multiple statements on one line with semicolon
    input_stream = StringIO("x = 1; import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test statement method
    import_obj = Import(
        line_number=1,
        indented=False,
        module="os",
        attribute="path",
        alias="p",
        cimport=False,
        file_path=None
    )
    assert import_obj.statement() == "from os import path as p"
    
    # Test statement method for straight import
    import_obj = Import(
        line_number=1,
        indented=False,
        module="numpy",
        alias="np",
        cimport=False,
        file_path=None
    )
    assert import_obj.statement() == "import numpy as np"
    
    # Test __str__ method
    import_obj = Import(
        line_number=5,
        indented=True,
        module="os",
        attribute="path",
        cimport=False,
        file_path=Path("test.py")
    )
    assert "test.py:5" in str(import_obj)
    assert "indented" in str(import_obj)
    
    # Test empty input
    input_stream = StringIO("")
    result = list(imports(input_stream))
    assert len(result) == 0
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("from os import path as path")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test redundant alias kept when config is False
    config = Config(remove_redundant_aliases=False)
    input_stream = StringIO("from os import path as path")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias == "path"


# LLM-generated content at query #23
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    assert result[0].indented is False
    
    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias is None
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from collections import defaultdict as dd\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "collections"
    assert result[0].attribute == "defaultdict"
    assert result[0].alias == "dd"
    
    # Test multiple imports
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "getcwd"
    
    # Test indented import
    input_stream = StringIO("    import json\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test with cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test import with comment
    input_stream = StringIO("import os  # operating system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test backslash continuation
    input_stream = StringIO("from os import \\\n    path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    
    # Test multiple statements on one line
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test top_only parameter stops at function definition
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test with file_path
    input_stream = StringIO("import os\n")
    file_path = Path("test.py")
    result = list(imports(input_stream, file_path=file_path))
    assert result[0].file_path == file_path
    
    # Test from import with multiple items
    input_stream = StringIO("from os import path, getcwd, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    assert result[2].attribute == "environ"
    
    # Test skipping non-import lines
    input_stream = StringIO("x = 1\nprint('hello')\nimport os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test line numbers are correct
    input_stream = StringIO("import os\n\nimport sys\n")
    result = list(imports(input_stream))
    assert result[0].line_number == 1
    assert result[1].line_number == 3


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from io import StringIO
from pathlib import Path
from isort.stdstream import imports, Import
from isort.settings import Config


def test_imports():
    """Test the imports function with various import statements."""
    
    # Test simple import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].line_number == 1
    
    # Test from import
    input_stream = StringIO("from pathlib import Path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "pathlib"
    assert result[0].attribute == "Path"
    assert result[0].alias is None
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os.path import join as path_join\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os.path"
    assert result[0].attribute == "join"
    assert result[0].alias == "path_join"
    
    # Test multiple imports on one line
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test from import with multiple attributes
    input_stream = StringIO("from os import path, getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "getcwd"
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "numpy"
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    
    # Test indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test with file_path
    input_stream = StringIO("import sys\n")
    file_path = Path("test.py")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path
    
    # Test top_only stops at function definition
    input_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only stops at class definition
    input_stream = StringIO("import os\n\nclass Foo:\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test lines with comments are ignored
    input_stream = StringIO("# import os\n")
    result = list(imports(input_stream))
    assert len(result) == 0
    
    # Test inline comments don't break parsing
    input_stream = StringIO("import os  # operating system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test empty file
    input_stream = StringIO("")
    result = list(imports(input_stream))
    assert len(result) == 0
    
    # Test multiple statements on one line
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test redundant alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None
    
    # Test redundant from import alias removal
    config = Config(remove_redundant_aliases=True)
    input_stream = StringIO("from os import path as path\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].alias is None


# LLM-generated content at query #25
#--------------------------

```python
def test_imports():
    """Test the imports function with various import statements."""
    from io import StringIO
    
    # Test basic straight import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].cimport is False
    
    # Test from import
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None
    
    # Test import with alias
    input_stream = StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    
    # Test from import with alias
    input_stream = StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    
    # Test multiple imports on one line
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test multiple from imports
    input_stream = StringIO("from os import path, getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "getcwd"
    
    # Test indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented is True
    
    # Test cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport is True
    
    # Test from cimport
    input_stream = StringIO("from libc.stdlib cimport malloc\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "libc.stdlib"
    assert result[0].attribute == "malloc"
    assert result[0].cimport is True
    
    # Test multiline import with parentheses
    input_stream = StringIO("from os import (\n    path,\n    getcwd\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test multiline import with backslash
    input_stream = StringIO("from os import path, \\\n    getcwd\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].attribute == "path"
    assert result[1].attribute == "getcwd"
    
    # Test line numbers (1-based indexing)
    input_stream = StringIO("# comment\nimport os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].line_number == 2
    
    # Test with file_path
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, file_path=Path("test.py")))
    assert len(result) == 1
    assert result[0].file_path == Path("test.py")
    
    # Test top_only stops at function definition
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test top_only stops at class definition
    input_stream = StringIO("import os\nclass Foo:\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skip comments
    input_stream = StringIO("import os  # comment\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    
    # Test skip lines with raise
    input_stream = StringIO("import os\nraise ImportError\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    
    # Test semicolon separated statements
    input_stream = StringIO("import os; import sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"


