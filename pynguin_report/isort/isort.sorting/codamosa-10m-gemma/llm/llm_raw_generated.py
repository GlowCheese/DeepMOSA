####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Mock Config object
    config = MagicMock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = []
    config.classes = []
    config.variables = []
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []

    # Test Case 1: Basic module name
    assert module_key("os", config) == "Bos"

    # Test Case 2: Relative import with underscore (reverse_relative=False)
    config.reverse_relative = False
    assert module_key(".utils", config) == "B.utils"

    # Test Case 3: Relative import with space (reverse_relative=True)
    config.reverse_relative = True
    assert module_key(".utils", config) == "B. utils"

    # Test Case 4: Ignore case
    config.case_sensitive = True
    assert module_key("OS", config, ignore_case=True) == "bos"

    # Test Case 5: Case insensitive config
    config.case_sensitive = False
    assert module_key("OS", config) == "bos"

    # Test Case 6: Sub-imports with order_by_type (Constants - Prefix A)
    config.order_by_type = True
    config.constants = ["MY_CONST"]
    assert module_key("MY_CONST", config, sub_imports=True) == "BA"

    # Test Case 7: Sub-imports with order_by_type (Classes - Prefix B)
    config.classes = ["MyClass"]
    assert module_key("MyClass", config, sub_imports=True) == "BB"

    # Test Case 8: Sub-imports with order_by_type (Variables - Prefix C)
    config.variables = ["my_var"]
    assert module_key("my_var", config, sub_imports=True) == "BC"

    # Test Case 9: Sub-imports with order_by_type (Upper case name - Prefix A)
    assert module_key("UPPER", config, sub_imports=True) == "BA"

    # Test Case 10: Force to top (Prefix A)
    config.force_to_top = ["sys"]
    assert module_key("sys", config) == "Asys"

    # Test Case 11: Length sort enabled
    config.length_sort = True
    assert module_key("abc", config) == "B3:abc"

    # Test Case 12: Length sort via section name
    config.length_sort = False
    config.length_sort_sections = ["imports"]
    assert module_key("abc", config, section_name="imports") == "B3:abc"

    # Test Case 13: Length sort via straight_import
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("abc", config, straight_import=True) == "B3:abc"

    # Test Case 14: Complex relative path with underscore
    config.reverse_relative = False
    assert module_key("..sub.mod", config) == "B.._sub.mod"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Mock Config object
    config = MagicMock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = []
    config.classes = []
    config.variables = []
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []

    # 1. Basic test: Standard module name
    assert module_key("os", config) == "Bos"

    # 2. Test relative imports (dot notation)
    config.reverse_relative = False
    assert module_key(".utils", config) == "B.utils"
    
    config.reverse_relatve = True # Note: typo in source code 'reverse_relatve' vs 'reverse_relative' 
    # However, looking at the provided source code: `sep = " " if config.reverse_relative else "_"`
    # Wait, the source code says: `sep = " " if config.reverse_relative else "_"`
    config.reverse_relative = True
    assert module_key(".utils", config) == "B. utils"

    # 3. Test case sensitivity
    config.case_sensitive = False
    assert module_key("OS", config) == "bos"
    config.case_sensitive = True
    assert module_key("OS", config) == "BOS"

    # 4. Test ignore_case parameter
    assert module_key("Utils", config, ignore_case=True) == "butils"

    # 5. Test force_to_top
    config.force_to_top = ["os"]
    assert module_key("os", config) == "Aos"

    # 6. Test sub_imports and order_by_type (Prefix logic)
    config.order_by_type = True
    config.constants = ["MY_CONST"]
    config.classes = ["MyClass"]
    config.variables = ["my_var"]

    # Prefix A: Constants or Uppercase
    assert module_key("MY_CONST", config, sub_imports=True) == "BA"
    assert module_key("UPPER", config, sub_imports=True) == "BA"
    
    # Prefix B: Classes or Capitalized
    assert module_key("MyClass", config, sub_imports=True) == "BB"
    assert module_key("Capitalized", config, sub_imports=True) == "BB"

    # Prefix C: Variables or lowercase
    assert module_key("my_var", config, sub_imports=True) == "BC"
    assert module_key("lowercase", config, sub_imports=True) == "BC"

    # 7. Test length_sort
    config.length_sort = True
    # Length of 'os' is 2. Prefix B + '2' + 'os'
    assert module_key("os", config) == "B2os"

    # 8. Test length_sort_straight with straight_import
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("os", config, straight_import=True) == "B2os"
    assert module_key("os", config, straight_import=False) == "Bos"

    # 9. Test length_sort_sections
    config.length_sort = False
    config.length_sort_sections = ["my_section"]
    assert module_key("os", config, section_name="my_section") == "B2os"
    assert module_key("os", config, section_name="other") == "Bos"

    # 10. Test complex relative path with underscores (default)
    config.reverse_relative = False
    assert module_key("..utils", config) == "B.._utils"
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_section_key():
    # Setup a base mock config
    def create_config(
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        sort_relative_in_force_sorted_sections=False,
        force_to_top=None,
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=False,
        length_sort=False,
    ):
        config = MagicMock()
        config.reverse_relative = reverse_relative
        config.group_by_package = group_by_package
        config.lexicographical = lexicographical
        config.sort_relative_in_force_sorted_sections = sort_relative_in_force_sorted_sections
        config.force_to_top = force_to_top or []
        config.honor_case_in_force_sorted_sections = honor_case_in_force_sorted_sections
        config.case_sensitive = case_sensitive
        config.order_by_type = order_by_type
        config.length_sort = length_sort
        return config

    # Test Case 1: Basic import, no special config
    config = create_config()
    assert section_key("import os", config) == "Bimport os"

    # Test Case 2: Force to top (Section A)
    config = create_config(force_to_top=["os"])
    assert section_key("import os", config) == "Aimport os"

    # Test Case 3: Lexicographical mode (replaces ' import ' with '.')
    config = create_config(lexicographical=True)
    assert section_key("from os import path", config) == "Bos.path"

    # Test Case 4: Group by package (splits at ' import ')
    config = create_config(group_by_package=True)
    assert section_key("from os import path", config) == "Bfrom os"

    # Test Case 5: Length sort enabled
    config = create_config(length_sort=True)
    # "import os" length is 9
    assert section_key("import os", config) == "B9import os"

    # Test Case 6: Relative imports with reverse_relative=True
    # If reverse_relative is True, 'from ..module' becomes 'from .. module'
    config = create_config(reverse_relative=True, sort_relative_in_force_sorted_sections=True)
    assert section_key("from ..module", config) == "Bfrom .. module"

    # Test Case 7: Case sensitivity and order_by_type logic (Complex branch)
    # Testing the honor_case_in_force_sorted_sections logic
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True
    )
    # Should lowercase module_name but NOT names (because order_by_type is True)
    # Wait, the code says: if not config.order_by_type: names = names.lower()
    # Since order_by_type is True, names remains case sensitive.
    assert section_key("from OS import Path", config) == "Bfrom os import Path"

    # Test Case 8: Case sensitivity and order_by_type logic (Names also lower)
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=False
    )
    assert section_key("from OS import Path", config) == "Bfrom os import path"

    # Test Case 9: Simple case insensitivity (no honor_case flag)
    config = create_config(order_by_type=False, case_sensitive=False)
    assert section_key("import OS", config) == "Bimport os"

    # Test Case 10: strip/prefix logic for 'from'
    config = create_config()
    assert section_key("from os import path", config) == "Bos import path"
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_section_key():
    # Mock Config object
    config = MagicMock()
    
    # Base configuration defaults
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.sort_relative_in_force_sorted_sections = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False

    # Test Case 1: Standard simple import
    # Expected: 'B' + 'import_name' -> 'Bimport_name'
    assert section_key("import os", config) == "Bos"

    # Test Case 2: Simple from import
    # Expected: 'B' + 'module_name' -> 'Bmodule_name'
    assert section_key("from os import path", config) == "Bos import path"

    # Test Case 3: Lexicographical mode
    # Should replace ' import ' with '.' and remove 'from ' or 'import '
    config.lexicographical = True
    assert section_key("from os import path", config) == "Bos.path"
    config.lexicographical = False

    # Test Case 4: Force to top (Section A)
    config.force_to_top = ["os"]
    assert section_key("import os", config) == "Aos"
    config.force_to_top = []

    # Test Case 5: Group by package
    config.group_by_package = True
    assert section_key("from os.path import exists", config) == "Bos.path"
    config.group_by_package = False

    # Test Case 6: Relative imports with reverse_relative=False
    # Should transform 'from ..module' to 'from .._module' (if sort_relative is True)
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    # Re-evaluating regex logic: re.sub(r"^(\.+)", rf"\1{sep}", line)
    # 'from ..module' -> 'from .._module'
    assert section_key("from ..module", config) == "B.._module"
    
    # Test Case 7: Relative imports with reverse_relative=True
    config.reverse_relative = True
    assert section_key("from ..module", config) == "B.. module"
    
    # Test Case 8: Case sensitivity and order_by_type (Complex Logic)
    # Testing the 'honor_case_in_force_sorted_sections' branch
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    # Line: "from OS import Path" -> module becomes "os", names remains "Path"
    assert section_key("from OS import Path", config) == "Bos import Path"

    # Test Case 9: Length sort enabled
    config.length_sort = True
    # line "os" (len 2) -> "B2os"
    assert section_key("import os", config) == "B2os"

    # Test Case 10: Case insensitive simple import
    config.case_sensitive = False
    config.order_by_type = False
    assert section_key("import OS", config) == "Bos"

    # Test Case 11: Checking section_key with 'from .' logic for reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    # line starts with 'from .' and reverse_relative is True
    # match.groups() for "from ..module" -> ("..", "module")
    # line becomes "from .. module"
    assert section_key("from ..module", config) == "Bfrom .. module"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Setup a mock Config object
    config = MagicMock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = []
    config.classes = []
    config.variables = []
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []

    # 1. Test standard absolute import
    assert module_key("os", config) == "Bos"

    # 2. Test relative import with underscore (default)
    assert module_key(".utils", config) == "B.utils"

    # 3. Test relative import with space (reverse_relative = True)
    config.reverse_relative = True
    assert module_key(".utils", config) == "B. utils"
    config.reverse_relative = False

    # 4. Test ignore_case = True
    assert module_key("OS", config, ignore_case=True) == "bos"

    # 5. Test case_sensitive = False
    config.case_sensitive = False
    assert module_key("OS", config) == "bos"
    config.case_sensitive = True

    # 6. Test sub_imports and order_by_type (Constants -> A)
    config.order_by_type = True
    config.constants = ["MY_CONST"]
    assert module_key("MY_CONST", config, sub_imports=True) == "BA"
    
    # 7. Test sub_imports and order_by_type (Classes -> B)
    config.classes = ["MyClass"]
    assert module_key("MyClass", config, sub_imports=True) == "BB"

    # 8. Test sub_imports and order_by_type (Variables -> C)
    config.variables = ["my_var"]
    assert module_key("my_var", config, sub_imports=True) == "BC"

    # 9. Test sub_imports and order_by_type (Uppercase string as A)
    assert module_key("UPPER", config, sub_imports=True) == "BA"

    # 10. Test force_to_top
    config.force_to_top = ["os"]
    assert module_key("os", config) == "Aos"

    # 11. Test length_sort
    config.length_sort = True
    assert module_key("abc", config) == "B3:abc"

    # 12. Test length_sort_straight with straight_import=True
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("abc", config, straight_import=True) == "B3:abc"
    assert module_key("abc", config, straight_import=False) == "Babc"

    # 13. Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["my_section"]
    assert module_key("abc", config, section_name="my_section") == "B3:abc"
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Mock Config object
    config = MagicMock()
    
    # Default settings
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.constants = []
    config.classes = []
    config.variables = []
    config.force_to_top = []

    # 1. Test basic module name
    assert module_key("os", config) == "Bos"

    # 2. Test relative imports with underscore separator (default)
    assert module_key(".utils", config) == "B.utils"
    
    # 3. Test relative imports with space separator (reverse_relative=True)
    config.reverse_relative = True
    assert module_key(".utils", config) == "B. utils"
    config.reverse_relative = False

    # 4. Test ignore_case
    assert module_key("OS", config, ignore_case=True) == "bos"
    assert module_key("OS", config, ignore_case=False) == "BOS"

    # 5. Test case_sensitive=False
    config.case_sensitive = False
    assert module_key("OS", config) == "bos"
    config.case_sensitive = True

    # 6. Test sub_imports and order_by_type with types
    config.order_by_type = True
    config.constants = ["MY_CONST"]
    config.classes = ["MyClass"]
    config.variables = ["my_var"]

    # Type A: Constants
    assert module_key("MY_CONST", config, sub_imports=True) == "BA MY_CONST"
    # Type A: Uppercase names (Issue #376)
    assert module_key("UPPER", config, sub_imports=True) == "BA UPPER"
    # Type B: Classes
    assert module_key("MyClass", config, sub_imports=True) == "BB MyClass"
    # Type B: Title case/Starts with upper
    assert module_key("SomeThing", config, sub_imports=True) == "BB SomeThing"
    # Type C: Variables/Lower
    assert module_key("my_var", config, sub_imports=True) == "BC my_var"
    # Type C: Default lowercase
    assert module_key("simple", config, sub_imports=True) == "BC simple"

    config.order_by_type = False

    # 7. Test force_to_top
    config.force_to_top = ["sys"]
    assert module_key("sys", config) == "Asys"
    config.force_to_top = []

    # 8. Test length_sort
    config.length_sort = True
    assert module_key("abc", config) == "B3:abc"
    config.length_sort = False

    # 9. Test length_sort_straight and straight_import
    config.length_sort_straight = True
    assert module_key("abc", config, straight_import=True) == "B3:abc"
    assert module_key("abc", config, straight_import=False) == "Babc"
    config.length_sort_straight = False

    # 10. Test length_sort_sections
    config.length_sort_sections = ["my_section"]
    assert module_key("abc", config, section_name="my_section") == "B3:abc"
    config.length_sort_sections = []

    # 11. Test complex relative path with underscore
    config.reverse_relative = True
    assert module_key("..utils", config) == "B.  utils"
    config.reverse_relative = False
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Setup a mock Config object
    config = MagicMock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.constants = []
    config.classes = []
    config.variables = []
    config.force_to_top = []

    # Test 1: Basic absolute import
    assert module_key("os", config) == "Bos"

    # Test 2: Relative import with underscore (reverse_relative=False)
    config.reverse_relative = False
    assert module_key(".utils", config) == "B.utils"

    # Test 3: Relative import with space (reverse_relative=True)
    config.reverse_relative = True
    assert module_key(".utils", config) == "B. utils"

    # Test 4: Ignore case
    config.case_sensitive = True
    assert module_key("OS", config, ignore_case=True) == "bos"
    
    # Test 5: Case sensitivity
    config.case_sensitive = False
    assert module_key("OS", config) == "os"

    # Test 6: Force to top
    config.force_to_top = ["os"]
    assert module_key("os", config) == "Aos"

    # Test 7: Order by type - Constant (A)
    config.order_by_type = True
    config.constants = ["MY_CONST"]
    assert module_key("MY_CONST", config, sub_imports=True) == "BA"

    # Test 8: Order by type - Class (B)
    config.classes = ["MyClass"]
    assert module_key("MyClass", config, sub_imports=True) == "BB"

    # Test 9: Order by type - Variable (C)
    config.variables = ["my_var"]
    assert module_key("my_var", config, sub_imports=True) == "BC"

    # Test 10: Order by type - Uppercase string (A)
    assert module_key("UPPER", config, sub_imports=True) == "BA"

    # Test 11: Length sort enabled
    config.length_sort = True
    assert module_key("abc", config) == "B6:abc"

    # Test 12: Length sort via section name
    config.length_sort = False
    config.length_sort_sections = ["imports"]
    assert module_key("abc", config, section_name="imports") == "B6:abc"

    # Test 13: Length sort via straight import flag
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("abc", config, straight_import=True) == "B6:abc"

    # Test 14: Complex relative path with underscore
    config.reverse_relative = False
    assert module_key("..sub.mod", config) == "B..sub.mod"
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Setup a base config object
    config = MagicMock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.constants = []
    config.classes = []
    config.variables = []
    config.force_to_top = []

    # 1. Test basic module name
    assert module_key("os", config) == "Bos"

    # 2. Test relative imports with underscore (default)
    assert module_key(".utils", config) == "B.utils"

    # 3. Test relative imports with space (reverse_relative=True)
    config.reverse_relative = True
    assert module_key(".utils", config) == "B. utils"
    config.reverse_relative = False

    # 4. Test case insensitivity (ignore_case=True)
    assert module_key("OS", config, ignore_case=True) == "Bos"

    # 5. Test case sensitivity (config.case_sensitive=False)
    config.case_sensitive = False
    assert module_key("OS", config) == "bos"
    config.case_sensitive = True

    # 6. Test force_to_top prefix 'A'
    config.force_to_top = ["sys"]
    assert module_key("sys", config) == "Asys"
    config.force_to_top = []

    # 7. Test sub_imports and order_by_type (Prefix A: Constants)
    config.order_by_type = True
    config.constants = ["MY_CONST"]
    assert module_key("MY_CONST", config, sub_imports=True) == "BA"
    
    # 8. Test sub_imports and order_by_type (Prefix B: Classes)
    config.classes = ["MyClass"]
    assert module_key("MyClass", config, sub_imports=True) == "BB"
    
    # 9. Test sub_imports and order_by_type (Prefix C: Variables/Others)
    assert module_key("my_var", config, sub_imports=True) == "BC"

    # 10. Test sub_imports and order_by_type (Prefix A: Uppercase module name)
    assert module_key("UPPER_CASE", config, sub_imports=True) == "BA"

    # 11. Test length_sort
    config.length_sort = True
    assert module_key("abc", config) == "B3:abc"
    config.length_sort = False

    # 12. Test length_sort_straight with straight_import=True
    config.length_sort_straight = True
    assert module_key("abc", config, straight_import=True) == "B3:abc"
    config.length_sort_straight = False

    # 13. Test length_sort_sections
    config.length_sort_sections = ["my_section"]
    assert module_key("abc", config, section_name="my_section") == "B3:abc"
    config.length_sort_sections = []

    # 14. Test complex relative path with underscore
    config.reverse_relative = False
    assert module_key("..core.utils", config) == "B..core_utils"

    # 15. Test complex relative path with space
    config.reverse_relative = True
    assert module_key("..core.utils", config) == "B.  core utils"
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Setup a mock config object
    config = MagicMock()
    
    # Default config state
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = []
    config.classes = []
    config.variables = []
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []

    # Test 1: Basic module name
    assert module_key("os", config) == "Bos"

    # Test 2: Relative imports with underscore separator (default)
    assert module_key(".utils", config) == "B.utils"
    
    # Test 3: Relative imports with space separator (reverse_relative=True)
    config.reverse_relative = True
    assert module_key(".utils", config) == "B. utils"
    config.reverse_relative = False

    # Test 4: Case insensitivity (ignore_case=True)
    assert module_key("OS", config, ignore_case=True) == "Bos"

    # Test 5: Case sensitivity (config.case_sensitive=False)
    config.case_sensitive = False
    assert module_key("OS", config) == "bos"
    config.case_sensitive = True

    # Test 6: Order by type - Constants (A)
    config.order_by_type = True
    config.constants = ["MY_CONST"]
    assert module_key("MY_CONST", config, sub_imports=True) == "BA"
    
    # Test 7: Order by type - Classes (B)
    config.classes = ["MyClass"]
    assert module_key("MyClass", config, sub_imports=True) == "BB"
    
    # Test 8: Order by type - Variables (C)
    config.variables = ["my_var"]
    assert module_key("my_var", config, sub_imports=True) == "BC"

    # Test 9: Order by type - Uppercase check (Issue #376)
    assert module_key("UPPER", config, sub_imports=True) == "BA"

    # Test 10: Force to top
    config.force_to_top = ["os"]
    assert module_key("os", config) == "Aos"

    # Test 11: Length sort
    config.length_sort = True
    assert module_key("abc", config) == "B3:abc"
    config.length_sort = False

    # Test 12: Length sort via section name
    config.length_sort_sections = ["Main"]
    assert module_key("abc", config, section_name="Main") == "B3:abc"
    config.length_sort_sections = []

    # Test 13: Straight import length sort
    config.length_sort_straight = True
    assert module_key("abc", config, straight_import=True) == "B3:abc"
    config.length_sort_straight = False

    # Test 14: Complex relative path with underscore
    config.reverse_relative = True
    assert module_key("..utils", config) == "B. .utils"
    config.reverse_relative = False
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockConfig:
    def __init__(self, **kwargs):
        self.reverse_relative = kwargs.get("reverse_relative", False)
        self.order_by_type = kwargs.get("order_by_type", False)
        self.constants = kwargs.get("constants", [])
        self.classes = kwargs.append([]) if "classes" not in kwargs else kwargs["classes"]
        self.variables = kwargs.get("variables", [])
        self.case_sensitive = kwargs.get("case_sensitive", True)
        self.length_sort = kwargs.get("length_sort", False)
        self.length_sort_straight = kwargs.get("length_sort_straight", False)
        self.length_sort_sections = kwargs.get("length_sort_sections", [])
        self.force_to_top = kwargs.get("force_to_top", [])

def test_module_key():
    # Base config setup
    base_params = {
        "reverse_relative": False,
        "order_by_type": False,
        "constants": [],
        "classes": [],
        "variables": [],
        "case_sensitive": True,
        "length_sort": False,
        "length_sort_straight": False,
        "length_sort_sections": [],
        "force_to_top": []
    }

    # Test 1: Standard module name
    config = MockConfig(**base_params)
    assert module_key("os", config) == "Bos"

    # Test 2: Relative import with underscore separator
    config = MockConfig(**base_params, reverse_relative=False)
    assert module_key(".utils", config) == "B.utils"
    
    config = Mock0Config(**base_params, reverse_relative=True)
    assert module_key(".utils", config) == "B. utils" # match.groups() returns ('.', 'utils') -> ' '.join -> '. utils'
    # Note: The implementation uses sep.join(match.groups()) where groups are ('.', 'utils')
    # If reverse_relative is True, sep is " ". Result: ". utils"

    # Test 3: Case Insensitivity
    config = MockConfig(**base_params, case_sensitive=False)
    assert module_key("OS", config) == "bos"

    # Test 4: Ignore Case flag
    config = MockConfig(**base_params)
    assert module_key("OS", config, ignore_case=True) == "BOS"

    # Test 5: Order by type - Constants (Prefix A)
    config = MockConfig(**base_params, order_by_type=True, constants=["MY_CONST"])
    assert module_key("MY_CONST", config, sub_imports=True) == "BA MY_CONST"

    # Test 6: Order by type - Classes (Prefix B)
    config = MockConfig(**base_params, order_by_type=True, classes=["MyClass"])
    assert module_key("MyClass", config, sub_imports=True) == "BB MyClass"

    # Test 7: Order by type - Variables (Prefix C)
    config = MockConfig(**base_params, order_by_type=True, variables=["my_var"])
    assert module_key("my_var", config, sub_imports=True) == "BC my_var"

    # Test 8: Order by type - Upper case string (Prefix A)
    config = MockConfig(**base_params, order_by_type=True)
    assert module_key("UPPER", config, sub_imports=True) == "BA UPPER"

    # Test 9: Force to top (Prefix A)
    config = MockConfig(**base_params, force_to_top=["os"])
    assert module_key("os", config) == "Aos"

    # Test 10: Length sorting
    config = MockConfig(**base_params, length_sort=True)
    assert module_key("abc", config) == "B3:abc"

    # Test 11: Length sort via section name
    config = MockConfig(**base_params, length_sort_sections=["MySection"])
    assert module_key("abc", config, section_name="MySection") == "B3:abc"

    # Test 12: Straight import length sort
    config = MockConfig(**base_params, length_sort_straight=True)
    assert module_key("abc", config, straight_import=True) == "B3:abc"
    assert module_key("abc", config, straight_import=False) == "Babc"

    # Test 13: Complex relative path with reverse_relative
    config = MockConfig(**base_params, reverse_relative=True)
    # match.groups() for "..module" is ('.', '.module') or ('.', 'module') depending on regex
    # re.match(r"^(\.+)\s*(.*)", "..module") -> groups: ('..', 'module')
    # sep is " " -> " ".join(('..', 'module')) -> ".. module"
    assert module_key("..module", config) == "B.. module"
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Setup a base config object
    class MockConfig:
        def __init__(self):
            self.reverse_relative = False
            self.order_by_type = False
            self.constants = []
            self.classes = []
            self.variables = []
            self.case_sensitive = True
            self.length_sort = False
            self.length_sort_straight = False
            self.length_sort_sections = []
            self.force_to_top = []

    config = MockConfig()

    # 1. Test standard absolute import
    assert module_key("os", config) == "Bos"

    # 2. Test relative import with default (no reverse_relative)
    # match.groups() for "..module" -> ('.', 'module') -> join with '_' -> '._module'
    # Wait, re.match(r"^(\.+)\s*(.*)", "..module") -> group1='..', group2='module'
    # sep = "_" -> ".._module"
    config.reverse_relative = False
    assert module .:module_key("..module", config) == "B.._module"

    # 3. Test relative import with reverse_relative (sep is ' ')
    config.reverse_relative = True
    assert module_key("..module", config) == "B.. module"

    # 4. Test ignore_case
    assert module_key("OS", config, ignore_case=True) == "Bos"

    # 5. Test case_sensitive = False
    config.case_sensitive = False
    assert module_key("OS", config) == "Bos"

    # 6. Test sub_imports and order_by_type (Prefix logic)
    config.order_by_type = True
    config.constants = ["MY_CONST"]
    config.classes = ["MyClass"]
    config.variables = ["my_var"]

    # Prefix A: In constants or Upper case
    assert module_key("MY_CONST", config, sub_imports=True) == "BA MY_CONST"
    assert module_key("UPPER", config, sub_imports=True) == "BA UPPER"
    
    # Prefix B: In classes or Starts with Upper
    assert module_key("MyClass", config, sub_imports=True) == "BB MyClass"
    assert module_key("SomeClass", config, sub_imports=True) == "BB SomeClass"

    # Prefix C: Variables or lowercase
    assert module_key("my_var", config, sub_imports=True) == "BC my_var"
    assert module_key("lowercase", config, sub_imports=True) == "BC lowercase"

    # 7. Test force_to_top (Prefix A)
    config.force_to_top = ["os"]
    assert module_key("os", config) == "Aos"

    # 8. Test length_sort
    config.length_sort = True
    # len("os") is 2 -> "2:os"
    assert module_key("os", config) == "B2:os"

    # 9. Test length_sort_straight and straight_import
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("os", config, straight_import=True) == "B2:os"
    assert module_key("os", config, straight_import=False) == "Bos"

    # 10. Test length_sort_sections
    config.length_sort = False
    config.length_sort_sections = ["my_section"]
    assert module_key("os", config, section_name="my_section") == "B2:os"

    # 11. Test complex combination
    # module_name="..mod", config.reverse_relative=True (sep=' '), 
    # sub_imports=True, order_by_type=True, module_name starts with Upper (Prefix B)
    # force_to_top is empty.
    config.reverse_relative = True
    config.order_by_top = False # Resetting from previous
    config.force_to_top = []
    config.order_by_type = True
    # We need to ensure the string processed for prefix check is the same as module_name
    # "..mod" -> ".. mod" (because reverse_relative is True)
    # Since ".. mod" does not start with Upper and not in lists, prefix is C
    assert module_key("..mod", config, sub_imports=True) == "BC.. mod"
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockConfig:
    def __init__(self, **kwargs):
        self.reverse_relative = kwargs.get("reverse_relative", False)
        self.sort_relative_in_force_sorted_sections = kwargs.get("sort_relative_in_force_sorted_sections", False)
        self.group_by_package = kwargs.get("group_by_package", False)
        self.lexicographical = kwargs.get("lexicographical", False)
        self.case_sensitive = kwargs.get("case_sensitive", True)
        self.order_by_type = kwargs.get("order_by_type", False)
        self.force_to_top = kwargs.get("force_to_top", [])
        self.honor_case_in_force_sorted_sections = kwargs.get("honor_case_in_force_sorted_sections", False)
        self.length_sort = kwargs.get("length_sort", False)

def test_section_key():
    # Test 1: Basic functionality, simple import
    config = MockConfig()
    assert section_key("import os", config) == "Bimport os"

    # Test 2: Force to top (Section A)
    config = MockConfig(force_to_top=["os"])
    assert section_key("import os", config).startswith("A")

    # Test 3: Lexicographical mode (removes from/import and replaces mid-line import with dot)
    config = MockConfig(lexicographical=True)
    assert section_key("from os import path", config) == "Bos.path"

    # Test 4: Case sensitivity (Not case sensitive)
    config = MockImportConfig(case_sensitive=False)
    assert section_key("import OS", config) == "Bimport os"

    # Test 5: Group by package (strips names after import)
    config = MockConfig(group_by_package=True)
    assert section_key("from os import path", config) == "Bfrom os"

    # Test 6: Length sort enabled
    config = MockConfig(length_sort=True)
    # "import os" length is 9
    assert section_key("import os", config) == "B9import os"

    # Test 7: Reverse relative with dot notation
    config = MockConfig(reverse_relative=True, sort_relative_in_force_sorted_sections=True)
    # "from . import module" -> "from .  import module" -> "from . import module" 
    # becomes "from . import module" (logic in code replaces dots with spaces)
    # The regex ^(\.+)\s*(.*) matches dots and replaces them.
    assert section_key("from . import module", config) == "Bfrom . import module"

    # Test 8: honor_case_in_force_sorted_sections with specific splits
    config = MockConfig(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=True,
        order_by_type=False
    )
    # Should lowercase the names part but keep module name
    assert section_key("from MyModule import MyClass", config) == "Bfrom MyModule import myclass"

    # Test 9: No order_by_type, lowercase everything
    config = MockConfig(order_by_type=False)
    assert section_key("import OS", config) == "Bimport os"

    # Test 10: Complex relative pathing with sort_relative_in_force_sorted_sections
    config = MockConfig(
        sort_relative_in_force_sorted_sections=True,
        reverse_relative=False
    )
    # "from .. import module" -> "from .._ import module"
    assert section_key("from .. import module", config) == "Bfrom .._ import module"

class MockImportConfig(MockConfig):
    pass
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_section_key():
    # Mock Config object
    config = MagicMock()
    
    # Default settings for most tests
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.sort_relative_in_force_sorted_sections = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False

    # Test 1: Simple import
    assert section_key("import os", config) == "Bos"

    # Test 2: From import
    assert section_key("from os import path", config) == "Bos import path"

    # Test 3: Lexicographical mode (replaces ' import ' with '.')
    config.lexicographical = True
    assert section_key("from os import path", config) == "Bos.os.path"
    config.lexicographical = False

    # Test 4: Force to top
    config.force_to_top = ["os"]
    assert section_key("import os", config) == "Aos"
    config.force_to_top = []

    # Test 5: Group by package
    config.group_by_package = True
    assert section_key("from os import path", config) == "Bos"
    config.group_by_package = False

    # Test 6: Relative imports with reverse_relative=False
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = True
    # Should turn ..module into .._module
    assert section_key("from ..module import func", config) == "B.._module import func"
    
    # Test 7: Relative imports with reverse_relative=True
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    # Should turn ..module into from .. module
    assert section_key("from ..module import func", config) == "Bfrom .. module import func"
    
    # Test 8: Length sort enabled
    config.length_sort = True
    assert section_key("import os", config) == "B2os"

    # Test 9: Case sensitivity and order_by_type interaction (Honor case)
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = False
    # Module name should be lower, names should be lower
    assert section_key("from MyModule import MyClass", config) == "Bmymodule import myclass"

    # Test 10: Case sensitivity and order_by_type interaction (No honor case, but split)
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = False
    config.order_by_type = True
    # Module name becomes lower, but names are untouched because order_by_type is True
    assert section_key("from MyModule import MyClass", config) == "Bmymodule import MyClass"

    # Test 11: Case sensitivity and order_by_type interaction (No honor case, all lower)
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = False
    config.order_by_type = False
    assert section_key("from MyModule import MyClass", config) == "Bmymodule import myclass"
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockConfig:
    def __init__(self, **kwargs):
        self.reverse_relative = kwargs.get("reverse_relative", False)
        self.order_by_type = kwargs.get("order_by_type", False)
        self.constants = kwargs.get("constants", [])
        self.classes = kwargs.append if "classes" in kwargs else kwargs.get("classes", [])
        self.variables = kwargs.get("variables", [])
        self.case_sensitive = kwargs.get("case_sensitive", True)
        self.length_sort = kwargs.get("length_sort", False)
        self.length_sort_straight = kwargs.get("length_sort_straight", False)
        self.length_sort_sections = kwargs.get("length_sort_sections", [])
        self.force_to_top = kwargs.get("force_to_top", [])

def test_module_key():
    # Setup base config
    base_params = {
        "reverse_relative": False,
        "order_by_type": False,
        "constants": [],
        "classes": [],
        "variables": [],
        "case_sensitive": True,
        "length_sort": False,
        "length_sort_straight": False,
        "length_sort_sections": [],
        "force_to_top": []
    }

    # Test 1: Basic module name
    config = MockConfig(**base_params)
    assert module_key("os", config) == "Bos"

    # Test 2: Relative imports with underscore separator (reverse_relative=False)
    config.reverse_relative = False
    assert module_key(".utils", config) == "B._utils"

    # Test 3: Relative imports with space separator (reverse_relative=True)
    config.reverse_relative = True
    assert module_key(".utils", config) == "B. utils"

    # Test 4: Ignore case
    config = MockConfig(**base_params)
    config.case_sensitive = False
    assert module_key("OS", config) == "bos"

    # Test 5: Sub-imports with order_by_type (Constants)
    config = MockConfig(**base_params)
    config.order_type = True # Note: the function uses config.order_by_type
    config.order_by_type = True
    config.constants = ["MY_CONST"]
    assert module_key("MY_CONST", config, sub_imports=True) == "BA MY_CONST"

    # Test 6: Sub-imports with order_by_type (Classes)
    config.classes = ["MyClass"]
    assert module_key("MyClass", config, sub_imports=True) == "BB MyClass"

    # Test 7: Sub-imports with order_by_type (Variables)
    config.variables = ["my_var"]
    assert module_key("my_var", config, sub_imports=True) == "BC my_var"

    # Test 8: Force to top
    config = MockConfig(**base_params)
    config.force_to_top = ["os"]
    assert module_key("os", config) == "Aos"

    # Test 9: Length sort via length_sort_straight and straight_import
    config = MockConfig(**base_params)
    config.length_sort_straight = True
    assert module_key("abc", config, straight_import=True) == "B3:abc"

    # Test 10: Length sort via section_name
    config = MockConfig(**base_params)
    config.length_sort_sections = ["important"]
    assert module_key("abc", config, section_name="important") == "B3:abc"

    # Test 11: Uppercase module name as a proxy for constant (Issue #376 logic)
    config = MockConfig(**base_params)
    config.order_by_type = True
    assert module_key("UPPER", config, sub_imports=True) == "BA UPPER"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_section_key():
    # Mock Config object
    config = MagicMock()
    
    # Base configuration defaults
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.sort_relative_in_force_sorted_sections = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False

    # Test 1: Simple import
    # Expected: 'Bimport_module' (B + stripped 'import ' + module)
    assert section_key("import module", config) == "Bmodule"

    # Test 2: From import
    # Expected: 'Bmodule' (B + stripped 'from ' + module)
    assert section_key("from module import func", config) == "Bmodule import func"

    # Test 3: Force to top
    config.force_to_top = ["os"]
    # Expected: 'Aos'
    assert section_key("import os", config) == "Aos"

    # Test 4: Lexicographical sorting (replaces ' import ' with '.')
    config.lexicographical = True
    # 'from module import func' -> 'module.func'
    assert section_key("from module import func", config) == "Bmodule.func"

    # Test 5: Group by package
    config.group_by_package = True
    # 'from module import func' -> 'from module' -> 'module'
    assert section_key("from module import func", config) == "Bmodule"

    # Test 6: Relative imports with reverse_relative=False
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = True
    # 'from ..module' -> '.._module'
    assert section_key("from ..module import func", config) == "B.._module import func"

    # Test 7: Relative imports with reverse_relative=True
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    # 'from ..module' -> 'from .. module'
    assert section_key("from ..module import func", config) == "Bfrom .. module import func"

    # Test 8: Length sort
    config.length_sort = True
    # 'import a' -> 'Ba' (Length of stripped 'a' is 1)
    # Note: line = re.sub("^import ", "", line) -> 'a'
    assert section_key("import a", config) == "B1a"

    # Test 9: Case sensitivity and order_by_type logic (Honor case)
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    # 'from Module import Func' -> 'module import Func'
    assert section_key("from Module import Func", config) == "Bmodule import Func"

    # Test 10: Case sensitivity and order_by_type logic (No honor case, names lower)
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = False
    config.order_by_type = False
    # 'from Module import Func' -> 'module import func'
    assert section_key("from Module import Func", config) == "Bmodule import func"

    # Test 11: Case sensitivity and order_by_type logic (No honor case, module lower)
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = False
    config.order_by_type = True
    # 'from Module import Func' -> 'module import Func'
    assert section_key("from Module import Func", config) == "Bmodule import Func"
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_section_key():
    # Mocking Config object
    config = MagicMock()
    
    # Default shared settings
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.sort_relative_in_force_sorted_sections = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False

    # 1. Test basic import
    assert section_key("import os", config) == "Bos" # "B" + "os" (re.sub removes 'import ')
    # Note: The logic re.sub("^import ", "", line) leaves "os"
    # Wait, looking at code: re.sub("^import ", "", "import os") -> "os"
    # Result: "B" + "os" -> "Bos"
    
    # 2. Test basic from import
    assert section_key("from math import sqrt", config) == "Bmath import sqrt" 
    # re.sub("^from ", "", "from math import sqrt") -> "math import sqrt"
    # Result: "Bmath import sqrt"

    # 3. Test lexicographical mode
    config.lexicographical = True
    # _import_line_midline_import_re replaces " import " with "."
    # _import_line_intro_re removes "from " or "import "
    assert section_key("from math import sqrt", config) == "Bmath.sqrt"

    # 4. Test force_to_top (Section A)
    config.lexicographical = False
    config.force_to_top = ["os"]
    assert section_key("import os", config) == "Aos"

    # 5. Test relative imports with reverse_relative=False (default)
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = True
    # line starts with "from ." -> regex matches -> line = "from ..module"
    # then re.sub(r"^(\.+)", rf"\1_{sep}", line) -> ".." becomes ".._"
    assert section_key("from ..module import func", config) == "B.._module import func"

    # 6. Test relative imports with reverse_relative=True
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    # sep is " "
    assert section_key("from ..module import func", config) == "B.. module import func"

    # 7. Test group_by_package
    config.group_by_package = True
    # line.split(" import ", 1)[0]
    assert section_key("from os import path", config) == "Bos"

    # 8. Test length_sort
    config.length_sort = True
    # section_key returns f"{section}{len(line)...}"
    # "import os" -> "os" -> len is 2
    assert section_key("import os", config) == "B2os"

    # 9. Test case sensitivity and order_by_type (Honor Case logic)
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    # line = "math import Sqrt"
    # module_name = "math" (lower), names = "Sqrt" (unchanged because order_by_type is True)
    assert section_key("from math import Sqrt", config) == "Bmath import Sqrt"

    config.case_sensitive = False
    config.order_by_type = False
    # module_name = "math" (lower), names = "sqrt" (lower)
    assert section_key("from math import Sqrt", config) == "Bmath import sqrt"

    # 10. Test case sensitivity (Simple)
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = False
    config.case_sensitive = False
    assert section_key("import OS", config) == "Bos"

    # 11. Test reverse_relative logic for 'from .' lines
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    # If line.startswith("from .") and reverse_relative:
    # match group 1 is "..", group 2 is "module"
    # line becomes "from .. module"
    # then regex sub removes "from " -> ".. module"
    assert section_key("from ..module import func", config) == "B.. module import func"
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockConfig:
    def __init__(self, **kwargs):
        self.reverse_relative = kwargs.get("reverse_relative", False)
        self.sort_relative_in_force_sorted_sections = kwargs.get("sort_relative_in_force_sorted_sections", False)
        self.group_by_package = kwargs.get("group_by_package", False)
        self.lexicographical = kwargs.get("lexicographical", False)
        self.case_sensitive = kwargs.get("case_sensitive", True)
        self.order_by_type = kwargs.get("order_by_type", False)
        self.force_to_top = kwargs.get("force_to_top", [])
        self.honor_case_in_force_sorted_sections = kwargs.get("honor_case_in_force_sorted_sections", False)
        self.length_sort = kwargs.get("length_sort", False)

def test_section_key():
    # Test Case 1: Basic functionality - simple import, default config
    config = MockConfig()
    assert section_key("import os", config) == "Bimport os"
    assert section_key("from math import pi", config) == "Bmath import pi"

    # Test Case 2: force_to_top section 'A'
    config = MockConfig(force_to_top=["os"])
    assert section_key("import os", config).startswith("A")

    # Test Case 3: lexicographical sorting (replacing ' import ' with '.')
    config = MockConfig(lexicographical=True)
    assert section_key("from math import pi", config) == "Bmath.pi"
    assert section_key("import os", config) == "Bos"

    # Test Case 4: group_by_package (stripping names after import)
    config = MockConfig(group_by_package=True)
    assert section_key("from os import path", config) == "Bos"

    # Test Case 5: reverse_relative and sort_relative_in_force_sorted_sections
    # If reverse_relative is True and not in force_sorted_sections, dots are joined by space
    config = MockConfig(reverse_relative=True, sort_relative_in_force_sorted_sections=False)
    assert section_key("from ..utils import func", config) == "B.. utils import func"

    # Test Case 6: sort_relative_in_force_sorted_sections with reverse_relative=False
    # Dots are joined by underscore
    config = MockConfig(reverse_relative=False, sort_relative_in_force_sorted_sections=True)
    assert section_key("from ..utils import func", config) == "B.._utils import func"

    # Test Case 7: case_sensitive=False and order_by_type=False (lower-casing everything)
    config = MockConfig(case_sensitive=False, order_by_type=False)
    assert section_key("import OS", config) == "Bios"

    # Test Case 8: honor_case_in_force_sorted_sections with mixed sensitivity
    # Module name follows case_sensitive, names follow order_by_type
    config = MockConfig(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=False
    )
    # line is "from math import PI" -> module "math" (lower), names "pi" (lower)
    assert section_key("from math import PI", config) == "Bmath import pi"

    config = MockConfig(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=True,
        order_by_type=False
    )
    # line is "from math import PI" -> module "math" (as is), names "pi" (lower)
    assert section_key("from math import PI", config) == "Bmath import pi"

    # Test Case 9: length_sort
    config = MockConfig(length_sort=True)
    # "import os" -> len 9
    assert section_key("import os", config) == "B9import os"

    # Test Case 10: strip 'from ' and 'import ' prefixes
    config = MockConfig()
    assert section_key("from sys import argv", config) == "Bsys import argv"
    assert section_key("import sys", config) == "Bsys"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Mock Config object
    config = MagicMock()
    
    # Default settings for basic tests
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = []
    config.classes = []
    config.variables = []
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []

    # Test 1: Simple module name
    assert module_key("os", config) == "Bos"

    # Test 2: Relative import with underscore (reverse_relative=False)
    assert module_key(".utils", config) == "B._utils"

    # Test 3: Relative import with space (reverse_relative=True)
    config.reverse_relative = True
    assert module_key(".utils", config) == "B. utils"
    config.reverse_relative = False

    # Test 4: Ignore case
    assert module_key("OS", config, ignore_case=True) == "bos"
    assert module_key("OS", config, ignore_case=False) == "BOS"

    # Test 5: Case sensitivity
    config.case_sensitive = False
    assert module_key("OS", config) == "bos"
    config.case_sensitive = True

    # Test 6: Sub-imports and order_by_type (Class detection)
    config.order_by_type = True
    config.classes = ["MyClass"]
    assert module_key("MyClass", config, sub_imports=True) == "BBMyClass"

    # Test 7: Sub-imports and order_by_type (Constant detection)
    config.constants = ["MY_CONST"]
    assert module_key("MY_CONST", config, sub_imports=True) == "BA MY_CONST"

    # Test 8: Sub-imports and order_by_type (Variable detection)
    config.variables = ["my_var"]
    assert module_key("my_var", config, sub_imports=True) == "BCmy_var"

    # Test 9: Sub-imports and order_by_type (Uppercase/PascalCase detection)
    config.constants = []
    config.classes = []
    config.variables = []
    assert module_key("SomeModule", config, sub_imports=True) == "BBSomeModule"

    # Test 10: Length sorting
    config.length_sort = True
    assert module_key("abc", config) == "B3:abc"

    # Test 11: Length sorting via section name
    config.length_sort = False
    config.length_sort_sections = ["Main"]
    assert module_key("abc", config, section_name="Main") == "B3:abc"

    # Test 12: Force to top
    config.force_to_top = ["os"]
    assert module_key("os", config) == "Aos"

    # Test 13: Straight import length sort
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("os", config, straight_import=True) == "B2:os"
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_section_key():
    # Setup a base mock config
    def create_config(
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        sort_relative_in_force_sorted_sections=False,
        force_to_top=None,
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=False,
        length_sort=False,
    ):
        config = MagicMock()
        config.reverse_relative = reverse_relative
        config.group_by_package = group_by_package
        config.lexicographical = lexicographical
        config.sort_relative_in_force_sorted_sections = sort_relative_in_force_sorted_sections
        config.force_to_top = force_to_top or []
        config.honor_case_in_force_sorted_sections = honor_case_in_force_sorted_sections
        config.case_sensitive = case_sensitive
        config.order_by_type = order_by_type
        config.length_sort = length_sort
        return config

    # Test 1: Basic import, section B, no special config
    config1 = create_config()
    assert section_key("import os", config1) == "Bimport os"

    # Test 2: Force to top (Section A)
    config2 = create_config(force_to_top=["os"])
    assert section_key("import os", config2) == "Aimport os"

    # Test 3: Lexicographical mode (replaces ' import ' with '.')
    config3 = create_config(lexicographical=True)
    assert section_key("from os import path", config3) == "Bfrom os.path"

    # Test 4: Group by package (strips everything after ' import ')
    config4 = create_config(group_by_package=True)
    assert section_key("from os import path", config4) == "Bfrom os"

    # Test 5: Length sort enabled
    config5 = create_config(length_sort=True)
    # "import os" length is 9
    assert section_key("import os", config5) == "B9import os"

    # Test 6: Relative imports with reverse_relative=True (space separator)
    # and sort_relative_in_force_sorted_sections=True
    config6 = create_import_config_relative(reverse_relative=True, sort_relative_in_force_sorted_sections=True)
    assert section_key("from ..module import name", config6) == "Bfrom .. module import name"

    # Test 7: Relative imports with reverse_relative=False (underscore separator)
    config7 = create_import_config_relative(reverse_relative=False, sort_relative_in_force_sorted_sections=True)
    assert section_key("from ..module import name", config7) == "Bfrom .._module import name"

    # Test 8: Case sensitivity and order_by_type logic (honor_case_in_force_sorted_sections)
    # If honor_case is True, and case_sensitive != order_by_type, it splits and processes parts
    config8 = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True
    )
    # Module name becomes lower, names stay as is (because order_by_type is True)
    assert section_key("from OS import Path", config8) == "Bfrom os import Path"

    # Test 9: Case sensitivity and order_by_type logic (No split/partial processing)
    config9 = create_config(
        honor_case_in_force_sorted_sections=False,
        order_by_type=False
    )
    # If order_by_type is False, the whole line is lowered
    assert section_key("import OS", config9) == "Bimport os"

    # Test 10: Case sensitivity and order_by_type logic (Partial processing - names lowered)
    config10 = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=True,
        order_by_type=False
    )
    # Module stays upper, names become lower
    assert section_key("from OS import Path", config10) == "Bfrom OS import path"

def create_import_config_relative(reverse_relative=False, sort_relative_in_force_sorted_sections=False):
    config = MagicMock()
    config.reverse_relative = reverse_relative
    config.sort_relative_in_force_sorted_sections = sort_relative_in_force_sorted_sections
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    return config
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Setup a mock Config object
    config = MagicMock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = []
    config.classes = []
    config.variables = []
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []

    # 1. Test basic module name
    assert module_key("os", config) == "Bos"

    # 2. Test relative imports (no reverse)
    config.reverse_relative = False
    assert module_key(".utils", config) == "B.utils"

    # 3. Test relative imports (with reverse)
    config.reverse_relative = True
    assert module_key(".utils", config) == "B. utils"

    # 4. Test case insensitivity
    config.case_sensitive = False
    assert module_key("OS", config) == "bos"

    # 5. Test ignore_case parameter
    assert module_key("OS", config, ignore_case=True) == "bos"

    # 6. Test force_to_top
    config.force_to_top = ["sys"]
    assert module_key("sys", config) == "Asys"

    # 7. Test sub_imports and order_by_type logic
    config.order_by_type = True
    config.constants = ["MY_CONST"]
    config.classes = ["MyClass"]
    config.variables = ["my_var"]

    # Constant prefix A
    assert module_key("MY_CONST", config, sub_imports=True) == "BA"
    # Class prefix B
    assert module_key("MyClass", config, sub_imports=True) == "BB"
    # Variable/Other prefix C
    assert module_key("my_var", config, sub_imports=True) == "BC"
    # Uppercase check (issue #376)
    assert module_key("UPPER", config, sub_imports=True) == "BA"

    # 8. Test length_sort
    config.length_sort = True
    # length of "os" is 2. Prefix B + length 2 + name os
    assert module_key("os", config) == "B2os"

    # 9. Test length_sort_straight with straight_import=True
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("os", config, straight_import=True) == "B2os"
    assert module_key("os", config, straight_import=False) == "Bos"

    # 10. Test length_sort_sections
    config.length_sort = False
    config.length_sort_sections = ["test_section"]
    assert module_key("os", config, section_name="test_section") == "B2os"
    assert module_key("os", config, section_name="other") == "Bos"

    # 11. Test complex regex match for relative path dots
    config.reverse_relative = False
    assert module_key("..sub.module", config) == "B..sub.module"
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Setup a mock config object
    config = MagicMock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.constants = []
    config.classes = []
    config.variables = []
    config.force_to_top = []

    # Test 1: Basic module name
    # Expected: Bmodule_name (B because not in force_to_top, no prefix because order_by_type is False)
    assert module_key("module_name", config) == "Bmodule_name"

    # Test 2: Relative imports (dots)
    # Expected: B.._module_name (since reverse_relative is False, uses '_')
    assert module_key("..module_name", config) == "B.._module_name"
    
    config.reverse_relative = True
    # Expected: B.. module_name (since reverse_relative is True, uses ' ')
    assert module_key("..module_name", config) == "B.. module_name"
    config.reverse_relative = False

    # Test 3: Case sensitivity and ignore_case
    assert module_key("ModuleName", config) == "BModuleName"
    assert module_key("ModuleName", config, ignore_case=True) == "Bmodulename"

    # Test 4: force_to_top
    config.force_to_top = ["important_module"]
    assert module_key("important_module", config) == "Aimportant_module"

    # Test 5: sub_imports and order_by_type (Prefix logic)
    config.order_by_type = True
    config.constants = ["CONST"]
    config.classes = ["MyClass"]
    config.variables = ["my_var"]

    # Prefix A: In constants
    assert module_key("CONST", config, sub_imports=True) == "BA"
    # Prefix A: Upper case name > 1 char
    assert module_key("UPPER", config, sub_imports=True) == "BA"
    # Prefix B: In classes
    assert module_key("MyClass", config, sub_imports=True) == "BB"
    # Prefix B: Starts with Upper case
    assert module_key("SomeClass", config, sub_imports=True) == "BB"
    # Prefix C: In variables
    assert module_key("my_var", config, sub_imports=True) == "BC"
    # Prefix C: Lower case, not in lists
    assert module_key("simple", config, sub_imports=True) == "BC"

    # Test 6: Length sorting
    config.length_sort = True
    # Expected: B[length]:name
    assert module_key("abc", config) == "B3:abc"

    # Test 7: Length sort via section name
    config.length_sort = False
    config.length_sort_sections = ["special"]
    assert module_key("abc", config, section_name="special") == "B3:abc"

    # Test 8: Length sort via straight_import and config.length_sort_straight
    config.length_sort_straight = True
    # Expected: B[length]:name
    assert module_key("abc", config, straight_import=True) == "B3:abc"

    # Test 9: Case sensitivity impact on final string
    config.case_sensitive = False
    assert module_key("ModuleName", config) == "Bmodulename"
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_module_key():
    # Mock Config object
    config = MagicMock()
    
    # Default settings for basic tests
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = []
    config.classes = []
    config.variables = []
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []

    # 1. Test basic module name
    assert module_key("os", config) == "Bos"

    # 2. Test relative imports (no reverse_relative)
    assert module_key(".utils", config) == "B.utils"
    
    # 3. Test relative imports (with reverse_relative)
    config.reverse_relative = True
    assert module_key(".utils", config) == "B. utils"
    config.reverse_relative = False

    # 4. Test ignore_case
    assert module_key("OS", config, ignore_case=True) == "bos"
    assert module_key("OS", config, ignore_case=False) == "BOS"

    # 5. Test case_sensitive
    config.case_sensitive = False
    assert module_key("MyModule", config) == "bmymodule"
    config.case_sensitive = True

    # 6. Test sub_imports and order_by_type (Type Prefixing)
    config.order_by_type = True
    config.constants = ["MY_CONST"]
    config.classes = ["MyClass"]
    config.variables = ["my_var"]

    # Prefix A: Constants or Upper Case
    assert module_key("MY_CONST", config, sub_imports=True) == "BA MY_CONST"
    assert module_key("UPPER", config, sub_imports=True) == "BA UPPER"
    
    # Prefix B: Classes (starts with Upper)
    assert module_key("MyClass", config, sub_imports=True) == "BB MyClass"
    
    # Prefix C: Variables or Lower Case
    assert module_key("my_var", config, sub_imports=True) == "BC my_var"
    assert module_key("lowercase", config, sub_imports=True) == "BC lowercase"

    # 7. Test force_to_top (Prefix A)
    config.force_to_top = ["os"]
    assert module_key("os", config) == "Aos"

    # 8. Test length_sort
    config.length_sort = True
    # module_key returns "length:name" if length_sort is True
    assert module_key("abc", config) == "B2:abc"

    # 9. Test length_sort_straight with straight_import=True
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("abc", config, straight_import=True) == "B3:abc"
    assert module_key("abc", config, straight_import=False) == "Babc"

    # 10. Test length_sort_sections
    config.length_sort = False
    config.length_sort_sections = ["['test_section']"]
    assert module_key("abc", config, section_name="test_section") == "B3:abc"

    # 11. Test complex relative path with regex match
    # match = re.match(r"^(\.+)\s*(.*)", module_name)
    # if match: sep = " " if config.reverse_relative else "_"
    config.reverse_relative = False
    assert module_key("..utils", config) == "B.._utils"
    
    config.reverse_relative = True
    assert module_key("..utils", config) == "B.. utils"
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockConfig:
    def __init__(self, **kwargs):
        self.reverse_relative = kwargs.get("reverse_relative", False)
        self.order_by_type = kwargs.get("order_by_type", False)
        self.constants = kwargs.get("constants", [])
        self.classes = kwargs.append if "classes" in kwargs else kwargs.get("classes", [])
        self.variables = kwargs.get("variables", [])
        self.case_sensitive = kwargs.get("case_sensitive", True)
        self.length_sort = kwargs.get("length_sort", False)
        self.length_sort_straight = kwargs.get("length_sort_straight", False)
        self.length_sort_sections = kwargs.get("length_sort_sections", [])
        self.force_to_top = kwargs.get("force_to_top", [])

def test_module_key():
    # Test Basic functionality
    config = MockConfig()
    assert module_key("os", config) == "Bos"
    
    # Test Relative imports with reverse_relative=False (underscore)
    config_rel = MockConfig(reverse_relative=False)
    assert module_key(".utils", config_rel) == "B. _utils" # Note: regex captures (.+) and (.*)
    # Re-evaluating the regex match logic: re.match(r"^(\.+)\s*(.*)", ".utils") -> groups: ('.', 'utils')
    # sep = "_" -> "._utils"
    assert module_key(".utils", config_rel) == "B._utils"
    
    # Test Relative imports with reverse_relative=True (space)
    config_rev_rel = MockConfig(reverse_relative=True)
    assert module_key(".utils", config_rev_rel) == "B. .utils" 
    # Note: match.groups() for ".utils" is ('.', 'utils'). join with ' ' -> ". utils"
    # However, re.match(r"^(\.+)\s*(.*)", ".utils") on ".utils" matches group 1 as "." and group 2 as "utils"
    # If input is "..utils", group 1 is ".." and group 2 is "utils"
    
    # Test Case Insensitivity
    config_case = MockConfig(case_sensitive=False)
    assert module_key("OS", config_case) == "Bos"
    
    # Test Ignore Case parameter
    assert module_key("OS", config, ignore_case=True) == "Bos"

    # Test Force to Top
    config_top = MockConfig(force_to_top=["os"])
    assert module_key("os", config_top) == "Aos"

    # Test Order by Type: Constants (Prefix A)
    config_type = MockConfig(order_by_type=True, constants=["MY_CONST"])
    assert module_key("MY_CONST", config_type, sub_imports=True) == "BA"
    # Wait, the logic: if module_name in config.constants: prefix = "A". 
    # result = f"{(module_name in config.force_to_top and 'A') or 'B'}{prefix}{...}"
    # If not in force_to_top, it starts with 'B'.
    assert module_key("MY_CONST", config_type, sub_imports=True) == "BA"

    # Test Order by Type: Classes (Prefix B)
    config_type_class = MockConfig(order_by_type=True, classes=["MyClass"])
    assert module_key("MyClass", config_type_class, sub_imports=True) == "BB"

    # Test Order by Type: Variables (Prefix C)
    config_type_var = MockConfig(order_by_type=True, variables=["my_var"])
    assert module_key("my_var", config_type_var, sub_imports=True) == "BC"

    # Test Order by Type: Uppercase (Prefix A)
    config_type_upper = MockConfig(order_by_type=True)
    assert module_key("UPPER", config_type_upper, sub_imports=True) == "BA"

    # Test Length Sort
    config_len = MockConfig(length_sort=True)
    assert module_key("abc", config_len) == "B3:abc"

    # Test Length Sort via section_name
    config_sec = MockConfig(length_sort_sections=["MY_SECTION"])
    assert module_key("abc", config_sec, section_name="MY_SECTION") == "B3:abc"

    # Test length_sort_straight and straight_import
    config_straight = MockConfig(length_sort_straight=True)
    assert module_key("abc", config_straight, straight_import=True) == "B3:abc"
    assert module_key("abc", config_straight, straight_import=False) == "Babc"

    # Test complex relative path formatting
    # Input: "..utils", reverse_relative=False -> groups ('.', '.utils')? 
    # No, regex ^(\.+)\s*(.*) on "..utils" -> group 1 is "..", group 2 is "utils"
    # sep is "_" -> ".._utils"
    config_rel_complex = MockConfig(reverse_relative=False)
    assert module_key("..utils", config_rel_complex) == "B.._utils"

    # Test type conversion to string
    assert module_key(123, config) == "B123"
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_section_key():
    # Helper to create a mock config object
    def create_config(**kwargs):
        config = MagicMock()
        defaults = {
            "reverse_relative": False,
            "group_by_package": False,
            "lexicographical": False,
            "sort_relative_in_force_sorted_sections": False,
            "force_to_top": [],
            "honor_case_in_force_sorted_sections": False,
            "case_sensitive": True,
            "order_by_type": False,
            "length_sort": False,
            "sort_relative_in_force_sorted_sections": False,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(config, k, v)
        return config

    # 1. Basic functionality: simple import
    config = create_config()
    assert section_key("import os", config) == "Bos"

    # 2. Basic functionality: simple from import
    assert section_key("from os import path", config) == "Bpath" if config.lexicographical else "Bos import path"
    # Re-evaluating: if lexicographical is False, line = re.sub("^from ", "", line) -> "os import path"
    # The default is False.
    assert section_key("from os import path", config) == "Bos import path"

    # 3. Force to top (Section A)
    config = create_config(force_to_top=["os"])
    assert section_key("import os", config).startswith("A")

    # 4. Reverse relative imports (dots)
    config = create_config(reverse_relative=True, sort_relative_in_force_sorted_sections=True)
    # line starts with "from .", matches regex, line becomes "from .. "
    # then sort_relative_in_force_sorted_sections applies sep=" "
    # "from .. " -> "from . . "
    assert section_key("from .. import module", config) == "Bfrom . . import module"

    # 5. Group by package
    config = create_config(group_by_package=True)
    assert section_key("from os import path", config) == "Bos"

    # 6. Lexicographical mode
    config = create_config(lexicographical=True)
    # "from os import path" -> "os.path"
    assert section_key("from os import path", config) == "Bos.path"

    # 7. Length sort enabled
    config = create_config(length_sort=True)
    line = "import os"
    # line becomes "os" (since from/import stripped)
    # key = B + len("os") + "os" = B2os
    assert section_key("import os", config) == "B2os"

    # 8. Honor case in force sorted sections (Case Sensitive != Order By Type)
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True
    )
    # line = "os import Path"
    # module_name = "os", names = "path"
    assert section_key("from os import Path", config) == "Bos import path"

    # 9. Case sensitivity (not sensitive, order_by_type False)
    config = create_config(order_by_type=False, case_sensitive=False)
    assert section_key("import OS", config) == "Bos"

    # 10. Test string stripping/regex for imports
    config = create_config()
    assert section_key("import sys", config) == "Bsys"
    assert section_key("from sys import argv", config) == "Bsys import argv"

    # 11. Test sort_relative_in_force_sorted_sections with underscore sep
    config = create_config(
        sort_relative_in_force_sorted_sections=True, 
        reverse_relative=False
    )
    # "from .module" -> "from ._module"
    # stripped -> "._module"
    assert section_key("from .module", config) == "B._module"

    # 12. Test sort_relative_in_force_sorted_sections with space sep
    config = create_config(
        sort_relative_in_force_sorted_sections=True, 
        reverse_relative=True
    )
    # "from .module" -> "from . module"
    # stripped -> ". module"
    assert section_key("from .module", config) == "B. module"
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_section_key():
    # Helper to create a mock config object
    def create_config(
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        sort_relative_in_force_sorted_sections=False,
        force_to_top=None,
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=False,
        length_sort=False,
    ):
        config = MagicMock()
        config.reverse_relative = reverse_relatve
        config.group_by_package = group_by_package
        config.lexicographical = lexicographical
        config.sort_relative_in_force_sorted_sections = sort_relative_in_force_sorted_sections
        config.force_to_top = force_to_top or []
        config.honor_case_in_force_sorted_sections = honor_case_in_force_sorted_sections
        config.case_sensitive = case_sensitive
        config.order_by_type = order_by_type
        config.length_sort = length_sort
        return config

    # Case 1: Basic import, no special config
    config = create_config()
    assert section_key("import os", config) == "B3import os" # length 7 (len("import os") is 9, wait, 'import os' -> 'os' if not lexicographical)
    # Re-evaluating: line = re.sub("^import ", "", line) -> line becomes "os"
    # len("os") is 2. Result: "B2os"
    
    # Case 2: Force to top
    config = create_config(force_to_top=["os"])
    assert section_key("import os", config).startswith("A")

    # Case 3: Lexicographical mode
    # "from os import path" -> replace " import " with "." -> "from os.path" -> remove "from " -> "os.path"
    config = create_config(lexicographical=True)
    assert section_key("from os import path", config) == "B8os.path"

    # Case 4: Group by package
    # "from os import path" -> split at " import " -> "from os" -> remove "from " -> "os"
    config = create_config(group_by_package=True)
    assert section_key("from os import path", config) == "B2os"

    # Case 5: Relative imports with reverse_relative
    # "from . import utils" -> line = "from . utils"
    config = create_config(reverse_relative=True, sort_relative_in_force_sorted_sections=True)
    assert section_key("from . import utils", config) == "B12. utils"

    # Case 6: Case sensitivity and order_by_type (Honor case logic)
    # If honor_case_in_force_sorted_sections is True and case_sensitive != order_by_type
    # Input: "from OS import Path"
    # If case_sensitive=False, order_by_type=True: module_name=os, names=path
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True
    )
    # line "from OS import Path" -> split -> module="OS", names="Path" -> lower -> "os import path"
    # Then re.sub("^from ", "") -> "os import path"
    assert section_key("from OS import Path", config) == "B14os import path"

    # Case 7: Length sort enabled
    config = create_config(length_sort=True)
    # "import os" -> "os" (len 2)
    assert section_key("import os", config) == "B2os"

    # Case 8: Simple import (no from)
    config = create_config()
    # "import sys" -> "sys"
    assert section_key("import sys", config) == "B3sys"
```


