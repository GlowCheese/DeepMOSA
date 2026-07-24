####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test the module_key function with various configurations and inputs."""
    
    # Create a mock config object with default values
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = set()
    config.force_to_top = set()

    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"

    # Test with force_to_top
    config.force_to_top = {"os"}
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = set()

    # Test relative imports with reverse_relative False
    config.reverse_relative = False
    result = module_key("..utils", config)
    assert "utils" in result

    # Test relative imports with reverse_relative True
    config.reverse_relative = True
    result = module_key("..utils", config)
    assert "utils" in result

    # Test with ignore_case
    result = module_key("MyModule", config, ignore_case=True)
    assert "mymodule" in result.lower()

    # Test with case_sensitive False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    config.case_sensitive = True

    # Test with length_sort enabled
    config.length_sort = True
    result = module_key("os", config)
    assert "2:os" in result
    config.length_sort = False

    # Test with length_sort_straight and straight_import
    config.length_sort_straight = True
    result = module_key("sys", config, straight_import=True)
    assert "3:sys" in result
    config.length_sort_straight = False

    # Test with length_sort_sections
    config.length_sort_sections = {"thirdparty"}
    result = module_key("module", config, section_name="thirdparty")
    assert "6:module" in result
    config.length_sort_sections = set()

    # Test with order_by_type and constants
    config.order_by_type = True
    config.constants = {"MAX_VALUE"}
    result = module_key("MAX_VALUE", config, sub_imports=True)
    assert result.startswith("BA")
    config.constants = set()

    # Test with order_by_type and classes
    config.classes = {"MyClass"}
    result = module_key("MyClass", config, sub_imports=True)
    assert result.startswith("BB")
    config.classes = set()

    # Test with order_by_type and variables
    config.variables = {"my_var"}
    result = module_key("my_var", config, sub_imports=True)
    assert result.startswith("BC")
    config.variables = set()

    # Test with order_by_type and uppercase module (issue #376)
    config.order_by_type = True
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result.startswith("BA")

    # Test with order_by_type and class-like name (starts with uppercase)
    result = module_key("ClassName", config, sub_imports=True)
    assert result.startswith("BB")

    # Test with order_by_type and regular variable
    result = module_key("variable_name", config, sub_imports=True)
    assert result.startswith("BC")
    config.order_by_type = False

    # Test empty module name
    result = module_key("", config)
    assert isinstance(result, str)

    # Test with multiple dots in relative import
    config.reverse_relative = False
    result = module_key("...package.module", config)
    assert isinstance(result, str)

    # Test combining multiple options
    config.case_sensitive = False
    config.length_sort = True
    config.force_to_top = {"special"}
    result = module_key("special", config)
    assert result.startswith("A")
    assert "7:special" in result


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations."""
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = []
    
    # Test with relative imports
    result = module_key("...module", config)
    assert result == "B___module"
    
    # Test with reverse_relative
    config.reverse_relative = True
    result = module_key("...module", config)
    assert result == "B...module"
    config.reverse_relative = False
    
    # Test ignore_case
    result = module_key("MyModule", config, ignore_case=True)
    assert "mymodule" in result
    
    # Test case_sensitive = False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert "mymodule" in result
    config.case_sensitive = True
    
    # Test length_sort
    config.length_sort = True
    result = module_key("os", config)
    assert "2:os" in result
    config.length_sort = False
    
    # Test sub_imports with order_by_type
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result.startswith("BA")
    
    # Test sub_imports with classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert result.startswith("BB")
    config.classes = []
    
    # Test sub_imports with variables
    config.variables = ["var"]
    result = module_key("var", config, sub_imports=True)
    assert result.startswith("BC")
    config.variables = []
    
    # Test uppercase variable (issue #376)
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result.startswith("BA")
    
    # Test sub_imports with lowercase (should be prefix C)
    config.order_by_type = True
    result = module_key("lowercase", config, sub_imports=True)
    assert result.startswith("BC")
    config.order_by_type = False
    
    # Test length_sort_straight
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    config.length_sort_straight = False
    
    # Test length_sort_sections
    config.length_sort_sections = ["thirdparty"]
    result = module_key("module", config, section_name="thirdparty")
    assert "6:module" in result
    config.length_sort_sections = []
    
    # Test with empty module name
    result = module_key("", config)
    assert isinstance(result, str)
    
    # Test numeric module name
    result = module_key("123module", config)
    assert isinstance(result, str)


# LLM-generated content at query #3
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test with from import
    config.force_to_top = []
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test lexicographical sorting
    config.lexicographical = True
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from os import path", config)
    assert "." in result
    assert "os" in result
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    result = section_key("from os import path, environ", config)
    assert "import" not in result
    assert "os" in result
    
    # Test case_sensitive False
    config.case_sensitive = False
    config.order_by_type = True
    config.group_by_package = False
    config.honor_case_in_force_sorted_sections = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test length_sort
    config.case_sensitive = True
    config.length_sort = True
    result = section_key("import os", config)
    assert "2:" in result  # length of "os"
    
    # Test honor_case_in_force_sorted_sections with mixed settings
    config.length_sort = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from os import PATH", config)
    assert result.startswith("B")
    
    # Test order_by_type False
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test relative imports with dots
    config.order_by_type = True
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from .. import module", config)
    assert result.startswith("B")
    
    # Test multiple spaces in relative imports
    config.reverse_relative = False
    result = section_key("from .  module import func", config)
    assert result.startswith("B")
    
    # Test empty force_to_top with special module
    config.force_to_top = ["__future__"]
    result = section_key("import __future__", config)
    assert result.startswith("A")
    
    # Test case where line doesn't contain " import "
    config.force_to_top = []
    result = section_key("import sys", config)
    assert result.startswith("B")
    assert "sys" in result


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    """Test section_key function with various configurations."""
    
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test with from import
    config.force_to_top = []
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test lexicographical mode
    config.lexicographical = True
    result = section_key("import os", config)
    assert "os" in result
    config.lexicographical = False
    
    # Test group_by_package
    config.group_by_package = True
    result = section_key("from os import path", config)
    assert "os" in result
    config.group_by_package = False
    
    # Test reverse_relative with relative imports
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test case_sensitive = False
    config.case_sensitive = False
    config.order_by_type = True
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test order_by_type = False
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from os import PATH", config)
    assert "path" in result.lower()
    
    # Test length_sort
    config.length_sort = True
    config.order_by_type = True
    result = section_key("import os", config)
    assert "2" in result  # length of "os"
    
    # Test honor_case_in_force_sorted_sections with split
    config.length_sort = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from os import Path", config)
    assert "B" in result
    assert "os" in result
    
    # Test multiple relative dots
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    result = section_key("from ... import module", config)
    assert result.startswith("B")
    
    # Test with relative and reverse_relative
    config.reverse_relative = True
    result = section_key("from .. import something", config)
    assert result.startswith("B")


# LLM-generated content at query #5
#--------------------------

```python
def test_module_key():
    """Test module_key function with various configurations."""
    from unittest.mock import Mock
    
    # Create mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with ignore_case
    result = module_key("OS", config, ignore_case=True)
    assert result == "Bos"
    
    # Test case_sensitive = False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    
    # Test relative imports
    config.case_sensitive = True
    config.reverse_relative = False
    result = module_key("...mymodule", config)
    assert "mymodule" in result
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    result = module_key("..mymodule", config)
    assert "mymodule" in result
    
    # Test force_to_top
    config.reverse_relative = False
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result.startswith("Aos")
    
    # Test sub_imports with order_by_type
    config.force_to_top = []
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    
    # Test sub_imports with classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    # Test sub_imports with variables
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result
    
    # Test uppercase module (issue #376)
    config.order_by_type = True
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert "A" in result
    
    # Test length_sort
    config.order_by_type = False
    config.length_sort = True
    result = module_key("module", config)
    assert "6:module" in result
    
    # Test length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    
    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["stdlib"]
    result = module_key("sys", config, section_name="stdlib")
    assert "3:sys" in result
    
    # Test combination of relative import with other options
    config.length_sort_sections = []
    config.reverse_relative = False
    result = module_key(".module", config)
    assert "module" in result
    
    # Test empty module name
    result = module_key("", config)
    assert isinstance(result, str)


# LLM-generated content at query #6
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test with reverse_relative and relative import
    config.force_to_top = []
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test with lexicographical sorting
    config.lexicographical = True
    config.reverse_relative = False
    result = section_key("from os import path", config)
    assert "." in result or "os" in result
    
    # Test with group_by_package
    config.lexicographical = False
    config.group_by_package = True
    result = section_key("from os import path", config)
    assert "os" in result
    
    # Test with case_sensitive and order_by_type different
    config.group_by_package = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    result = section_key("import OS", config)
    assert result.startswith("B")
    
    # Test with order_by_type False
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = False
    result = section_key("import MyModule", config)
    assert "mymodule" in result.lower()
    
    # Test length_sort enabled
    config.order_by_type = True
    config.length_sort = True
    result = section_key("import os", config)
    assert "2:" in result  # length of "os" is 2
    
    # Test with sort_relative_in_force_sorted_sections
    config.length_sort = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from .. import module", config)
    assert ".._" in result or ".. " in result
    
    # Test from vs import prefix removal
    config.sort_relative_in_force_sorted_sections = False
    result1 = section_key("from os import path", config)
    result2 = section_key("import os", config)
    # Both should contain "os" after prefix removal
    assert "os" in result1
    assert "os" in result2
    
    # Test with multiple spaces in relative imports
    config.reverse_relative = True
    result = section_key("from ... import something", config)
    assert result.startswith("B")


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations."""
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = []
    
    # Test with ignore_case
    result = module_key("MyModule", config, ignore_case=True)
    assert result == "Bmymodule"
    
    # Test with ignore_case=False
    result = module_key("MyModule", config, ignore_case=False)
    assert result == "BMyModule"
    
    # Test case_sensitive=False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    config.case_sensitive = True
    
    # Test relative imports with reverse_relative=False
    result = module_key("...module", config)
    assert result == "B..._module"
    
    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    result = module_key("...module", config)
    assert result == "B... module"
    config.reverse_relative = False
    
    # Test with sub_imports and order_by_type
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result.startswith("BA")
    
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert result.startswith("BB")
    
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert result.startswith("BC")
    
    # Test uppercase module name (issue #376)
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert result.startswith("BA")
    
    # Test class-like module name
    result = module_key("MyModule", config, sub_imports=True)
    assert result.startswith("BB")
    
    config.order_by_type = False
    
    # Test length_sort
    config.length_sort = True
    result = module_key("module", config)
    assert "6:module" in result
    config.length_sort = False
    
    # Test length_sort_straight
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert "6:module" in result
    config.length_sort_straight = False
    
    # Test length_sort_sections
    config.length_sort_sections = ["future"]
    result = module_key("module", config, section_name="future")
    assert "6:module" in result
    config.length_sort_sections = []
    
    # Test empty module name
    result = module_key("", config)
    assert result == "B"
    
    # Test numeric in module name
    result = module_key("module123", config)
    assert result == "Bmodule123"
    
    # Test force_to_top with multiple items
    config.force_to_top = ["os", "sys"]
    result = module_key("os", config)
    assert result == "Aos"
    result = module_key("sys", config)
    assert result == "Asys"
    result = module_key("other", config)
    assert result == "Bother"
    
    # Test relative import with single dot
    config.force_to_top = []
    config.reverse_relative = False
    result = module_key(".module", config)
    assert result == "B._module"
    
    # Test relative import with multiple dots
    result = module_key("....module", config)
    assert result == "B...._module"


# LLM-generated content at query #8
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["sys"]
    result = module_key("sys", config)
    assert result == "Asys"
    
    # Test with ignore_case
    config.force_to_top = []
    result = module_key("MyModule", config, ignore_case=True)
    assert "mymodule" in result
    
    # Test relative imports with reverse_relative=False
    config.reverse_relative = False
    result = module_key("..utils", config)
    assert "_" in result
    
    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    result = module_key("..utils", config)
    assert " " in result
    
    # Test with case_sensitive=False
    config.case_sensitive = False
    config.reverse_relative = False
    result = module_key("MyModule", config)
    assert "mymodule" in result
    
    # Test with length_sort=True
    config.case_sensitive = True
    config.length_sort = True
    result = module_key("abc", config)
    assert "3:abc" in result
    
    # Test with length_sort_straight=True and straight_import=True
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    
    # Test with sub_imports and order_by_type
    config.length_sort_straight = False
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    
    # Test with sub_imports and classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    # Test with sub_imports and variables
    config.variables = ["myvar"]
    result = module_key("myvar", config, sub_imports=True)
    assert "C" in result
    
    # Test with sub_imports and uppercase variable (issue #376)
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert "A" in result
    
    # Test with sub_imports and class-like name
    result = module_key("MyName", config, sub_imports=True)
    assert "B" in result
    
    # Test with sub_imports and regular variable
    result = module_key("myname", config, sub_imports=True)
    assert "C" in result
    
    # Test with section_name in length_sort_sections
    config.order_by_type = False
    config.length_sort = False
    config.length_sort_sections = ["future"]
    result = module_key("os", config, section_name="future")
    assert ":" in result
    
    # Test empty module name
    result = module_key("", config)
    assert isinstance(result, str)
    
    # Test relative import with single dot
    config.length_sort_sections = []
    config.reverse_relative = False
    result = module_key(".utils", config)
    assert "_" in result
    
    # Test relative import with multiple dots
    result = module_key("...utils", config)
    assert "_" in result


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test the module_key function with various configurations and inputs."""
    
    # Mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = set()
    config.force_to_top = set()
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = {"os"}
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = set()
    
    # Test with relative imports (reverse_relative=False)
    config.reverse_relative = False
    result = module_key("...utils", config)
    assert "._._._utils" in result or "..." in result
    
    # Test with relative imports (reverse_relative=True)
    config.reverse_relative = True
    result = module_key("...utils", config)
    assert result is not None
    
    # Test ignore_case parameter
    result = module_key("MyModule", config, ignore_case=True)
    assert "mymodule" in result.lower()
    
    # Test case_sensitive=False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    config.case_sensitive = True
    
    # Test length_sort
    config.length_sort = True
    result = module_key("module", config)
    assert "6:module" in result
    config.length_sort = False
    
    # Test length_sort_straight
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert "6:module" in result
    config.length_sort_straight = False
    
    # Test order_by_type with constants
    config.order_by_type = True
    config.constants = {"MY_CONSTANT"}
    result = module_key("MY_CONSTANT", config, sub_imports=True)
    assert "A" in result
    config.constants = set()
    
    # Test order_by_type with classes
    config.classes = {"MyClass"}
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    config.classes = set()
    
    # Test order_by_type with variables
    config.variables = {"my_var"}
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result
    config.variables = set()
    
    # Test order_by_type with uppercase (issue #376)
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    
    # Test order_by_type with capitalized name
    result = module_key("ClassName", config, sub_imports=True)
    assert "B" in result
    
    # Test order_by_type with lowercase name
    config.order_by_type = True
    result = module_key("variable", config, sub_imports=True)
    assert "C" in result
    config.order_by_type = False
    
    # Test length_sort_sections
    config.length_sort_sections = {"thirdparty"}
    result = module_key("requests", config, section_name="thirdparty")
    assert "8:requests" in result
    config.length_sort_sections = set()
    
    # Test empty module name
    result = module_key("", config)
    assert result is not None
    
    # Test with sub_imports=False
    result = module_key("module", config, sub_imports=False)
    assert result == "Bmodule"
    
    # Test section_name parameter
    result = module_key("module", config, section_name="stdlib")
    assert result is not None


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    """Test the section_key function with various configurations."""
    
    # Mock config object
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    
    # Test basic import statement
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import statement
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top configuration
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    config.force_to_top = []
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test relative imports with sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from . import module", config)
    assert "module" in result
    
    # Test lexicographical sorting
    config.lexicographical = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from os import path", config)
    assert "." in result
    config.lexicographical = False
    
    # Test group_by_package
    config.group_by_package = True
    result = section_key("from os import path, walk", config)
    assert "import" not in result
    config.group_by_package = False
    
    # Test case sensitivity
    config.case_sensitive = False
    config.order_by_type = False
    result = section_key("from OS import PATH", config)
    assert "os" in result.lower()
    
    # Test order_by_type with honor_case_in_force_sorted_sections
    config.case_sensitive = True
    config.order_by_type = True
    config.honor_case_in_force_sorted_sections = False
    result = section_key("from module import Name", config)
    assert result.startswith("B")
    
    # Test honor_case_in_force_sorted_sections with different case_sensitive and order_by_type
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from Module import name", config)
    assert result.startswith("B")
    
    # Test length_sort
    config.length_sort = True
    config.honor_case_in_force_sorted_sections = False
    result = section_key("import os", config)
    assert "2" in result or "3" in result
    config.length_sort = False
    
    # Test multiple dot relative imports
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from ... import module", config)
    assert result.startswith("B")
    
    # Test empty force_to_top list
    config.force_to_top = []
    result = section_key("import sys", config)
    assert result.startswith("B")
    
    # Test with special characters in import
    result = section_key("from my_module import my_function", config)
    assert result.startswith("B")


# LLM-generated content at query #11
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test with length_sort enabled
    config.force_to_top = []
    config.length_sort = True
    result = section_key("import os", config)
    assert "2:" in result  # length of "os" is 2
    
    # Test relative imports with reverse_relative
    config.length_sort = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from . import test", config)
    assert "." in result
    
    # Test group_by_package
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = False
    config.group_by_package = True
    result = section_key("from os import path", config)
    assert "os" in result
    assert "path" not in result
    
    # Test lexicographical
    config.group_by_package = False
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert result.startswith("B")
    
    # Test case_sensitive with order_by_type mismatch
    config.lexicographical = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    result = section_key("from OS import Path", config)
    assert result.startswith("B")
    
    # Test case insensitive sorting when order_by_type is False
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("import MyModule", config)
    assert "mymodule" in result.lower()
    
    # Test from import statement
    config.order_by_type = True
    result = section_key("from collections import defaultdict", config)
    assert result.startswith("B")
    
    # Test multiple relative dots
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from ... import something", config)
    assert "." in result
    
    # Test empty force_to_top with normal module
    config.force_to_top = []
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("import unittest", config)
    assert result.startswith("B")


# LLM-generated content at query #12
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test with from import
    config.force_to_top = []
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test reverse_relative with relative imports
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import something", config)
    assert result.startswith("B")
    
    # Test lexicographical sorting
    config.lexicographical = True
    config.reverse_relative = False
    result = section_key("from os import path", config)
    assert "os.path" in result
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    result = section_key("from os import path", config)
    assert "os" in result
    assert "import" not in result
    
    # Test case_sensitive False
    config.case_sensitive = False
    config.order_by_type = False
    config.group_by_package = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test length_sort
    config.length_sort = True
    config.case_sensitive = True
    config.order_by_type = True
    result = section_key("import os", config)
    assert "2:" in result  # length of "os" is 2
    
    # Test sort_relative_in_force_sorted_sections
    config.length_sort = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from .. import something", config)
    assert ".._" in result or ".." in result
    
    # Test honor_case_in_force_sorted_sections with different case_sensitive and order_by_type
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from os import Path", config)
    assert "path" in result.lower()
    
    # Test relative imports with reverse_relative
    config.honor_case_in_force_sorted_sections = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from ... import something", config)
    assert result.startswith("B")
    
    # Test with multiple relative dots
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from ...package import item", config)
    assert result.startswith("B")


# LLM-generated content at query #13
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = set()
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = {"os"}
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = set()
    
    # Test with ignore_case
    result = module_key("MyModule", config, ignore_case=True)
    assert result == "Bmymodule"
    
    # Test with relative imports (dots)
    config.reverse_relative = False
    result = module_key("...module", config)
    assert "module" in result
    
    # Test with reverse_relative
    config.reverse_relative = True
    result = module_key("...module", config)
    assert "module" in result
    
    # Test sub_imports with order_by_type
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    
    config.classes = {"MyClass"}
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    config.variables = {"var"}
    result = module_key("var", config, sub_imports=True)
    assert "C" in result
    
    # Test uppercase variable (issue #376)
    config.order_by_type = True
    config.constants = set()
    config.classes = set()
    config.variables = set()
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert "A" in result
    
    # Test case_sensitive = False
    config.order_by_type = False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert "mymodule" in result.lower()
    
    # Test length_sort
    config.case_sensitive = True
    config.length_sort = True
    result = module_key("os", config)
    assert "2:" in result
    
    # Test length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("os", config, straight_import=True)
    assert "2:" in result
    
    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["stdlib"]
    result = module_key("os", config, section_name="stdlib")
    assert "2:" in result
    
    # Test module name in force_to_top
    config.length_sort_sections = []
    config.force_to_top = {"os"}
    result = module_key("os", config)
    assert result.startswith("A")
    
    result = module_key("sys", config)
    assert result.startswith("B")


# LLM-generated content at query #14
#--------------------------

```python
def test_module_key():
    """Test the module_key function with various configurations."""
    from unittest.mock import Mock
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = []
    
    # Test relative imports
    result = module_key("..utils", config)
    assert result == "B..utils"
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    result = module_key("..utils", config)
    assert result == "B. . utils"
    config.reverse_relative = False
    
    # Test ignore_case
    result = module_key("MyModule", config, ignore_case=True)
    assert result == "Bmymodule"
    
    # Test case_sensitive = False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    config.case_sensitive = True
    
    # Test sub_imports with order_by_type
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result.startswith("BA")
    
    # Test sub_imports with classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert result.startswith("BB")
    
    # Test sub_imports with variables
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert result.startswith("BC")
    
    # Test sub_imports with uppercase constant detection
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("CONST", config, sub_imports=True)
    assert result.startswith("BA")
    
    # Test sub_imports with class detection (first letter uppercase)
    result = module_key("MyClass", config, sub_imports=True)
    assert result.startswith("BB")
    
    # Test length_sort
    config.order_by_type = False
    config.length_sort = True
    result = module_key("abc", config)
    assert "3:abc" in result
    config.length_sort = False
    
    # Test length_sort_straight
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    config.length_sort_straight = False
    
    # Test length_sort_sections
    config.length_sort_sections = ["future"]
    result = module_key("os", config, section_name="future")
    assert "2:os" in result
    config.length_sort_sections = []
    
    # Test relative imports with multiple dots
    result = module_key("...package.module", config)
    assert "..." in result
    
    # Test empty module name
    result = module_key("", config)
    assert result == "B"
    
    # Test single character module
    result = module_key("a", config)
    assert result == "Ba"


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations and inputs."""
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = []
    
    # Test with relative imports
    result = module_key("..utils", config)
    assert result == "B_utils"
    
    # Test with reverse_relative
    config.reverse_relative = True
    result = module_key("..utils", config)
    assert result == "B utils"
    config.reverse_relative = False
    
    # Test case_sensitive = False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    config.case_sensitive = True
    
    # Test ignore_case parameter
    result = module_key("MyModule", config, ignore_case=True)
    assert result == "Bmymodule"
    
    # Test length_sort
    config.length_sort = True
    result = module_key("abc", config)
    assert "3:abc" in result
    config.length_sort = False
    
    # Test length_sort_straight with straight_import=True
    config.length_sort_straight = True
    result = module_key("abcd", config, straight_import=True)
    assert "4:abcd" in result
    config.length_sort_straight = False
    
    # Test length_sort_sections
    config.length_sort_sections = ["thirdparty"]
    result = module_key("abc", config, section_name="thirdparty")
    assert "3:abc" in result
    config.length_sort_sections = []
    
    # Test order_by_type with constants
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    config.constants = []
    
    # Test order_by_type with classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    config.classes = []
    
    # Test order_by_type with variables
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result
    config.variables = []
    
    # Test uppercase name detection (issue #376)
    config.order_by_type = True
    result = module_key("CONSTANT_NAME", config, sub_imports=True)
    assert "A" in result
    
    # Test uppercase single letter detection
    result = module_key("A", config, sub_imports=True)
    assert result == "BA"
    
    # Test uppercase first letter detection
    config.classes = []
    result = module_key("MyVar", config, sub_imports=True)
    assert "B" in result
    
    # Test lowercase variable
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result
    config.order_by_type = False
    
    # Test multiple relative dots
    result = module_key("....module", config)
    assert "module" in result
    
    # Test empty relative part
    result = module_key(".", config)
    assert result == "B"
    
    # Test complex relative import
    config.reverse_relative = False
    result = module_key("...pkg.module", config)
    assert "pkg.module" in result


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations."""
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = set()
    config.force_to_top = set()

    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"

    # Test with force_to_top
    config.force_to_top = {"os"}
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = set()

    # Test with relative imports
    config.reverse_relative = False
    result = module_key("...module", config)
    assert "module" in result

    # Test with reverse_relative
    config.reverse_relative = True
    result = module_key("..module", config)
    assert result is not None

    # Test ignore_case
    config.case_sensitive = True
    result = module_key("MyModule", config, ignore_case=True)
    assert "mymodule" in result.lower()

    # Test sub_imports with order_by_type
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result

    # Test sub_imports with classes
    config.constants = set()
    config.classes = {"MyClass"}
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result

    # Test sub_imports with variables
    config.classes = set()
    config.variables = {"my_var"}
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result

    # Test uppercase detection in sub_imports
    config.variables = set()
    result = module_key("UPPERCASE", config, sub_imports=True, ignore_case=False)
    assert "A" in result

    # Test capitalized detection in sub_imports
    result = module_key("Capitalized", config, sub_imports=True)
    assert "B" in result

    # Test lowercase detection in sub_imports
    result = module_key("lowercase", config, sub_imports=True)
    assert "C" in result

    # Test case_sensitive = False
    config.order_by_type = False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert "mymodule" in result.lower()

    # Test length_sort
    config.case_sensitive = True
    config.length_sort = True
    result = module_key("os", config)
    assert "2:" in result

    # Test length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("os", config, straight_import=True)
    assert "2:" in result

    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = {"stdlib"}
    result = module_key("os", config, section_name="stdlib")
    assert "2:" in result

    # Test relative imports with spaces
    config.reverse_relative = True
    result = module_key("...module.name", config)
    assert result is not None

    # Test empty module name
    result = module_key("", config)
    assert "B" in result

    # Test complex relative import
    config.reverse_relative = False
    result = module_key(".module", config)
    assert "module" in result or "_" in result


# LLM-generated content at query #17
#--------------------------

```python
def test_section_key():
    """Test section_key function with various configurations."""
    from unittest.mock import Mock
    
    # Create a mock config with default values
    def create_config(**kwargs):
        config = Mock()
        config.sort_relative_in_force_sorted_sections = False
        config.reverse_relative = False
        config.group_by_package = False
        config.lexicographical = False
        config.force_to_top = []
        config.honor_case_in_force_sorted_sections = False
        config.case_sensitive = True
        config.order_by_type = False
        config.length_sort = False
        
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config
    
    # Test basic import line
    config = create_config()
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import line
    config = create_config()
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config = create_config(force_to_top=["os"])
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test reverse_relative with relative imports
    config = create_config(reverse_relative=True)
    result = section_key("from . import something", config)
    assert "something" in result
    
    # Test group_by_package
    config = create_config(group_by_package=True)
    result = section_key("from os import path, walk", config)
    assert "os" in result
    
    # Test lexicographical sorting
    config = create_config(lexicographical=True)
    result = section_key("from os import path", config)
    assert "." in result  # lexicographical replaces " import " with "."
    
    # Test case_sensitive = False
    config = create_config(case_sensitive=False)
    result = section_key("import OS", config)
    assert result.lower() == result
    
    # Test order_by_type = False with case_sensitive = True
    config = create_config(case_sensitive=True, order_by_type=False)
    result = section_key("import MyModule", config)
    assert "mymodule" in result
    
    # Test length_sort
    config = create_config(length_sort=True)
    result = section_key("import os", config)
    assert len(result) > 0
    
    # Test sort_relative_in_force_sorted_sections
    config = create_config(sort_relative_in_force_sorted_sections=True, reverse_relative=False)
    result = section_key("from .. import something", config)
    assert "_" in result
    
    # Test sort_relative_in_force_sorted_sections with reverse_relative
    config = create_config(sort_relative_in_force_sorted_sections=True, reverse_relative=True)
    result = section_key("from ... import something", config)
    assert " " in result
    
    # Test honor_case_in_force_sorted_sections with mixed settings
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=True,
        order_by_type=False
    )
    result = section_key("from os import PATH", config)
    assert "os" in result
    
    # Test multiple dots in relative import
    config = create_config()
    result = section_key("from ... import module", config)
    assert "module" in result
    
    # Test empty force_to_top list
    config = create_config(force_to_top=[])
    result = section_key("import sys", config)
    assert result.startswith("B")
    
    # Test that section starts with A or B
    config = create_config()
    result = section_key("import anything", config)
    assert result[0] in ["A", "B"]


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations and inputs."""
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = []
    
    # Test case insensitive
    result = module_key("MyModule", config, ignore_case=True)
    assert result == "Bmymodule"
    
    # Test case sensitive (default)
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    config.case_sensitive = True
    
    # Test relative imports with reverse_relative False
    result = module_key("..module", config)
    assert result == "B__module"
    
    # Test relative imports with reverse_relative True
    config.reverse_relative = True
    result = module_key("..module", config)
    assert result == "B  module"
    config.reverse_relative = False
    
    # Test length_sort
    config.length_sort = True
    result = module_key("os", config)
    assert result == "B2:os"
    config.length_sort = False
    
    # Test length_sort_straight
    config.length_sort_straight = True
    result = module_key("sys", config, straight_import=True)
    assert result == "B3:sys"
    config.length_sort_straight = False
    
    # Test length_sort_sections
    config.length_sort_sections = ["thirdparty"]
    result = module_key("requests", config, section_name="thirdparty")
    assert result == "B8:requests"
    config.length_sort_sections = []
    
    # Test sub_imports with order_by_type
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result == "BAconstant"
    
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert result == "BBmyclass"
    
    config.variables = ["var"]
    result = module_key("var", config, sub_imports=True)
    assert result == "BCvar"
    config.order_by_type = False
    
    # Test uppercase name detection (issue #376)
    config.order_by_type = True
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert result == "BAuppercase"
    config.order_by_type = False
    
    # Test single letter uppercase
    config.order_by_type = True
    result = module_key("X", config, sub_imports=True)
    assert result == "Bx"
    config.order_by_type = False
    
    # Test class-like name detection
    config.order_by_type = True
    result = module_key("MyClass", config, sub_imports=True)
    assert result == "BBmyclass"
    config.order_by_type = False
    
    # Test lowercase name (should get C prefix)
    config.order_by_type = True
    result = module_key("variable", config, sub_imports=True)
    assert result == "BCvariable"
    config.order_by_type = False
    
    # Test multiple relative dots
    result = module_key("...package.module", config)
    assert result == "B___package.module"
    
    # Test complex scenario with multiple options
    config.case_sensitive = False
    config.force_to_top = ["django"]
    result = module_key("django", config)
    assert result == "Adjango"
    
    # Test empty module name
    result = module_key("", config)
    assert result == "B"
    
    # Test with all features enabled
    config.case_sensitive = False
    config.length_sort = True
    config.order_by_type = True
    config.force_to_top = []
    result = module_key("mymodule", config, sub_imports=True)
    assert ":" in result
    assert "mymodule" in result


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    """Test the section_key function with various configurations and input lines."""
    
    # Test 1: Basic import line without special config
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test 2: From import line
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test 3: Force to top module
    config.force_to_top = ["django"]
    result = section_key("import django", config)
    assert result.startswith("A")
    
    # Test 4: Relative imports with reverse_relative
    config.force_to_top = []
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import module", config)
    assert "B" in result
    
    # Test 5: Group by package
    config.group_by_package = True
    config.reverse_relative = False
    result = section_key("from os import path, sep", config)
    assert "os" in result
    assert "import" not in result
    
    # Test 6: Lexicographical sorting
    config.group_by_package = False
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "os.path" in result
    
    # Test 7: Case insensitive
    config.lexicographical = False
    config.order_by_type = True
    config.case_sensitive = False
    result = section_key("import MyModule", config)
    assert "mymodule" in result.lower()
    
    # Test 8: Length sort enabled
    config.order_by_type = False
    config.length_sort = True
    result = section_key("import os", config)
    assert "2:" in result  # Length of "os" is 2
    
    # Test 9: Sort relative in force sorted sections
    config.length_sort = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from .. import module", config)
    assert ".." in result
    
    # Test 10: Honor case in force sorted sections
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from os import Path, sep", config)
    assert "os import" in result
    assert "path" in result.lower()
    
    # Test 11: Multiple relative dots
    config.honor_case_in_force_sorted_sections = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from ... import module", config)
    assert "B" in result
    
    # Test 12: Empty force_to_top with various imports
    config.force_to_top = []
    result = section_key("import sys", config)
    assert result.startswith("B")
    
    # Test 13: Relative import with sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from .submodule import func", config)
    assert "B" in result
    
    # Test 14: Case sensitive with order_by_type mismatch
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from module import Class", config)
    assert "class" in result.lower()
    
    # Test 15: Very simple import
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = False
    result = section_key("import a", config)
    assert "B" in result


# LLM-generated content at query #20
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["sys"]
    result = module_key("sys", config)
    assert result == "Asys"
    
    # Test with relative imports
    config.force_to_top = []
    config.reverse_relative = False
    result = module_key("..utils", config)
    assert "utils" in result
    
    # Test with reverse_relative
    config.reverse_relative = True
    result = module_key("..utils", config)
    assert "utils" in result
    
    # Test with ignore_case
    result = module_key("MyModule", config, ignore_case=True)
    assert "mymodule" in result.lower()
    
    # Test with case_sensitive False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    
    # Test with length_sort
    config.case_sensitive = True
    config.length_sort = True
    result = module_key("test", config)
    assert "4:test" in result
    
    # Test with length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    
    # Test with length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["stdlib"]
    result = module_key("test", config, section_name="stdlib")
    assert "4:test" in result
    
    # Test sub_imports with order_by_type
    config.length_sort_sections = []
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    
    # Test sub_imports with classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    # Test sub_imports with variables
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result
    
    # Test sub_imports with uppercase (issue #376)
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert "A" in result
    
    # Test sub_imports with class-like name (starts with uppercase)
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    # Test sub_imports with lowercase (variable)
    result = module_key("lowercase", config, sub_imports=True)
    assert "C" in result


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    # Test basic import line
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=[],
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False,
    )
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import line
    config.order_by_type = False
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result.lower()
    
    # Test force_to_top module
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test relative imports with reverse_relative
    config.force_to_top = []
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import something", config)
    assert result.startswith("B")
    
    # Test relative imports with sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from . import something", config)
    assert result.startswith("B")
    
    # Test group_by_package
    config.group_by_package = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from os import path, sep", config)
    assert "import" not in result.lower()
    
    # Test lexicographical
    config.group_by_package = False
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "." in result
    
    # Test length_sort
    config.lexicographical = False
    config.length_sort = True
    result = section_key("import os", config)
    assert "2" in result  # length of "os"
    
    # Test case_sensitive with order_by_type mismatch
    config.length_sort = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    result = section_key("from OS import Path", config)
    assert "os" in result.lower()
    
    # Test multiple relative dots
    config.honor_case_in_force_sorted_sections = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from ... import module", config)
    assert result.startswith("B")
    
    # Test empty force_to_top with case insensitive
    config.case_sensitive = True
    config.order_by_type = False
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("import MyModule", config)
    assert "mymodule" in result.lower()
    
    # Test complex from import with honor_case_in_force_sorted_sections
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from MyModule import MyClass", config)
    assert result.startswith("B")
    assert "MyModule" in result
    assert "myclass" in result.lower()


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    """Test the section_key function with various configurations and inputs."""
    
    # Create a mock config object with default settings
    def create_config(**kwargs):
        config = Mock()
        config.sort_relative_in_force_sorted_sections = kwargs.get('sort_relative_in_force_sorted_sections', False)
        config.reverse_relative = kwargs.get('reverse_relative', False)
        config.group_by_package = kwargs.get('group_by_package', False)
        config.lexicographical = kwargs.get('lexicographical', False)
        config.force_to_top = kwargs.get('force_to_top', [])
        config.honor_case_in_force_sorted_sections = kwargs.get('honor_case_in_force_sorted_sections', False)
        config.case_sensitive = kwargs.get('case_sensitive', True)
        config.order_by_type = kwargs.get('order_by_type', False)
        config.length_sort = kwargs.get('length_sort', False)
        return config
    
    # Test basic import statement
    config = create_config()
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import statement
    config = create_config()
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config = create_config(force_to_top=["os"])
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test relative imports with reverse_relative
    config = create_config(reverse_relative=True, sort_relative_in_force_sorted_sections=False)
    result = section_key("from . import something", config)
    assert result.startswith("B")
    
    # Test group_by_package
    config = create_config(group_by_package=True)
    result = section_key("from os import path, sep", config)
    assert "os" in result
    assert "import" not in result or result.index("os") < result.index("import")
    
    # Test lexicographical sorting
    config = create_config(lexicographical=True)
    result = section_key("from os import path", config)
    assert "os.path" in result
    
    # Test case insensitive with order_by_type=False
    config = create_config(case_sensitive=False, order_by_type=False)
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test honor_case_in_force_sorted_sections with different case_sensitive and order_by_type
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=True,
        order_by_type=False
    )
    result = section_key("from os import PATH", config)
    assert result.startswith("B")
    
    # Test honor_case_in_force_sorted_sections with split module
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True
    )
    result = section_key("from MyModule import MyClass", config)
    assert "mymodule" in result.lower()
    
    # Test length_sort
    config = create_config(length_sort=True)
    result = section_key("import os", config)
    assert "2" in result  # length of "os"
    
    # Test relative imports with sort_relative_in_force_sorted_sections
    config = create_config(
        sort_relative_in_force_sorted_sections=True,
        reverse_relative=False
    )
    result = section_key("from ... import something", config)
    assert result.startswith("B")
    
    # Test relative imports with sort_relative_in_force_sorted_sections and reverse_relative
    config = create_config(
        sort_relative_in_force_sorted_sections=True,
        reverse_relative=True
    )
    result = section_key("from .. import something", config)
    assert result.startswith("B")
    
    # Test order_by_type=False lowercases the line
    config = create_config(order_by_type=False)
    result = section_key("import MyModule", config)
    assert "mymodule" in result.lower()
    
    # Test multiple dots in relative import
    config = create_config(reverse_relative=True, sort_relative_in_force_sorted_sections=False)
    result = section_key("from ... import something", config)
    assert result.startswith("B")
    
    # Test empty force_to_top
    config = create_config(force_to_top=[])
    result = section_key("import sys", config)
    assert result.startswith("B")
    
    # Test with both case_sensitive=False and order_by_type=False
    config = create_config(case_sensitive=False, order_by_type=False)
    result = section_key("from OS import PATH", config)
    assert result.startswith("B")


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test lexicographical mode
    config.force_to_top = []
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "." in result  # lexicographical replaces " import " with "."
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    result = section_key("from os import path", config)
    assert "import" not in result
    
    # Test relative imports with reverse_relative
    config.group_by_package = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import something", config)
    assert "from" in result or "." in result
    
    # Test sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from .. import something", config)
    assert ".." in result
    
    # Test case_sensitive and order_by_type interaction
    config.sort_relative_in_force_sorted_sections = False
    config.case_sensitive = False
    config.order_by_type = True
    config.honor_case_in_force_sorted_sections = True
    result = section_key("from MyModule import MyClass", config)
    assert "mymodule" in result.lower() or result is not None
    
    # Test length_sort
    config.case_sensitive = True
    config.order_by_type = False
    config.honor_case_in_force_sorted_sections = False
    config.length_sort = True
    result = section_key("import os", config)
    assert "2:" in result  # "os" has length 2
    
    # Test import statement
    config.length_sort = False
    result = section_key("import sys", config)
    assert "sys" in result
    assert result.startswith("B")
    
    # Test from...import statement
    result = section_key("from collections import defaultdict", config)
    assert "collections" in result
    
    # Test multiple dots in relative imports
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from ... import utils", config)
    assert "..." in result or result is not None
    
    # Test honor_case_in_force_sorted_sections with split
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    result = section_key("from MyModule import MyClass", config)
    assert result is not None
    
    # Test order_by_type False with case conversion
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = False
    config.case_sensitive = True
    result = section_key("from Module import Class", config)
    assert result.lower() == result or "module" in result.lower()


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import MagicMock


def test_module_key():
    # Test basic module name without special handling
    config = MagicMock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []

    result = module_key("os", config)
    assert result == "Bos"

    # Test with case_sensitive = False
    config.case_sensitive = False
    result = module_key("OS", config)
    assert result == "Bos"

    # Test with force_to_top
    config.force_to_top = ["sys"]
    result = module_key("sys", config)
    assert result == "Asys"

    # Test with different module
    result = module_key("os", config)
    assert result == "Bos"

    # Test relative imports with dots
    config.force_to_top = []
    config.reverse_relative = True
    result = module_key("...module", config)
    assert "module" in result

    # Test relative imports without reverse_relative
    config.reverse_relative = False
    result = module_key("..module", config)
    assert "module" in result

    # Test with order_by_type and sub_imports
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result

    # Test with classes in order_by_type
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result

    # Test with variables in order_by_type
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result

    # Test uppercase module name without being in constants
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("UPPER", config, sub_imports=True)
    assert "A" in result

    # Test with ignore_case parameter
    config.case_sensitive = True
    result = module_key("MyModule", config, ignore_case=True)
    assert "mymodule" in result

    # Test with length_sort enabled
    config.case_sensitive = False
    config.length_sort = True
    result = module_key("test", config)
    assert "4:test" in result

    # Test with length_sort_straight enabled
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result

    # Test with section in length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["thirdparty"]
    result = module_key("test", config, section_name="thirdparty")
    assert "4:test" in result

    # Test with multiple dots in relative import
    config.length_sort = False
    config.reverse_relative = False
    result = module_key("....package.module", config)
    assert "package" in result or "module" in result

    # Test force_to_top with relative imports
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result.startswith("A")

    # Test combination of parameters
    config.force_to_top = []
    config.order_by_type = True
    config.case_sensitive = False
    config.length_sort = True
    config.classes = ["Class"]
    result = module_key("Class", config, sub_imports=True)
    assert "B" in result
    assert "5:class" in result


# LLM-generated content at query #25
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Create a mock config object with default values
    def create_config(**kwargs):
        config = Mock()
        config.sort_relative_in_force_sorted_sections = kwargs.get('sort_relative_in_force_sorted_sections', False)
        config.reverse_relative = kwargs.get('reverse_relative', False)
        config.group_by_package = kwargs.get('group_by_package', False)
        config.lexicographical = kwargs.get('lexicographical', False)
        config.force_to_top = kwargs.get('force_to_top', [])
        config.honor_case_in_force_sorted_sections = kwargs.get('honor_case_in_force_sorted_sections', False)
        config.case_sensitive = kwargs.get('case_sensitive', True)
        config.order_by_type = kwargs.get('order_by_type', False)
        config.length_sort = kwargs.get('length_sort', False)
        return config
    
    # Test basic import line
    config = create_config()
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import line
    config = create_config()
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config = create_config(force_to_top=["os"])
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test relative imports with reverse_relative
    config = create_config(reverse_relative=True, sort_relative_in_force_sorted_sections=False)
    result = section_key("from . import module", config)
    assert "module" in result
    
    # Test relative imports with sort_relative_in_force_sorted_sections
    config = create_config(reverse_relative=True, sort_relative_in_force_sorted_sections=True)
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test group_by_package
    config = create_config(group_by_package=True)
    result = section_key("from os import path, sep", config)
    assert "os" in result
    
    # Test lexicographical
    config = create_config(lexicographical=True)
    result = section_key("from os import path", config)
    assert "os.path" in result
    
    # Test case sensitivity with order_by_type
    config = create_config(case_sensitive=False, order_by_type=False)
    result = section_key("import Os", config)
    assert "os" in result.lower()
    
    # Test honor_case_in_force_sorted_sections
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True
    )
    result = section_key("from Os import Path", config)
    assert result.startswith("B")
    
    # Test length_sort
    config = create_config(length_sort=True)
    result = section_key("import os", config)
    assert "2:" in result  # length of "os" is 2
    
    # Test multiple relative dots
    config = create_config(reverse_relative=False, sort_relative_in_force_sorted_sections=False)
    result = section_key("from ... import module", config)
    assert result.startswith("B")
    
    # Test empty force_to_top with various inputs
    config = create_config(force_to_top=[])
    result1 = section_key("import a", config)
    result2 = section_key("import z", config)
    assert result1.startswith("B")
    assert result2.startswith("B")
    
    # Test case insensitive without order_by_type
    config = create_config(case_sensitive=True, order_by_type=False)
    result = section_key("import MyModule", config)
    assert "mymodule" in result.lower()


# LLM-generated content at query #26
#--------------------------

```python
def test_section_key():
    """Test the section_key function with various configurations and inputs."""
    from unittest.mock import Mock
    
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test with reverse_relative and relative imports
    config.force_to_top = []
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import something", config)
    assert result.startswith("B")
    
    # Test with lexicographical sorting
    config.lexicographical = True
    config.reverse_relative = False
    result = section_key("from os import path", config)
    assert "." in result  # lexicographical replaces " import " with "."
    
    # Test with group_by_package
    config.lexicographical = False
    config.group_by_package = True
    result = section_key("from os import path, environ", config)
    assert "import" not in result  # group_by_package strips everything after import
    
    # Test with length_sort
    config.group_by_package = False
    config.length_sort = True
    result = section_key("import os", config)
    assert result.startswith("B2")  # length of "os" is 2
    
    # Test case_sensitive and order_by_type mismatch
    config.length_sort = False
    config.case_sensitive = False
    config.order_by_type = True
    config.honor_case_in_force_sorted_sections = True
    result = section_key("from MyModule import MyClass", config)
    assert "mymodule" in result.lower()
    
    # Test order_by_type = False converts to lowercase
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("import OS", config)
    assert "os" in result
    
    # Test sort_relative_in_force_sorted_sections with relative imports
    config.order_by_type = True
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from ... import something", config)
    assert "_" in result  # underscore separator for non-reversed relative
    
    config.reverse_relative = True
    result = section_key("from ... import something", config)
    assert " " in result  # space separator for reversed relative
    
    # Test from statement without import
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    result = section_key("from os", config)
    assert "os" in result
    
    # Test multiple relative dots
    result = section_key("from .... import module", config)
    assert result.startswith("B")
    
    # Test with honor_case_in_force_sorted_sections and split module
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from Module import CONSTANT", config)
    assert "Module" in result  # module_name keeps case
    assert "constant" in result  # names are lowercased


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations."""
    
    # Create mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with case insensitive
    result = module_key("Os", config, ignore_case=True)
    assert result == "Bos"
    
    # Test relative imports with reverse_relative=False
    result = module_key("..utils", config)
    assert result == "B_ utils"
    
    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    result = module_key("..utils", config)
    assert result == "B_utils"
    
    # Test with case_sensitive=False
    config.reverse_relative = False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    
    # Test with force_to_top
    config.case_sensitive = True
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    
    # Test with length_sort
    config.force_to_top = []
    config.length_sort = True
    result = module_key("abc", config)
    assert "3:abc" in result
    
    # Test with order_by_type and constants
    config.length_sort = False
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    
    # Test with order_by_type and classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    # Test with order_by_type and variables
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result
    
    # Test uppercase constant detection
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    
    # Test uppercase class detection
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    # Test lowercase variable detection
    result = module_key("my_variable", config, sub_imports=True)
    assert "C" in result
    
    # Test length_sort_straight
    config.order_by_type = False
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    
    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["thirdparty"]
    result = module_key("test", config, section_name="thirdparty")
    assert "5:test" in result
    
    # Test with multiple dots in relative import
    config.reverse_relative = False
    config.length_sort_sections = []
    result = module_key("...package.module", config)
    assert "_" in result or "package" in result
    
    # Test empty module name
    result = module_key("", config)
    assert result == "B"
    
    # Test single character module
    result = module_key("a", config)
    assert result == "Ba"
    
    # Test with numeric module name
    result = module_key("123abc", config)
    assert "123abc" in result


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    """Test section_key function with various configurations"""
    
    # Mock config with default values
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    
    # Test basic import statement
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import statement
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top section
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    config.force_to_top = []
    
    # Test with length_sort enabled
    config.length_sort = True
    result = section_key("import os", config)
    assert "2" in result  # length of "os"
    
    config.length_sort = False
    
    # Test case sensitivity
    config.case_sensitive = False
    config.order_by_type = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    config.case_sensitive = True
    
    # Test lexicographical sorting
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "." in result  # Should replace " import " with "."
    
    config.lexicographical = False
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test relative imports with spaces
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from ... import module", config)
    assert result.startswith("B")
    
    # Test group_by_package
    config.group_by_package = True
    result = section_key("from os import path, sep", config)
    assert result.startswith("B")
    
    config.group_by_package = False
    
    # Test honor_case_in_force_sorted_sections with different case_sensitive and order_by_type
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from os import Path, CONSTANT", config)
    assert result.startswith("B")
    
    # Test with multiple spaces in import line
    config.honor_case_in_force_sorted_sections = False
    result = section_key("import   os", config)
    assert result.startswith("B")
    
    # Test empty force_to_top list doesn't affect section
    config.force_to_top = []
    result = section_key("import sys", config)
    assert result.startswith("B")
    
    # Test that section is "A" when module in force_to_top
    config.force_to_top = ["sys"]
    result = section_key("import sys", config)
    assert result.startswith("A")
    
    # Test module not in force_to_top returns "B"
    config.force_to_top = ["sys"]
    result = section_key("import os", config)
    assert result.startswith("B")


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    """Test the section_key function with various configurations."""
    
    # Mock config object with default values
    def create_config(**kwargs):
        config = Mock()
        config.sort_relative_in_force_sorted_sections = kwargs.get('sort_relative_in_force_sorted_sections', False)
        config.reverse_relative = kwargs.get('reverse_relative', False)
        config.group_by_package = kwargs.get('group_by_package', False)
        config.lexicographical = kwargs.get('lexicographical', False)
        config.force_to_top = kwargs.get('force_to_top', [])
        config.honor_case_in_force_sorted_sections = kwargs.get('honor_case_in_force_sorted_sections', False)
        config.case_sensitive = kwargs.get('case_sensitive', True)
        config.order_by_type = kwargs.get('order_by_type', False)
        config.length_sort = kwargs.get('length_sort', False)
        return config
    
    # Test basic import line
    config = create_config()
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import line
    config = create_config()
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config = create_config(force_to_top=["os"])
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test reverse_relative with relative import
    config = create_config(reverse_relative=True, sort_relative_in_force_sorted_sections=False)
    result = section_key("from . import something", config)
    assert result.startswith("B")
    
    # Test sort_relative_in_force_sorted_sections with reverse_relative
    config = create_config(sort_relative_in_force_sorted_sections=True, reverse_relative=True)
    result = section_key("from . import something", config)
    assert "." in result
    
    # Test sort_relative_in_force_sorted_sections without reverse_relative
    config = create_config(sort_relative_in_force_sorted_sections=True, reverse_relative=False)
    result = section_key("from . import something", config)
    assert "." in result
    
    # Test group_by_package
    config = create_config(group_by_package=True)
    result = section_key("from os import path, sys", config)
    assert "os" in result
    
    # Test lexicographical
    config = create_config(lexicographical=True)
    result = section_key("from os import path", config)
    assert "." in result  # "import" is replaced with "."
    
    # Test case_sensitive with order_by_type difference
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True
    )
    result = section_key("from OS import Path", config)
    assert result.startswith("B")
    
    # Test order_by_type false with lowercase conversion
    config = create_config(order_by_type=False)
    result = section_key("from OS import PATH", config)
    assert "os" in result.lower()
    
    # Test length_sort
    config = create_config(length_sort=True)
    result = section_key("import os", config)
    assert "2" in result  # length of "os"
    
    # Test multiple relative dots
    config = create_config(reverse_relative=True, sort_relative_in_force_sorted_sections=False)
    result = section_key("from .. import something", config)
    assert result.startswith("B")
    
    # Test complex case: force_to_top + case_sensitive
    config = create_config(
        force_to_top=["django"],
        case_sensitive=True,
        order_by_type=False
    )
    result = section_key("import django", config)
    assert result.startswith("A")
    
    # Test honor_case_in_force_sorted_sections with split
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True
    )
    result = section_key("from MyModule import MyClass", config)
    assert "mymodule" in result.lower()
    
    # Test empty force_to_top
    config = create_config(force_to_top=[])
    result = section_key("import sys", config)
    assert result.startswith("B")
    
    # Test relative import with sort_relative_in_force_sorted_sections
    config = create_config(
        sort_relative_in_force_sorted_sections=True,
        reverse_relative=False
    )
    result = section_key("from ...package import module", config)
    assert "." in result


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    # Test basic module name without relative imports
    config = Mock()
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
    
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with ignore_case
    result = module_key("Os", config, ignore_case=True)
    assert result == "Bos"
    
    # Test with case_sensitive False
    config.case_sensitive = False
    result = module_key("Os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = []
    
    # Test with relative imports
    config.reverse_relative = False
    result = module_key("...module", config)
    assert "module" in result
    
    # Test with relative imports and reverse_relative
    config.reverse_relative = True
    result = module_key("..module", config)
    assert "module" in result
    
    # Test with length_sort
    config.length_sort = True
    config.case_sensitive = True
    result = module_key("abc", config)
    assert "3:abc" in result
    
    # Test with order_by_type and constants
    config.length_sort = False
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result.startswith("BA")
    
    # Test with order_by_type and classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert result.startswith("BB")
    
    # Test with order_by_type and variables
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert result.startswith("BC")
    
    # Test with uppercase module name (issue #376)
    config.order_by_type = True
    result = module_key("UPPER", config, sub_imports=True)
    assert result.startswith("BA")
    
    # Test with capitalized module name
    config.classes = []
    result = module_key("MyModule", config, sub_imports=True)
    assert result.startswith("BB")
    
    # Test with lowercase module name (default to C)
    result = module_key("mymodule", config, sub_imports=True)
    assert result.startswith("BC")
    
    # Test with length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    
    # Test with length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["thirdparty"]
    result = module_key("module", config, section_name="thirdparty")
    assert "6:module" in result
    
    # Test combined: relative import with multiple dots
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_sections = []
    result = module_key("....package.module", config)
    assert "package" in result or "module" in result
    
    # Test with all features combined
    config.reverse_relative = True
    config.order_by_type = True
    config.case_sensitive = False
    config.length_sort = True
    config.constants = ["TEST"]
    config.force_to_top = ["os"]
    result = module_key("os", config, sub_imports=True, ignore_case=False)
    assert result.startswith("A")
    assert "2:os" in result


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["requests"]
    result = section_key("import requests", config)
    assert result.startswith("A")
    
    # Test relative imports with reverse_relative
    config.force_to_top = []
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import module", config)
    assert "B" in result
    
    # Test lexicographical sorting
    config.lexicographical = True
    config.reverse_relative = False
    result = section_key("from module import name", config)
    assert "." in result or "module" in result.lower()
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    result = section_key("from package import something", config)
    assert "package" in result.lower()
    
    # Test case_sensitive False with order_by_type True
    config.group_by_package = False
    config.case_sensitive = False
    config.order_by_type = True
    config.honor_case_in_force_sorted_sections = False
    result = section_key("import MyModule", config)
    assert "mymodule" in result.lower()
    
    # Test order_by_type False
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("import MyModule", config)
    assert "mymodule" in result.lower()
    
    # Test length_sort
    config.order_by_type = True
    config.length_sort = True
    result = section_key("import os", config)
    assert "2:" in result  # length of "os"
    
    # Test honor_case_in_force_sorted_sections with different case_sensitive and order_by_type
    config.length_sort = False
    config.case_sensitive = True
    config.order_by_type = False
    config.honor_case_in_force_sorted_sections = True
    result = section_key("from Module import Name", config)
    assert "name" in result.lower()
    
    # Test relative imports with sort_relative_in_force_sorted_sections
    config.honor_case_in_force_sorted_sections = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from ... import something", config)
    assert "." in result
    
    # Test relative imports with reverse_relative and sort_relative_in_force_sorted_sections
    config.reverse_relative = True
    result = section_key("from .. import module", config)
    assert "B" in result
    
    # Test empty import line
    result = section_key("", config)
    assert result.startswith("B")
    
    # Test from import with multiple spaces
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = True
    result = section_key("from os import path", config)
    assert "os" in result.lower()
    
    # Test force_to_top with multiple modules
    config.force_to_top = ["__future__", "os"]
    result = section_key("import __future__", config)
    assert result.startswith("A")
    
    result = section_key("import os", config)
    assert result.startswith("A")
    
    result = section_key("import sys", config)
    assert result.startswith("B")


# LLM-generated content at query #32
#--------------------------

```python
def test_section_key():
    """Test the section_key function with various configurations."""
    from unittest.mock import Mock
    
    # Create a mock config with default values
    def create_config(**kwargs):
        config = Mock()
        config.sort_relative_in_force_sorted_sections = kwargs.get('sort_relative_in_force_sorted_sections', False)
        config.reverse_relative = kwargs.get('reverse_relative', False)
        config.group_by_package = kwargs.get('group_by_package', False)
        config.lexicographical = kwargs.get('lexicographical', False)
        config.force_to_top = kwargs.get('force_to_top', [])
        config.honor_case_in_force_sorted_sections = kwargs.get('honor_case_in_force_sorted_sections', False)
        config.case_sensitive = kwargs.get('case_sensitive', True)
        config.order_by_type = kwargs.get('order_by_type', False)
        config.length_sort = kwargs.get('length_sort', False)
        return config
    
    # Test basic import statement
    config = create_config()
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import statement
    config = create_config()
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config = create_config(force_to_top=["os"])
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test relative imports with reverse_relative
    config = create_config(reverse_relative=True, sort_relative_in_force_sorted_sections=False)
    result = section_key("from . import something", config)
    assert "something" in result or "." in result
    
    # Test relative imports with sort_relative_in_force_sorted_sections
    config = create_config(reverse_relative=False, sort_relative_in_force_sorted_sections=True)
    result = section_key("from . import something", config)
    assert result.startswith("B")
    
    # Test group_by_package
    config = create_config(group_by_package=True)
    result = section_key("from os import path, system", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test lexicographical sorting
    config = create_config(lexicographical=True)
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "." in result
    
    # Test case sensitivity with order_by_type mismatch
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=True,
        order_by_type=False
    )
    result = section_key("from os import MyClass", config)
    assert result.startswith("B")
    
    # Test case insensitive
    config = create_config(case_sensitive=False, order_by_type=False)
    result = section_key("from OS import Path", config)
    assert "os" in result.lower()
    
    # Test length_sort
    config = create_config(length_sort=True)
    result = section_key("import os", config)
    assert result.startswith("B")
    
    # Test multiple force_to_top entries
    config = create_config(force_to_top=["os", "sys", "json"])
    result = section_key("import sys", config)
    assert result.startswith("A")
    
    # Test non-force_to_top with multiple entries
    config = create_config(force_to_top=["os", "sys"])
    result = section_key("import json", config)
    assert result.startswith("B")
    
    # Test relative import with reverse_relative and sort_relative_in_force_sorted_sections
    config = create_config(
        reverse_relative=True,
        sort_relative_in_force_sorted_sections=True
    )
    result = section_key("from .. import module", config)
    assert result.startswith("B")
    
    # Test honor_case_in_force_sorted_sections with split module
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True
    )
    result = section_key("from os import path", config)
    assert result.startswith("B")
    
    # Test empty force_to_top list
    config = create_config(force_to_top=[])
    result = section_key("import anything", config)
    assert result.startswith("B")


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations"""
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = []
    
    # Test relative imports
    config.reverse_relative = False
    result = module_key("...module", config)
    assert "_" in result
    
    config.reverse_relative = True
    result = module_key("...module", config)
    assert " " in result
    
    # Test case sensitivity
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert "mymodule" in result
    
    config.case_sensitive = True
    result = module_key("MyModule", config)
    assert "MyModule" in result
    
    # Test ignore_case parameter
    result = module_key("MyModule", config, ignore_case=True)
    assert "mymodule" in result
    
    # Test length_sort
    config.length_sort = True
    result = module_key("os", config)
    assert "2:" in result
    
    config.length_sort = False
    
    # Test length_sort_straight
    config.length_sort_straight = True
    result = module_key("os", config, straight_import=True)
    assert "2:" in result
    
    config.length_sort_straight = False
    
    # Test length_sort_sections
    config.length_sort_sections = ["stdlib"]
    result = module_key("os", config, section_name="stdlib")
    assert "2:" in result
    
    config.length_sort_sections = []
    
    # Test order_by_type with constants
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    
    # Test order_by_type with classes
    config.constants = []
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    # Test order_by_type with variables
    config.classes = []
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result
    
    # Test uppercase detection (issue #376)
    config.variables = []
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert "A" in result
    
    # Test uppercase with single character
    result = module_key("A", config, sub_imports=True)
    assert "C" in result
    
    # Test class-like detection
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    config.order_by_type = False
    
    # Test with relative imports and reverse_relative
    config.reverse_relative = False
    result = module_key("..utils", config)
    assert "utils" in result
    
    # Test empty module name handling
    result = module_key("", config)
    assert isinstance(result, str)
    
    # Test complex relative import
    config.reverse_relative = False
    result = module_key("....package.module", config)
    assert isinstance(result, str)
    
    # Test sub_imports without order_by_type
    result = module_key("module", config, sub_imports=True)
    assert isinstance(result, str)


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    """Test the section_key function with various configurations."""
    
    # Mock config object
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    
    # Test basic import line
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import line
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    config.force_to_top = []
    
    # Test with relative imports and reverse_relative
    config.reverse_relative = True
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test group_by_package
    config.group_by_package = True
    result = section_key("from os.path import join", config)
    assert "import" not in result
    config.group_by_package = False
    
    # Test lexicographical
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "." in result  # midline import replaced with dot
    config.lexicographical = False
    
    # Test case sensitivity
    config.case_sensitive = False
    config.order_by_type = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test length_sort
    config.length_sort = True
    result = section_key("import os", config)
    assert any(char.isdigit() for char in result)
    config.length_sort = False
    
    # Test sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from ... import module", config)
    assert "_" in result or "." in result
    
    # Test honor_case_in_force_sorted_sections with different case_sensitive and order_by_type
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from os import PATH", config)
    assert "path" in result.lower()
    
    # Test with complex import statement
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from package.submodule import Class", config)
    assert result.startswith("B")
    
    # Test empty force_to_top with various lines
    config.force_to_top = []
    result = section_key("import sys", config)
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test multiple dots in relative import
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from .. import module", config)
    assert result.startswith("B")


# LLM-generated content at query #3
#--------------------------

```python
def test_section_key():
    # Mock Config object
    class MockConfig:
        def __init__(self):
            self.sort_relative_in_force_sorted_sections = False
            self.reverse_relative = False
            self.group_by_package = False
            self.lexicographical = False
            self.force_to_top = []
            self.honor_case_in_force_sorted_sections = False
            self.case_sensitive = True
            self.order_by_type = False
            self.length_sort = False

    config = MockConfig()

    # Test basic import line
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result

    # Test from import line
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result

    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")

    # Test lexicographical sorting
    config.force_to_top = []
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "." in result

    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    result = section_key("from os import path, walk", config)
    assert "os" in result
    assert "import" not in result

    # Test relative imports with reverse_relative
    config.group_by_package = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import module", config)
    assert "." in result

    # Test relative imports with sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from . import module", config)
    assert "." in result

    # Test case sensitivity
    config.sort_relative_in_force_sorted_sections = False
    config.case_sensitive = False
    result = section_key("import OS", config)
    assert result.lower() == result

    # Test order_by_type with honor_case_in_force_sorted_sections
    config.case_sensitive = True
    config.order_by_type = False
    config.honor_case_in_force_sorted_sections = True
    result = section_key("from Module import Name", config)
    assert "import" in result

    # Test length_sort
    config.honor_case_in_force_sorted_sections = False
    config.length_sort = True
    result = section_key("import os", config)
    assert result.startswith("B2")

    # Test multiple relative dots
    config.length_sort = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from ... import module", config)
    assert "." in result

    # Test empty import line
    result = section_key("import", config)
    assert result.startswith("B")

    # Test with order_by_type True (should not lowercase)
    config.order_by_type = True
    result = section_key("import MyModule", config)
    assert "MyModule" in result or "mymodule" in result


# LLM-generated content at query #4
#--------------------------

```python
def test_section_key():
    """Test the section_key function with various configurations."""
    from unittest.mock import Mock
    
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["django"]
    result = section_key("import django", config)
    assert result.startswith("A")
    
    # Test lexicographical sorting
    config.force_to_top = []
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "." in result  # midline import should be replaced with dot
    
    # Test relative imports with reverse_relative
    config.lexicographical = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test group_by_package
    config.group_by_package = True
    config.reverse_relative = False
    result = section_key("from os import path, getcwd", config)
    assert "os" in result
    assert "import" not in result
    
    # Test length_sort
    config.group_by_package = False
    config.length_sort = True
    result = section_key("import os", config)
    assert "2" in result  # length of "os"
    
    # Test case_sensitive = False
    config.length_sort = False
    config.case_sensitive = False
    config.order_by_type = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test honor_case_in_force_sorted_sections with split
    config.case_sensitive = False
    config.order_by_type = True
    config.honor_case_in_force_sorted_sections = True
    result = section_key("from os import Path", config)
    assert result.startswith("B")
    
    # Test sort_relative_in_force_sorted_sections with reverse_relative
    config.honor_case_in_force_sorted_sections = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    config.case_sensitive = True
    config.order_by_type = True
    result = section_key("from ... import module", config)
    assert "." in result
    assert " " in result  # separator should be space when reverse_relative is True
    
    # Test sort_relative_in_force_sorted_sections without reverse_relative
    config.reverse_relative = False
    result = section_key("from .. import module", config)
    assert "." in result
    assert "_" in result  # separator should be underscore when reverse_relative is False
    
    # Test with order_by_type = False
    config.sort_relative_in_force_sorted_sections = False
    config.order_by_type = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test empty force_to_top list
    config.force_to_top = []
    result = section_key("import sys", config)
    assert result.startswith("B")
    
    # Test multiple word imports
    config.case_sensitive = True
    config.order_by_type = True
    config.honor_case_in_force_sorted_sections = False
    result = section_key("from os.path import join", config)
    assert "os" in result or "path" in result
    
    # Test from import with spaces
    result = section_key("from os import path", config)
    assert result.startswith("B")


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations"""
    
    # Create mock config
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    
    # Test ignore_case
    config.force_to_top = []
    result = module_key("MyModule", config, ignore_case=True)
    assert result == "Bmymodule"
    
    # Test case_sensitive False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    
    # Test relative imports with reverse_relative False
    config.reverse_relative = False
    config.case_sensitive = True
    result = module_key("...package", config)
    assert "_" in result
    
    # Test relative imports with reverse_relative True
    config.reverse_relative = True
    result = module_key("...package", config)
    assert " " in result
    
    # Test sub_imports with order_by_type
    config.order_by_type = True
    config.constants = ["CONST_MODULE"]
    result = module_key("CONST_MODULE", config, sub_imports=True)
    assert "A" in result
    
    # Test sub_imports with class in config.classes
    config.classes = ["ClassName"]
    result = module_key("ClassName", config, sub_imports=True)
    assert "B" in result
    
    # Test sub_imports with variable in config.variables
    config.variables = ["variable_name"]
    result = module_key("variable_name", config, sub_imports=True)
    assert "C" in result
    
    # Test uppercase module name (issue #376)
    config.order_by_type = True
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert "A" in result
    
    # Test capitalized module name
    result = module_key("Capitalized", config, sub_imports=True)
    assert "B" in result
    
    # Test lowercase module name
    result = module_key("lowercase", config, sub_imports=True)
    assert "C" in result
    
    # Test length_sort
    config.order_by_type = False
    config.length_sort = True
    result = module_key("ab", config)
    assert "2:ab" in result
    
    # Test length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    
    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["thirdparty"]
    result = module_key("abc", config, section_name="thirdparty")
    assert "3:abc" in result
    
    # Test without any length_sort options
    config.length_sort_sections = []
    result = module_key("simple", config)
    assert result == "Bsimple"
    
    # Test relative import with dots and package
    config.reverse_relative = False
    result = module_key("..utils", config)
    assert "utils" in result
    
    # Test empty relative prefix
    result = module_key(".module", config)
    assert "module" in result


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations and inputs."""
    
    # Create a mock config object with default values
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = []
    
    # Test with ignore_case
    result = module_key("MyModule", config, ignore_case=True)
    assert result == "Bmymodule"
    
    # Test with relative imports (single dot)
    config.reverse_relative = False
    result = module_key(".module", config)
    assert result == "B._module"
    
    # Test with relative imports reversed
    config.reverse_relative = True
    result = module_key(".module", config)
    assert result == "B. module"
    
    # Test with multiple relative dots
    result = module_key("..module", config)
    assert result == "B.._module"
    
    # Test case_sensitive = False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    
    # Test with sub_imports and order_by_type
    config.order_by_type = True
    config.case_sensitive = True
    config.constants = ["CONST"]
    result = module_key("CONST", config, sub_imports=True)
    assert result.startswith("BA")
    
    # Test with uppercase constant detection
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert result.startswith("BA")
    
    # Test with class detection
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert result.startswith("BB")
    
    # Test with variables
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert result.startswith("BC")
    
    # Test with length_sort enabled
    config.order_by_type = False
    config.length_sort = True
    result = module_key("test", config)
    assert "4:test" in result
    
    # Test with length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    
    # Test with length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["local"]
    result = module_key("test", config, section_name="local")
    assert "4:test" in result
    
    # Test combined: relative import + force_to_top
    config.reverse_relative = False
    config.force_to_top = ["module"]
    result = module_key(".module", config)
    assert result.startswith("AA")
    
    # Test empty module name
    config.force_to_top = []
    result = module_key("", config)
    assert result == "B"
    
    # Test with multiple dots and spaces
    config.reverse_relative = True
    result = module_key("...submodule", config)
    assert "..." in result
    
    # Test sub_imports with uppercase first letter (class-like)
    config.order_by_type = True
    result = module_key("MyVariable", config, sub_imports=True)
    assert result.startswith("BB")
    
    # Test sub_imports with lowercase (variable-like)
    result = module_key("my_variable", config, sub_imports=True)
    assert result.startswith("BC")


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import line
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test relative imports with reverse_relative
    config.force_to_top = []
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import foo", config)
    assert result.startswith("B")
    
    # Test lexicographical sorting
    config.lexicographical = True
    config.reverse_relative = False
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "." in result
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    result = section_key("from os import path, sys", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test case_sensitive False
    config.group_by_package = False
    config.case_sensitive = False
    config.order_by_type = False
    result = section_key("import OS", config)
    assert result.startswith("B")
    assert "os" in result.lower()
    
    # Test length_sort
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = True
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "2:" in result
    
    # Test sort_relative_in_force_sorted_sections
    config.length_sort = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from ... import foo", config)
    assert result.startswith("B")
    assert "." in result
    
    # Test honor_case_in_force_sorted_sections with different case_sensitive and order_by_type
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from os import PATH", config)
    assert result.startswith("B")
    
    # Test honor_case_in_force_sorted_sections with same case_sensitive and order_by_type
    config.case_sensitive = False
    config.order_by_type = False
    result = section_key("from OS import Path", config)
    assert result.startswith("B")
    assert "os" in result.lower()
    
    # Test complex relative import path
    config.honor_case_in_force_sorted_sections = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from .. import module", config)
    assert result.startswith("B")
    
    # Test force_to_top with multiple modules
    config.force_to_top = ["sys", "os"]
    result = section_key("import sys", config)
    assert result.startswith("A")
    
    result = section_key("import json", config)
    assert result.startswith("B")


# LLM-generated content at query #8
#--------------------------

```python
def test_module_key():
    """Unit tests for module_key function"""
    from unittest.mock import Mock
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with case_sensitive False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    
    # Test with force_to_top
    config.force_to_top = ["sys"]
    result = module_key("sys", config)
    assert result == "Asys"
    
    # Test relative imports with reverse_relative False
    config.force_to_top = []
    config.reverse_relative = False
    result = module_key("...module", config)
    assert "module" in result
    
    # Test relative imports with reverse_relative True
    config.reverse_relative = True
    result = module_key("...module", config)
    assert "module" in result
    
    # Test with length_sort enabled
    config.length_sort = True
    config.case_sensitive = True
    result = module_key("test", config)
    assert "4:" in result  # "test" has length 4
    
    # Test with length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("abc", config, straight_import=True)
    assert "3:" in result
    
    # Test with length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["third_party"]
    result = module_key("module", config, section_name="third_party")
    assert "6:" in result
    
    # Test order_by_type with constants
    config.length_sort_sections = []
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result and "CONSTANT" in result
    
    # Test order_by_type with classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result and "MyClass" in result
    
    # Test order_by_type with variables
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result and "my_var" in result
    
    # Test uppercase module name detection
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert "A" in result
    
    # Test ignore_case parameter
    config.case_sensitive = True
    result = module_key("TestModule", config, ignore_case=True)
    assert "testmodule" in result.lower()
    
    # Test sub_imports without order_by_type
    config.order_by_type = False
    result = module_key("module", config, sub_imports=True)
    assert result == "Bmodule"
    
    # Test combination of parameters
    config.case_sensitive = False
    config.force_to_top = ["test"]
    result = module_key("test", config)
    assert result == "Atest"
    
    # Test empty force_to_top with module not in list
    config.force_to_top = []
    result = module_key("other", config)
    assert result.startswith("B")


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations"""
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = []
    
    # Test with ignore_case
    result = module_key("MyModule", config, ignore_case=True)
    assert result == "Bmymodule"
    
    # Test with case_sensitive disabled
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    config.case_sensitive = True
    
    # Test relative imports with reverse_relative False
    result = module_key("...module", config)
    assert "module" in result
    
    # Test relative imports with reverse_relative True
    config.reverse_relative = True
    result = module_key("...module", config)
    assert "module" in result
    config.reverse_relative = False
    
    # Test sub_imports with order_by_type
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result
    
    # Test uppercase variable detection (issue #376)
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert "A" in result
    
    # Test class-like name detection
    result = module_key("ClassName", config, sub_imports=True)
    assert "B" in result
    
    # Test regular variable detection
    result = module_key("variable", config, sub_imports=True)
    assert "C" in result
    
    config.order_by_type = False
    
    # Test length_sort
    config.length_sort = True
    result = module_key("ab", config)
    assert "2:ab" in result
    
    result = module_key("abcde", config)
    assert "5:abcde" in result
    config.length_sort = False
    
    # Test length_sort_straight
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert "6:module" in result
    config.length_sort_straight = False
    
    # Test length_sort_sections
    config.length_sort_sections = ["thirdparty"]
    result = module_key("module", config, section_name="thirdparty")
    assert "7:module" in result
    config.length_sort_sections = []
    
    # Test empty module name
    result = module_key("", config)
    assert result == "B"
    
    # Test numeric module name
    result = module_key("123", config)
    assert "123" in result
    
    # Test special characters in module name
    result = module_key("module-name", config)
    assert "module-name" in result


# LLM-generated content at query #10
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Create mock config object
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    
    # Test basic import statement
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import statement
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top module
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    config.force_to_top = []
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from . import module", config)
    assert result.startswith("B")
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = False
    
    # Test lexicographical sorting
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "." in result
    config.lexicographical = False
    
    # Test group_by_package
    config.group_by_package = True
    result = section_key("from os import path, getcwd", config)
    assert result.startswith("B")
    assert "os" in result
    config.group_by_package = False
    
    # Test case sensitivity
    config.case_sensitive = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    config.case_sensitive = True
    
    # Test length_sort
    config.length_sort = True
    result = section_key("import os", config)
    assert "2:" in result or "3:" in result
    config.length_sort = False
    
    # Test order_by_type with case_sensitive difference
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from os import Path", config)
    assert result.startswith("B")
    config.honor_case_in_force_sorted_sections = False
    
    # Test with complex relative imports
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from ... import module", config)
    assert result.startswith("B")
    
    # Test empty force_to_top list
    config.force_to_top = []
    result = section_key("import sys", config)
    assert result.startswith("B")
    
    # Test multiple spaces in import
    result = section_key("from  os  import  path", config)
    assert result.startswith("B")


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations"""
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with relative imports
    config.reverse_relative = False
    result = module_key("..module", config)
    assert "module" in result
    
    # Test with reverse_relative
    config.reverse_relative = True
    result = module_key("..module", config)
    assert "module" in result
    
    # Test case insensitive
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert "mymodule" in result.lower()
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    config.case_sensitive = True
    result = module_key("os", config)
    assert result.startswith("A")
    
    # Test with order_by_type and constants
    config.order_by_type = True
    config.sub_imports = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    
    # Test with order_by_type and classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    # Test with order_by_type and variables
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result
    
    # Test with uppercase variable (issue #376)
    config.order_by_type = True
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result
    
    # Test with length_sort
    config.order_by_type = False
    config.length_sort = True
    config.force_to_top = []
    result = module_key("ab", config)
    assert "2:ab" in result
    
    # Test with length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    
    # Test with length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["stdlib"]
    result = module_key("sys", config, section_name="stdlib")
    assert "3:sys" in result
    
    # Test ignore_case parameter
    config.case_sensitive = True
    result = module_key("TestModule", config, ignore_case=True)
    assert "testmodule" in result.lower()
    
    # Test with both case_sensitive False and ignore_case True
    config.case_sensitive = False
    result = module_key("TestModule", config, ignore_case=True)
    assert "testmodule" in result.lower()
    
    # Test empty module name
    result = module_key("", config)
    assert isinstance(result, str)
    
    # Test numeric module name
    result = module_key("module2", config)
    assert "module2" in result
    
    # Test single letter relative import
    config.reverse_relative = False
    result = module_key(".module", config)
    assert "module" in result
    
    # Test multiple dots in relative import
    result = module_key("...module", config)
    assert "module" in result


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock


def test_module_key():
    """Test module_key function with various configurations and inputs."""
    
    # Create a mock config object
    config = MagicMock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    
    # Test with relative imports
    config.force_to_top = []
    config.reverse_relative = False
    result = module_key("...package.module", config)
    assert "package_module" in result
    
    # Test with reverse_relative
    config.reverse_relative = True
    result = module_key("...package.module", config)
    assert "package module" in result
    
    # Test case insensitive
    config.reverse_relative = False
    config.ignore_case = True
    result = module_key("MyModule", config, ignore_case=True)
    assert "mymodule" in result.lower()
    
    # Test case_sensitive = False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert "mymodule" in result.lower()
    
    # Test length_sort
    config.case_sensitive = True
    config.length_sort = True
    result = module_key("os", config)
    assert "2:os" in result
    
    # Test length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("os", config, straight_import=True)
    assert "2:os" in result
    
    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["future"]
    result = module_key("os", config, section_name="future")
    assert "2:os" in result
    
    # Test sub_imports with order_by_type
    config.length_sort_sections = []
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result.startswith("BA")
    
    # Test sub_imports with classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert result.startswith("BB")
    
    # Test sub_imports with variables
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert result.startswith("BC")
    
    # Test uppercase module (issue #376)
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("OS", config, sub_imports=True)
    assert "A" in result
    
    # Test uppercase first letter
    result = module_key("MyModule", config, sub_imports=True)
    assert "B" in result
    
    # Test lowercase module
    result = module_key("mymodule", config, sub_imports=True)
    assert "C" in result
    
    # Test empty force_to_top
    config.order_by_type = False
    config.force_to_top = []
    result = module_key("os", config)
    assert result.startswith("B")
    
    # Test force_to_top with module
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result.startswith("A")
    
    # Test combined: case_sensitive=False + length_sort
    config.force_to_top = []
    config.case_sensitive = False
    config.length_sort = True
    result = module_key("MyModule", config)
    assert "mymodule" in result.lower()
    assert "8:" in result


# LLM-generated content at query #13
#--------------------------

```python
def test_section_key():
    """Test section_key function with various configurations."""
    from unittest.mock import MagicMock
    
    # Test basic import line
    config = MagicMock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test relative imports with reverse_relative
    config.force_to_top = []
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import module", config)
    assert "B" in result
    
    # Test lexicographical sorting
    config.lexicographical = True
    config.reverse_relative = False
    result = section_key("from os import path", config)
    assert "os.path" in result
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    result = section_key("from os import path, sep", config)
    assert "os" in result
    assert "path" not in result
    
    # Test case_sensitive False
    config.group_by_package = False
    config.case_sensitive = False
    config.order_by_type = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test length_sort
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = True
    result = section_key("import os", config)
    assert len(result) > 0
    
    # Test honor_case_in_force_sorted_sections with from import
    config.length_sort = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    result = section_key("from OS import Path", config)
    assert "os" in result
    
    # Test relative imports with sort_relative_in_force_sorted_sections
    config.honor_case_in_force_sorted_sections = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from ... import module", config)
    assert "." in result
    
    # Test from import pattern
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    result = section_key("from collections import defaultdict", config)
    assert "collections" in result
    
    # Test multiple relative dots with reverse_relative and sort_relative
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from .. import utils", config)
    assert "." in result
    
    # Test edge case: empty force_to_top with from import
    config.sort_relative_in_force_sorted_sections = False
    config.force_to_top = []
    result = section_key("from typing import List", config)
    assert result.startswith("B")
    
    # Test order_by_type False case sensitivity
    config.order_by_type = False
    config.case_sensitive = True
    result = section_key("from Module import Class", config)
    assert "module" in result.lower()


# LLM-generated content at query #14
#--------------------------

```python
def test_section_key():
    """Test the section_key function with various configurations."""
    from unittest.mock import Mock
    
    # Test basic import statement
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top configuration
    config.force_to_top = ["django"]
    result = section_key("import django", config)
    assert result.startswith("A")
    
    # Test from import with lexicographical ordering
    config.force_to_top = []
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "." in result  # lexicographical replaces space with dot
    
    # Test reverse_relative with relative imports
    config.lexicographical = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import module", config)
    assert "." in result
    
    # Test group_by_package
    config.group_by_package = True
    config.reverse_relative = False
    result = section_key("from os import path, sep", config)
    assert "import" not in result
    
    # Test case insensitive with order_by_type False
    config.group_by_package = False
    config.order_by_type = False
    config.case_sensitive = False
    result = section_key("import Django", config)
    assert "django" in result.lower()
    
    # Test length_sort
    config.length_sort = True
    config.order_by_type = True
    result = section_key("import os", config)
    assert "2:" in result  # length prefix
    
    # Test honor_case_in_force_sorted_sections with split
    config.length_sort = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from os import Path", config)
    assert "path" in result.lower()
    
    # Test relative imports with sort_relative_in_force_sorted_sections
    config.honor_case_in_force_sorted_sections = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from .. import module", config)
    assert "_" in result
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    result = section_key("from .. import module", config)
    assert " " in result
    
    # Test with multiple dots
    result = section_key("from ... import module", config)
    assert "." in result
    
    # Test empty line handling
    result = section_key("", config)
    assert result.startswith("B")
    
    # Test case sensitivity combinations
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    result = section_key("from Os import Path", config)
    assert "os" in result.lower()
    
    # Test with force_to_top and multiple word match
    config.force_to_top = ["collections"]
    config.honor_case_in_force_sorted_sections = False
    result = section_key("import collections", config)
    assert result.startswith("A")


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    """Test section_key function with various configurations."""
    
    # Create a mock config object
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    
    # Test basic import statement
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import statement
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    config.force_to_top = []
    
    # Test case_sensitive = False
    config.case_sensitive = False
    config.order_by_type = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test lexicographical = True
    config.lexicographical = True
    config.case_sensitive = True
    result = section_key("from os import path", config)
    assert "." in result  # midline import replaced with dot
    config.lexicographical = False
    
    # Test reverse_relative with relative imports
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from . import module", config)
    assert " " in result or "_" in result
    
    # Test group_by_package
    config.sort_relative_in_force_sorted_sections = False
    config.group_by_package = True
    result = section_key("from os import path", config)
    assert "os" in result
    assert "path" not in result
    config.group_by_package = False
    
    # Test length_sort
    config.length_sort = True
    result = section_key("import os", config)
    assert len(result.split("os")[0]) > 1  # Should have length prefix
    config.length_sort = False
    
    # Test honor_case_in_force_sorted_sections with split
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from OS import Path", config)
    assert result.startswith("B")
    
    # Test honor_case_in_force_sorted_sections without split
    config.honor_case_in_force_sorted_sections = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test multiple relative dots
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from ... import module", config)
    assert result.startswith("B")
    
    # Test order_by_type = True with case_sensitive = True
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = True
    config.case_sensitive = True
    result = section_key("from os import Path", config)
    assert result.startswith("B")


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations and inputs."""
    
    # Create a mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.force_to_top = []
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = []
    
    # Test with case_sensitive=False
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    config.case_sensitive = True
    
    # Test with ignore_case=True parameter
    result = module_key("MyModule", config, ignore_case=True)
    assert result == "Bmymodule"
    
    # Test relative imports
    config.reverse_relative = False
    result = module_key(".module", config)
    assert "_module" in result
    
    result = module_key("..module", config)
    assert "__module" in result
    
    # Test with reverse_relative=True
    config.reverse_relative = True
    result = module_key(".module", config)
    assert " module" in result
    
    # Test with length_sort=True
    config.length_sort = True
    result = module_key("os", config)
    assert "2:os" in result
    config.length_sort = False
    
    # Test with length_sort_straight=True and straight_import=True
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    config.length_sort_straight = False
    
    # Test with order_by_type=True and sub_imports=True
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert "A" in result  # Should have prefix A for constants
    
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result  # Should have prefix B for classes
    
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result  # Should have prefix C for variables
    
    # Test with uppercase module name (issue #376)
    config.order_by_type = True
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert "A" in result  # Uppercase with length > 1 should be prefix A
    
    # Test with capitalized module name
    result = module_key("Capitalized", config, sub_imports=True)
    assert "B" in result  # Capitalized should be prefix B
    
    # Test with lowercase module name
    result = module_key("lowercase", config, sub_imports=True)
    assert "C" in result  # Lowercase should be prefix C
    
    # Test with section_name in length_sort_sections
    config.order_by_type = False
    config.length_sort = False
    config.length_sort_sections = ["future"]
    result = module_key("os", config, section_name="future")
    assert "2:os" in result
    
    # Test combination of multiple parameters
    config.case_sensitive = False
    config.force_to_top = ["sys"]
    result = module_key("sys", config, ignore_case=True)
    assert result.startswith("A")
    assert "sys" in result


# LLM-generated content at query #17
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Create a mock config object
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    # Test basic import statement
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import statement
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    config.force_to_top = []
    
    # Test lexicographical sorting
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "os.path" in result
    config.lexicographical = False
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from . import module", config)
    assert "." in result
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = False
    
    # Test group_by_package
    config.group_by_package = True
    result = section_key("from os import path, sep", config)
    assert "import" not in result
    config.group_by_package = False
    
    # Test case_sensitive and order_by_type interaction
    config.case_sensitive = False
    config.order_by_type = True
    config.honor_case_in_force_sorted_sections = True
    result = section_key("from OS import Path", config)
    assert "os" in result.lower()
    
    # Test order_by_type = False
    config.case_sensitive = True
    config.order_by_type = False
    config.honor_case_in_force_sorted_sections = False
    result = section_key("from OS import Path", config)
    assert "os import path" in result.lower()
    
    # Test length_sort
    config.order_by_type = True
    config.length_sort = True
    result = section_key("import os", config)
    assert "2:" in result  # length of "os" is 2
    config.length_sort = False
    
    # Test with complex relative import
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from ... import module", config)
    assert "." in result
    
    # Test empty force_to_top with various inputs
    config.force_to_top = []
    result = section_key("import sys", config)
    assert result.startswith("B")
    
    result = section_key("from collections import OrderedDict", config)
    assert result.startswith("B")


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    """Test section_key function with various configurations and inputs."""
    
    # Create a mock config object with default values
    def create_config(**kwargs):
        config = Mock()
        config.sort_relative_in_force_sorted_sections = kwargs.get('sort_relative_in_force_sorted_sections', False)
        config.reverse_relative = kwargs.get('reverse_relative', False)
        config.group_by_package = kwargs.get('group_by_package', False)
        config.lexicographical = kwargs.get('lexicographical', False)
        config.force_to_top = kwargs.get('force_to_top', [])
        config.honor_case_in_force_sorted_sections = kwargs.get('honor_case_in_force_sorted_sections', False)
        config.case_sensitive = kwargs.get('case_sensitive', True)
        config.order_by_type = kwargs.get('order_by_type', False)
        config.length_sort = kwargs.get('length_sort', False)
        return config
    
    # Test basic import statement
    config = create_config()
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import statement
    config = create_config()
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config = create_config(force_to_top=["os"])
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test relative imports with reverse_relative
    config = create_config(reverse_relative=True)
    result = section_key("from . import something", config)
    assert result is not None
    
    # Test relative imports with sort_relative_in_force_sorted_sections
    config = create_config(sort_relative_in_force_sorted_sections=True, reverse_relative=True)
    result = section_key("from . import something", config)
    assert result is not None
    
    # Test group_by_package
    config = create_config(group_by_package=True)
    result = section_key("from os import path", config)
    assert "os" in result
    
    # Test lexicographical sorting
    config = create_config(lexicographical=True)
    result = section_key("from os import path", config)
    assert result is not None
    
    # Test case_sensitive with order_by_type mismatch
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=True,
        order_by_type=False
    )
    result = section_key("from Os import Path", config)
    assert result is not None
    
    # Test case_sensitive disabled
    config = create_config(case_sensitive=False, order_by_type=True)
    result = section_key("from OS import PATH", config)
    assert "os" in result.lower() or "path" in result.lower()
    
    # Test length_sort
    config = create_config(length_sort=True)
    result = section_key("import os", config)
    assert len(result) > 1  # Should include length prefix
    
    # Test order_by_type false applies lowercase
    config = create_config(order_by_type=False)
    result = section_key("from OS import PATH", config)
    assert "os" in result or "path" in result or result.islower() or "B" in result
    
    # Test multiple spaces and complex imports
    config = create_config()
    result = section_key("from   os   import   path", config)
    assert result is not None
    
    # Test empty force_to_top list
    config = create_config(force_to_top=[])
    result = section_key("import sys", config)
    assert result.startswith("B")
    
    # Test relative import with multiple dots
    config = create_config(reverse_relative=False)
    result = section_key("from ... import something", config)
    assert result is not None


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["django"]
    result = section_key("import django", config)
    assert result.startswith("A")
    
    # Test relative imports with reverse_relative
    config.force_to_top = []
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import utils", config)
    assert "B" in result
    
    # Test lexicographical sorting
    config.lexicographical = True
    config.reverse_relative = False
    result = section_key("from os import path", config)
    assert "." in result
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    result = section_key("from os import path", config)
    assert "os" in result
    assert "import" not in result
    
    # Test case_sensitive = False
    config.group_by_package = False
    config.case_sensitive = False
    config.order_by_type = False
    result = section_key("import Django", config)
    assert "django" in result.lower()
    
    # Test length_sort
    config.case_sensitive = True
    config.length_sort = True
    result = section_key("import abc", config)
    assert "3" in result  # length of "abc"
    
    # Test sort_relative_in_force_sorted_sections with reverse_relative
    config.length_sort = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from ... import module", config)
    assert "_" in result or "." in result
    
    # Test honor_case_in_force_sorted_sections with split
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from Os import Path", config)
    assert "import" in result
    
    # Test relative import with dots
    config.honor_case_in_force_sorted_sections = False
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from ..module import func", config)
    assert "B" in result
    
    # Test from import with force_to_top
    config.force_to_top = ["module"]
    result = section_key("from module import something", config)
    assert result.startswith("A")
    
    # Test multiple spaces in relative imports
    config.force_to_top = []
    config.reverse_relative = True
    result = section_key("from .  .  module import func", config)
    assert "B" in result
    
    # Test case sensitivity with order_by_type
    config.case_sensitive = False
    config.order_by_type = True
    result = section_key("import MYMODULE", config)
    assert "mymodule" in result.lower()
    
    # Test empty force_to_top list
    config.force_to_top = []
    result = section_key("import sys", config)
    assert result.startswith("B")


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    """Test section_key function with various configurations."""
    
    # Mock config object
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    
    # Test basic import statement
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import statement
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top section
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    config.force_to_top = []
    
    # Test relative imports with sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test relative imports with reverse_relative
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = True
    result = section_key("from . import module", config)
    assert "module" in result
    
    # Test lexicographical sorting
    config.reverse_relative = False
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "os.path" in result
    config.lexicographical = False
    
    # Test group_by_package
    config.group_by_package = True
    result = section_key("from os import path, sep", config)
    assert "os" in result
    assert "path" not in result or "sep" not in result
    config.group_by_package = False
    
    # Test case sensitivity with order_by_type
    config.case_sensitive = False
    config.order_by_type = True
    config.honor_case_in_force_sorted_sections = True
    result = section_key("from OS import Path", config)
    assert "os" in result.lower()
    
    # Test length_sort
    config.length_sort = True
    config.case_sensitive = True
    config.order_by_type = False
    config.honor_case_in_force_sorted_sections = False
    result = section_key("import os", config)
    assert "2" in result  # length of "os"
    config.length_sort = False
    
    # Test case insensitive
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("import OS", config)
    assert "OS" in result
    
    result = section_key("import os", config)
    assert "os" in result
    
    # Test honor_case_in_force_sorted_sections with split
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    result = section_key("from OS import Path", config)
    assert result.startswith("B")
    
    # Test with empty force_to_top and various import styles
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    
    result = section_key("import sys", config)
    assert isinstance(result, str)
    assert result.startswith("B")
    
    result = section_key("from typing import List", config)
    assert isinstance(result, str)
    assert result.startswith("B")
    
    # Test relative imports with dots
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from .. import module", config)
    assert isinstance(result, str)
    assert result.startswith("B")


# LLM-generated content at query #21
#--------------------------

```python
def test_section_key():
    """Test the section_key function with various configurations."""
    from unittest.mock import Mock
    
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top section
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test relative imports with reverse_relative
    config.force_to_top = []
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from . import something", config)
    assert result.startswith("B")
    
    # Test lexicographical sorting
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "." in result  # lexicographical replaces " import " with "."
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    result = section_key("from os import path", config)
    assert "import" not in result.lower() or "path" not in result
    
    # Test case_sensitive False
    config.case_sensitive = False
    config.order_by_type = False
    config.group_by_package = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test length_sort
    config.length_sort = True
    config.case_sensitive = True
    config.order_by_type = True
    result = section_key("import os", config)
    assert result.startswith("B")
    assert any(char.isdigit() for char in result)  # length should be in result
    
    # Test sort_relative_in_force_sorted_sections with reverse_relative
    config.length_sort = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    result = section_key("from ... import something", config)
    assert result.startswith("B")
    
    # Test honor_case_in_force_sorted_sections
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    config.sort_relative_in_force_sorted_sections = False
    result = section_key("from module import NAME", config)
    assert result.startswith("B")
    
    # Test from import syntax
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = True
    result = section_key("from collections import defaultdict", config)
    assert result.startswith("B")
    
    # Test multiple dots in relative imports
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from .. import module", config)
    assert result.startswith("B")
    
    # Test empty force_to_top with various imports
    config.force_to_top = []
    result = section_key("import sys", config)
    assert result.startswith("B")
    assert "sys" in result


# LLM-generated content at query #22
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test basic import line
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test lexicographical sorting
    config.force_to_top = []
    config.lexicographical = True
    result = section_key("from os import path", config)
    assert "." in result
    
    # Test relative imports with reverse_relative
    config.lexicographical = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    result = section_key("from . import module", config)
    assert result.startswith("B")
    
    # Test group_by_package
    config.reverse_relative = False
    config.sort_relative_in_force_sorted_sections = False
    config.group_by_package = True
    result = section_key("from os import path, sep", config)
    assert "import" not in result
    
    # Test case_sensitive False with order_by_type True
    config.group_by_package = False
    config.case_sensitive = False
    config.order_by_type = True
    config.honor_case_in_force_sorted_sections = False
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test length_sort
    config.case_sensitive = True
    config.length_sort = True
    result = section_key("import os", config)
    assert "2:" in result  # length of "os" is 2
    
    # Test honor_case_in_force_sorted_sections with different case_sensitive and order_by_type
    config.length_sort = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    result = section_key("from os import Path", config)
    assert "path" in result.lower()
    
    # Test from import line
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    result = section_key("from os import path", config)
    assert "os" in result
    
    # Test multiple relative dots
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    result = section_key("from ... import module", config)
    assert result.startswith("B")
    
    # Test empty force_to_top with various line formats
    config.sort_relative_in_force_sorted_sections = False
    config.force_to_top = ["django"]
    result = section_key("import django", config)
    assert result.startswith("A")
    
    result = section_key("import requests", config)
    assert result.startswith("B")


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_section_key():
    """Test section_key function with various configurations and inputs."""
    
    # Create mock config with default values
    def create_config(**kwargs):
        config = Mock()
        config.sort_relative_in_force_sorted_sections = kwargs.get('sort_relative_in_force_sorted_sections', False)
        config.reverse_relative = kwargs.get('reverse_relative', False)
        config.group_by_package = kwargs.get('group_by_package', False)
        config.lexicographical = kwargs.get('lexicographical', False)
        config.force_to_top = kwargs.get('force_to_top', [])
        config.honor_case_in_force_sorted_sections = kwargs.get('honor_case_in_force_sorted_sections', False)
        config.case_sensitive = kwargs.get('case_sensitive', True)
        config.order_by_type = kwargs.get('order_by_type', False)
        config.length_sort = kwargs.get('length_sort', False)
        return config
    
    # Test basic import statement
    config = create_config()
    result = section_key("import os", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test from import statement
    config = create_config()
    result = section_key("from os import path", config)
    assert result.startswith("B")
    assert "os" in result
    
    # Test force_to_top
    config = create_config(force_to_top=["os"])
    result = section_key("import os", config)
    assert result.startswith("A")
    
    # Test reverse_relative with relative imports
    config = create_config(reverse_relative=True)
    result = section_key("from . import something", config)
    assert "." in result
    
    # Test reverse_relative with multiple dots
    config = create_config(reverse_relative=True)
    result = section_key("from .. import something", config)
    assert ".." in result
    
    # Test group_by_package
    config = create_config(group_by_package=True)
    result = section_key("from os import path, sep", config)
    assert "os" in result
    assert "import" not in result
    
    # Test lexicographical
    config = create_config(lexicographical=True)
    result = section_key("from os import path", config)
    assert "." in result  # lexicographical replaces " import " with "."
    
    # Test case_sensitive with order_by_type mismatch
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True
    )
    result = section_key("from Os import Path", config)
    assert "os" in result.lower()
    
    # Test sort_relative_in_force_sorted_sections
    config = create_config(
        sort_relative_in_force_sorted_sections=True,
        reverse_relative=True
    )
    result = section_key("from . import something", config)
    assert " " in result or "_" in result
    
    # Test length_sort
    config = create_config(length_sort=True)
    result = section_key("import os", config)
    assert len(result) > 0
    
    # Test case insensitive without order_by_type
    config = create_config(case_sensitive=False, order_by_type=False)
    result = section_key("import OS", config)
    assert "os" in result.lower()
    
    # Test from relative import with sort_relative_in_force_sorted_sections and reverse_relative
    config = create_config(
        sort_relative_in_force_sorted_sections=True,
        reverse_relative=False
    )
    result = section_key("from . import module", config)
    assert "_" in result
    
    # Test complex case with honor_case_in_force_sorted_sections
    config = create_config(
        honor_case_in_force_sorted_sections=True,
        case_sensitive=True,
        order_by_type=False
    )
    result = section_key("from Module import Class", config)
    assert "module" in result.lower()
    
    # Test multiple dots in relative import
    config = create_config(reverse_relative=True)
    result = section_key("from ... import something", config)
    assert "..." in result


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations."""
    
    # Mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with ignore_case
    result = module_key("Os", config, ignore_case=True)
    assert result == "Bos"
    
    # Test with case_sensitive disabled
    config.case_sensitive = False
    result = module_key("Os", config)
    assert result == "Bos"
    
    # Test relative imports with dots
    config.case_sensitive = True
    config.reverse_relative = False
    result = module_key("..module", config)
    assert "module" in result
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    result = module_key("..module", config)
    assert "module" in result
    
    # Test with force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result.startswith("A")
    
    # Test with length_sort enabled
    config.force_to_top = []
    config.length_sort = True
    result = module_key("test", config)
    assert "4:test" in result
    
    # Test with length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("test", config, straight_import=True)
    assert "4:test" in result
    
    # Test with length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["stdlib"]
    result = module_key("test", config, section_name="stdlib")
    assert "4:test" in result
    
    # Test sub_imports with order_by_type
    config.length_sort_sections = []
    config.order_by_type = True
    config.constants = ["CONSTANT"]
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result[1] == "A"
    
    # Test sub_imports with classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert result[1] == "B"
    
    # Test sub_imports with variables
    config.classes = []
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert result[1] == "C"
    
    # Test uppercase detection (issue #376)
    config.order_by_type = True
    config.constants = []
    config.variables = []
    result = module_key("UPPER", config, sub_imports=True)
    assert result[1] == "A"
    
    # Test class-like name detection
    config.order_by_type = True
    result = module_key("ClassName", config, sub_imports=True)
    assert result[1] == "B"
    
    # Test lowercase variable detection
    result = module_key("variable", config, sub_imports=True)
    assert result[1] == "C"
    
    # Test case_sensitive disabled with module_name
    config.case_sensitive = False
    config.order_by_type = False
    result = module_key("MyModule", config)
    assert "mymodule" in result.lower()


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import Mock


def test_module_key():
    """Test module_key function with various configurations."""
    
    # Mock config object
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = []
    config.constants = []
    config.classes = []
    config.variables = []
    
    # Test basic module name
    result = module_key("os", config)
    assert result == "Bos"
    
    # Test with ignore_case
    result = module_key("OS", config, ignore_case=True)
    assert result == "Bos"
    
    # Test relative imports with dots
    config.reverse_relative = False
    result = module_key("..module", config)
    assert "module" in result
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    result = module_key("..module", config)
    assert "module" in result
    
    # Test force_to_top
    config.force_to_top = ["os"]
    result = module_key("os", config)
    assert result.startswith("A")
    
    # Test case_sensitive False
    config.force_to_top = []
    config.case_sensitive = False
    result = module_key("MyModule", config)
    assert result == "Bmymodule"
    
    # Test sub_imports with order_by_type
    config.case_sensitive = True
    config.order_by_type = True
    config.constants = ["CONST"]
    result = module_key("CONST", config, sub_imports=True)
    assert "A" in result
    
    # Test sub_imports with classes
    config.classes = ["MyClass"]
    result = module_key("MyClass", config, sub_imports=True)
    assert "B" in result
    
    # Test sub_imports with variables
    config.variables = ["my_var"]
    result = module_key("my_var", config, sub_imports=True)
    assert "C" in result
    
    # Test uppercase variable detection
    config.constants = []
    config.classes = []
    config.variables = []
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert "A" in result
    
    # Test length_sort
    config.order_by_type = False
    config.length_sort = True
    result = module_key("module", config)
    assert "6:module" in result
    
    # Test length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert "6:module" in result
    
    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["stdlib"]
    result = module_key("module", config, section_name="stdlib")
    assert "6:module" in result
    
    # Test with force_to_top and prefix
    config.force_to_top = ["os"]
    config.order_by_type = True
    config.classes = ["os"]
    result = module_key("os", config, sub_imports=True)
    assert result.startswith("A")
    
    # Test empty module name
    config.force_to_top = []
    config.order_by_type = False
    result = module_key("", config)
    assert result == "B"
    
    # Test relative import with space preservation
    config.reverse_relative = False
    result = module_key(". module", config)
    assert "module" in result


