####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure it doesn't have decorator_list
    assert get_first_line_number(node_no_decorators) == 10

    # Test Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # In this case, the function should return the first decorator's lineno
    assert get_first_line_number(node_with_decorators) == 5

    # Test Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Test Case 4: Node where decorator_list is missing (getattr fallback)
    node_missing_attr = MagicMock()
    node_missing_attr.lineno = 30
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Case 1: Node has no decorators, should return node.lineno
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute is missing
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators, should return lineno of the first decorator
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    
    node_with_decorators = Mock()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has empty decorator_list, should return node.lineno
    node_empty_decorators = Mock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node with no decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist for test
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with an empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 30
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node where decorator_list is missing (testing getattr default)
    node_missing_attr = MagicMock(spec=['lineno'])
    node_missing_attr.lineno = 40
    # Explicitly remove attribute if it was added by MagicMock during setup
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 40
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with empty decorator_list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 3: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10  # The node itself starts later
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 4: Node with a single decorator
    node_single_decorator = MagicMock()
    decorator = MagicMock()
    decorator.lineno = 15
    node_single_decorator.decorator_list = [decorator]
    node_single_decorator.lineno = 20
    assert get_first_line_number(node_single_decorator) == 15
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list # Ensure attribute doesn't exist
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator_list (should return node.lineno)
    node_empty_decorator_list = MagicMock()
    node_empty_decorator_list.decorator_list = []
    node_empty_decorator_list.lineno = 15
    assert get_first_line_number(node_empty_decorator_list) == 15

    # Test case 4: Node where decorator_list is present but contains one item
    node_single_decorator = MagicMock()
    single_decorator = MagicMock()
    single_decorator.lineno = 2
    node_single_decorator.decorator_list = [single_decorator]
    node_single_decorator.lineno = 10
    assert get_first_line_number(node_single_decorator) == 2
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator_list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node where decorator_list attribute is missing (should return node.lineno)
    node_missing_attr = MagicMock()
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    node_missing_attr.lineno = 30
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators
    node_with_decorators = Mock()
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = Mock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node has no decorators, should return node.lineno
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators, should return lineno of the first decorator
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator list, should return node.lineno
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 4: Node where decorator_list is present but contains one item
    single_decorator_node = MagicMock()
    single_decorator = MagicMock()
    single_decorator.lineno = 1
    single_decorator_node.decorator_list = [single_decorator]
    assert get_first_line_number(single_decorator_node) == 1
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (uses node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (uses first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # Even if node.lineno is different, it should return the first decorator's lineno
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 10
    node_with_decorators.decorator_list = [decorator1, decorator2]
    
    # Should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node has no decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators, should return the lineno of the first decorator
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has empty decorator_list attribute
    node_empty_list = MagicMock()
    node_empty_list.decorator_list = []
    node_empty_list.lineno = 20
    assert get_first_line_number(node_empty_list) == 20
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators
    node_with_decorators = Mock()
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # The function should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = Mock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 4: Node where decorator_list is missing (handled by getattr default)
    node_missing_attr = Mock()
    node_missing_attr.lineno = 30
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute is missing to test getattr default
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with empty decorator list
    node_empty_decorators = Mock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return lineno of the first decorator)
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    
    node_with_decorators = Mock()
    node_with_decorators.lineno = 15
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node where lineno is different from decorators
    node_single_decorator = Mock()
    node_single_decorator.lineno = 30
    decorator3 = Mock()
    decorator3.lineno = 25
    node_single_decorator.decorator_list = [decorator3]
    assert get_first_line_number(node_single_decorator) == 25
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node with no decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist to test getattr default
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorator_list present but empty
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return lineno of the first decorator)
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 10
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = Mock()
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node where decorator_list is an empty list
    node_empty_decorators = Mock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node with only one decorator
    node_single_decorator = Mock()
    decorator3 = Mock()
    decorator3.lineno = 15
    node_single_decorator.decorator_list = [decorator3]
    assert get_first_line_number(node_single_decorator) == 15
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with an empty decorator list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 3: Node with decorators (should return the lineno of the first decorator)
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    
    assert get_first_line_number(node_with_decorators) == 5

    # Case 2: Node without decorators (should return node's lineno)
    node_without_decorators = MagicMock()
    node_without_decorators.lineno = 10
    del node_without_decorators.decorator_list  # Ensure attribute is missing
    
    assert get_first_line_number(node_without_decorators) == 10

    # Case 3: Node with empty decorator list (should return node's lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure it doesn't have the attribute
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = Mock()
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = Mock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node where decorator_list attribute is missing (handled by getattr default)
    node_missing_attr = Mock()
    node_missing_attr.lineno = 30
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 4: Node where decorator_list is present but node.lineno is the fallback
    node_with_attr_but_no_content = MagicMock()
    node_with_attr_but_no_content.decorator_list = []
    node_with_attr_but_no_content.lineno = 30
    assert get_first_line_number(node_with_attr_but_no_content) == 30
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 10
    node_with_decorators.decorator_list = [decorator1, decorator2]
    
    # Should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return the lineno of the first decorator)
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 15
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node where decorator_list attribute is missing entirely
    node_missing_attr = MagicMock(spec=[])
    node_missing_attr.lineno = 30
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node with decorators
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    
    assert get_first_line_number(node_with_decorators) == 5

    # Case 2: Node without decorators
    node_without_decorators = MagicMock()
    node_without_decorators.decorator_list = []
    node_without_decorators.lineno = 10
    
    assert get_first_line_number(node_without_decorators) == 10

    # Case 3: Node where decorator_list attribute is missing
    node_missing_attr = MagicMock(spec=[])
    del node_missing_attr.decorator_list
    node_missing_attr.lineno = 20
    
    assert get_first_line_number(node_missing_attr) == 20
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with an empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return the first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure it doesn't have the attribute
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (returns node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with an empty decorator list (returns node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (returns first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node where lineno and first decorator lineno are same
    node_same_line = MagicMock()
    decorator_same = MagicMock()
    decorator_same.lineno = 15
    node_same_line.decorator_list = [decorator_same]
    node_same_line.lineno = 15
    assert get_first_line_number(node_same_line) == 15
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = Mock()
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = Mock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure no decorator_list attribute exists
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # The function should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node where decorator_list is an empty list (explicitly present)
    node_attr_empty = MagicMock()
    node_attr_empty.lineno = 30
    node_attr_empty.decorator_list = []
    assert get_first_line_number(node_attr_empty) == 30
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node has no decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators, should return the lineno of the first decorator
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator_list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 3: Node with decorators
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 15
    node_with_decorators.decorator_list = [decorator1, decorator2]
    
    # Should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with an empty decorator list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node has no decorators, return node.lineno
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators, return lineno of the first decorator
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator_list, return node.lineno
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (returns node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (returns first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list (returns node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    # Ensure decorator_list doesn't exist or is empty
    del node_no_decorators.decorator_list
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator_list attribute
    node_empty_list = MagicMock()
    node_empty_list.decorator_list = []
    node_empty_list.lineno = 15
    assert get_first_line_number(node_empty_list) == 15
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure it doesn't have decorator_list attribute
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return the first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Test case 1: Node without decorators (should return node.lineno)
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorator_list attribute but empty (should return node.lineno)
    node_empty_decorators = Mock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 3: Node with decorators (should return first decorator's lineno)
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    
    node_with_decorators = Mock()
    node_with_decorators.lineno = 10
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 4: Node where decorator_list is missing (should return node.lineno)
    # This covers the getattr(node, "decorator_list", []) fallback
    node_missing_attr = Mock()
    node_missing_attr.lineno = 30
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #11
#--------------------------

```python
class MockNode:
    def __init__(self, lineno, decorator_list=None):
        self.lineno = lineno
        if decorator_list is not None:
            self.decorator_list = decorator_list

class MockDecorator:
    def __init__(self, lineno):
        self.lineno = lineno

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MockNode(lineno=10)
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    decorator1 = MockDecorator(lineno=5)
    decorator2 = MockDecorator(lineno=7)
    node_with_decorators = MockNode(lineno=10, decorator_list=[decorator1, decorator2])
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MockNode(lineno=15, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # The main node lineno is irrelevant if decorators exist
    node_with_decorators.lineno = 10 
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node where decorator_list is an empty list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with an empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return lineno of the first decorator)
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 15
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node has no decorators, should return node.lineno
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators, should return lineno of the first decorator
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator_list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with an empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return the lineno of the first decorator)
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 15
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node where decorators exist but are not the focus (testing logic flow)
    node_single_decorator = MagicMock()
    node_single_decorator.lineno = 30
    decorator_only = MagicMock()
    decorator_only.lineno = 25
    node_single_decorator.decorator_list = [decorator_only]
    assert get_first_line_number(node_single_decorator) == 25
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure it doesn't have the attribute
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # The node's own lineno should be ignored if decorators exist
    node_with_decorators.lineno = 10 
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator_list attribute
    node_empty_list = MagicMock()
    node_empty_list.decorator_list = []
    node_empty_list.lineno = 20
    assert get_first_line_number(node_empty_list) == 20

    # Case 4: Node where decorator_list is missing (handled by getattr default)
    node_missing_attr = MagicMock()
    node_missing_attr.lineno = 30
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node has no decorators, should return node.lineno
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators, should return lineno of the first decorator
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator_list, should return node.lineno
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node has decorator_list but it's None (though getattr handles empty list fallback)
    # Testing the logic when decorators is an empty list via getattr default
    node_no_attr = MagicMock(spec=[])
    # Manually deleting to simulate attribute absence for getattr(node, "decorator_list", [])
    if hasattr(node_no_attr, 'decorator_list'):
        del node_no_attr.decorator_list
    node_no_attr.lineno = 30
    assert get_first_line_number(node_no_attr) == 30
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure it doesn't have decorator_list
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # Even if the node's lineno is different, it should return the first decorator's lineno
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator_list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15

    # Case 4: Node where decorator_list attribute is missing (handled by getattr default)
    node_no_attr = MagicMock(spec=[]) 
    # We manually set lineno because spec=[] removes all other attributes
    node_no_attr.lineno = 20
    assert get_first_line_number(node_no_attr) == 20
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    
    assert get_first_line_number(node_with_decorators) == 5

    # Case 2: Node without decorators (should return node's lineno)
    node_without_decorators = MagicMock()
    node_without_decorators.lineno = 10
    # Ensure decorator_list is not present or empty
    del node_without_decorators.decorator_list 
    
    assert get_first_line_number(node_without_decorators) == 10

    # Case 3: Node with empty decorator_list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    
    assert get_first_line_number(node_with_decorators) == 5

    # Case 2: Node without decorators (should return node's lineno)
    node_without_decorators = MagicMock()
    node_without_decorators.lineno = 10
    del node_without_decorators.decorator_list  # Simulate attribute missing
    
    assert get_first_line_number(node_without_decorators) == 10

    # Case 3: Node with empty decorator list (should return node's lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure no decorator_list attribute exists
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with an empty decorator list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure decorator_list doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # lineno of the node itself is different from first decorator
    node_with_decorators.lineno = 10 
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with an empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node has no decorators, should return node.lineno
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators, should return lineno of the first decorator
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator_list attribute
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure no decorator_list attribute exists
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with an empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node has no decorators, should return node.lineno
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure it doesn't have the attribute
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators, should return lineno of the first decorator
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # Even if node.lineno is different, it should return the first decorator's lineno
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator_list attribute
    node_empty_list = MagicMock()
    node_empty_list.decorator_list = []
    node_empty_list.lineno = 20
    assert get_first_line_number(node_empty_list) == 20

    # Case 4: Node has decorator_list but it contains only one element
    single_decorator = MagicMock()
    single_decorator.lineno = 1
    node_single_decorator = MagicMock()
    node_single_decorator.decorator_list = [single_decorator]
    node_single_decorator.lineno = 5
    assert get_first_line_number(node_single_decorator) == 1
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators (uses node.lineno)
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators (uses first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list (uses node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    
    node_with_decorators = Mock()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    # Should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node where decorator_list is an empty list
    node_empty_decorators = Mock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15

    # Case 4: Node where decorator_list attribute does not exist
    node_no_attr = Mock(spec=[])
    node_no_attr.lineno = 20
    # delattr to ensure it's truly missing if Mock pre-filled it
    if hasattr(node_no_attr, "decorator_list"):
        del node_no_attr.decorator_list
    assert get_first_line_number(node_no_attr) == 20
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with an empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return lineno of the first decorator)
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 15
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = Mock()
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # The lineno of the node itself might be 10, but it should return first decorator's line
    node_with_decorators.lineno = 10 
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node where decorator_list is an empty list
    node_empty_decorator_list = Mock()
    node_empty_decorator_list.lineno = 20
    node_empty_decorator_list.decorator_list = []
    assert get_first_line_number(node_empty_decorator_list) == 20

    # Case 4: Node where decorator_list attribute does not exist
    node_no_attr = Mock(spec=[])
    node_no_attr.lineno = 30
    # delete lineno if it was added by spec for safety in testing logic
    if hasattr(node_no_attr, 'lineno'):
        del node_no_attr.lineno
    # Re-mocking to ensure no decorator_list attribute exists
    node_no_attr = Mock()
    node_no_attr.lineno = 30
    # We use a manual mock to simulate the absence of the attribute
    class NoAttrNode:
        def __init__(self):
            self.lineno = 30
    
    node_no_attr_actual = NoAttrNode()
    assert get_first_line_number(node_no_attr_actual) == 30
```


