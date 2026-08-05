####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
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
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # The function should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with an empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node has no decorators, should return node.lineno
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list # Ensure attribute doesn't exist
    node_no_decorators.lineno = 10
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
    node_empty_list = MagicMock()
    node_empty_list.decorator_list = []
    node_empty_list.lineno = 20
    assert get_first_line_number(node_empty_list) == 20

    # Case 4: Node has decorators but first decorator is the target
    node_single_decorator = MagicMock()
    decorator_single = MagicMock()
    decorator_single.lineno = 1
    node_single_decorator.decorator_list = [decorator_single]
    assert get_first_line_number(node_single_decorator) == 1
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node has no decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist to test getattr default
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators (should return the lineno of the first decorator)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    node_no_decorators.lineno = 10
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
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #5
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

    # Case 3: Node with empty decorator list
    node_empty_decorators = Mock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node where decorator_list attribute is missing
    node_no_attr = Mock(spec=[])
    node_no_attr.lineno = 30
    # Del attributes to simulate absence if not using spec
    if hasattr(node_no_attr, 'decorator_list'):
        del node_no_attr.decorator_list
    assert get_first_line_number(node_no_attr) == 30
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure decorator_list doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 4: Node where decorator_list is present but contains an object with lineno
    node_single_decorator = MagicMock()
    decorator = MagicMock()
    decorator.lineno = 1
    node_single_decorator.decorator_list = [decorator]
    assert get_first_line_number(node_single_decorator) == 1
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
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorator_list empty
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 3: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 30
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 4: Node where decorator_list is missing (handled by getattr default)
    node_missing_attr = MagicMock(spec=[])
    node_missing_attr.lineno = 40
    assert get_first_line_number(node_missing_attr) == 40
```


# LLM-generated content at query #8
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

    # Test case 2: Node with decorators (should return first decorator's lineno)
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


# LLM-generated content at query #9
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
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #10
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

    # Test case 3: Node with empty decorator_list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 4: Node where decorator_list attribute is missing but lineno exists
    node_missing_attr = MagicMock(spec=["lineno"])
    node_missing_attr.lineno = 30
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #11
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

    # Case 2: Node with decorator_list attribute but empty (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5
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
    del node_no_decorators.decorator_list  # Ensure attribute is missing for getattr default
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorator_list but it is empty, should return node.lineno
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node has decorators, should return lineno of the first decorator
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 15
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node has decorator_list attribute as an empty list explicitly
    node_explicit_empty = MagicMock()
    node_explicit_empty.lineno = 30
    node_explicit_empty.decorator_list = []
    assert get_first_line_number(node_explicit_empty) == 30
```


# LLM-generated content at query #13
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

    # Case 2: Node has decorators, should return the first decorator's lineno
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator_list attribute
    node_empty_list = MagicMock()
    node_empty_list.decorator_list = []
    node_empty_list.lineno = 15
    assert get_first_line_number(node_empty_list) == 15

    # Case 4: Node has lineno but decorator_list is missing (handled by getattr default)
    node_missing_attr = MagicMock()
    node_missing_attr.lineno = 20
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 20
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist for testing getattr fallback
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with empty decorator list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 30
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node where decorator_list is present but node has a different lineno
    node_complex = MagicMock()
    node_complex.lineno = 100
    decorator_top = MagicMock()
    decorator_top.lineno = 95
    node_complex.decorator_list = [decorator_top]
    assert get_first_line_number(node_complex) == 95
```


# LLM-generated content at query #15
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

    # Test case 2: Node with decorators (should return first decorator's lineno)
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


# LLM-generated content at query #16
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
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 30
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # Should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #17
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

    # Case 2: Node with decorators (should return first decorator's lineno)
    decorator1 = MockDecorator(lineno=5)
    decorator2 = MockDecorator(lineno=7)
    node_with_decorators = MockNode(lineno=10, decorator_list=[decorator1, decorator2])
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MockNode(lineno=15, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 15

    # Case 4: Node where decorator_list attribute is missing (fallback to lineno)
    node_missing_attr = MockNode(lineno=20)
    del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 20
```


# LLM-generated content at query #18
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


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node with no decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node where decorator_list is an empty list
    node_empty_decorator_list = MagicMock()
    node_empty_decorator_list.decorator_list = []
    node_empty_decorator_list.lineno = 15
    assert get_first_line_number(node_empty_decorator_list) == 15

    # Case 4: Node where decorator_list attribute is missing
    node_no_attr = MagicMock(spec=[])
    # We use delete to simulate missing attribute if it was added by MagicMock
    if hasattr(node_no_attr, 'lineno'):
        del node_no_attr.lineno
    node_no_attr.lineno = 20
    assert get_first_line_number(node_no_attr) == 20
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

    # Case 4: Node where decorator_list is missing but lineno exists
    node_missing_attr = Mock()
    node_missing_attr.lineno = 30
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    node_no_decorators.lineno = 10
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
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node where decorator_list attribute is missing (should return node.lineno)
    node_missing_attr = MagicMock(spec=["lineno"])
    del node_missing_attr.decorator_list
    node_missing_attr.lineno = 30
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #22
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

    # Test case 2: Node with decorator_list but empty (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 3: Node with decorators (should return lineno of first decorator)
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 4: Node where node.lineno is different from first decorator's lineno
    # This verifies the logic specifically mentioned in the docstring
    decorator_only = MagicMock()
    decorator_only.lineno = 1
    node_decorated = MagicMock()
    node_decorated.decorator_list = [decorator_only]
    node_decorated.lineno = 5
    assert get_first_line_number(node_decorated) == 1
```


# LLM-generated content at query #23
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

    # Test case 2: Node with decorator_list as empty list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 3: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #24
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

    # Case 2: Node with empty decorator_list (uses node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (uses first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # Even if node.lineno is different, it should return the first decorator's lineno
    node_with_decorators.lineno = 10 
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #25
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

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
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
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return lineno of first decorator)
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 10
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5
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

    # Case 2: Node with decorators (should return first decorator's lineno)
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
    # Case 1: Node without decorators (uses node.lineno)
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (uses first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list (uses node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node where decorator_list attribute is missing but lineno exists
    node_missing_attr = MagicMock()
    node_missing_attr.lineno = 30
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 30
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
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # The lineno of the node itself should be ignored if decorators exist
    node_with_decorators.lineno = 10 
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node where decorator_list attribute is missing
    node_no_attr = MagicMock(spec=[])
    del node_no_attr.lineno # Ensure it doesn't have lineno to test error if logic fails, 
                             # but the function expects it if no decorators exist.
    # Re-mocking specifically for attribute absence
    node_no_attr = MagicMock()
    node_no_attr.lineno = 30
    if hasattr(node_no_attr, 'decorator_list'):
        del node_no_attr.decorator_list
    assert get_first_line_number(node_no_attr) == 30
```


# LLM-generated content at query #30
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


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
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

    # Case 2: Node has decorators, should return lineno of the first decorator
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node has decorator_list but it's not the primary source of truth (testing getattr fallback)
    node_with_attr = MagicMock()
    decorator_top = MagicMock()
    decorator_top.lineno = 1
    node_with_attr.decorator_list = [decorator_top]
    assert get_first_line_number(node_with_attr) == 1
```


# LLM-generated content at query #2
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
    decorator2.lineno = 7
    
    node_with_decorators = Mock()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    # Should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list attribute
    node_empty_list = Mock()
    node_empty_list.decorator_list = []
    node_empty_list.lineno = 15
    assert get_first_line_number(node_empty_list) == 15

    # Case 4: Node missing decorator_list attribute entirely
    node_no_attr = Mock(spec=[])
    node_no_attr.lineno = 20
    # Using delattr to ensure the attribute is truly missing for the test
    if hasattr(node_no_attr, 'decorator_list'):
        delattr(node_no_attr, 'decorator_list')
    assert get_first_line_number(node_no_attr) == 20
```


# LLM-generated content at query #3
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

    # Case 4: Node where decorator_list attribute is missing (handled by getattr default)
    node_missing_attr = MagicMock(spec=['lineno'])
    node_missing_attr.lineno = 40
    assert get_first_line_number(node_missing_attr) == 40
```


# LLM-generated content at query #4
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

    # Test case 3: Node with decorators (should return lineno of the first decorator)
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    
    node_with_decorators = Mock()
    node_with_decorators.lineno = 10
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 4: Node where lineno and first decorator lineno are the same
    node_same_line = Mock()
    node_same_line.lineno = 15
    decorator_same = Mock()
    decorator_same.lineno = 15
    node_same_line.decorator_list = [decorator_same]
    assert get_first_line_number(node_same_line) == 15
```


# LLM-generated content at query #5
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
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Case 1: Node has no decorators, should return node.lineno
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators, should return the lineno of the first decorator
    node_with_decorators = Mock()
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator_list, should return node.lineno
    node_empty_decorators = Mock()
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

    # Case 4: Node where lineno is different from decorator lineno
    node_mismatch = MagicMock()
    decorator_top = MagicMock()
    decorator_top.lineno = 1
    node_mismatch.decorator_list = [decorator_top]
    node_mismatch.lineno = 3
    assert get_first_line_number(node_mismatch) == 1
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    if hasattr(node_no_decorators, 'decorator_list'):
        del node_no_decorators.decorator_list
    else:
        # Use a mock that doesn't have the attribute to simulate absence
        pass 
    
    # To be safe with MagicMock, we explicitly delete or ensure it's not there
    delattr(node_no_decorators, 'decorator_list') if hasattr(node_no_decorators, 'decorator_list') else None
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

    # Case 3: Node with an empty decorator_list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #11
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
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node with no decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return the first decorator's lineno)
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

    # Case 4: Node where decorator_list attribute is missing (handled by getattr default)
    node_missing_attr = MagicMock(spec=['lineno'])
    node_missing_attr.lineno = 20
    assert get_first_line_number(node_missing_attr) == 20
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


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator list (should return node.lineno)
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
    # Test case 1: Node without decorators (uses node.lineno)
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure it doesn't have the attribute
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


# LLM-generated content at query #16
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
    decorator2.lineno = 7
    
    node_with_decorators = Mock()
    node_with_decorators.lineno = 10
    node_with_decorators.decorator_list = [decorator1, decorator2]
    
    # Should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with an empty decorator list attribute
    node_empty_list = Mock()
    node_empty_list.lineno = 15
    node_empty_list.decorator_list = []
    assert get_first_line_number(node_empty_list) == 15

    # Case 4: Node missing decorator_list attribute entirely
    node_missing_attr = Mock()
    node_missing_attr.lineno = 20
    # Delete to ensure it's not there (in case of mock reuse)
    if hasattr(node_missing_attr, "decorator_list"):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 20
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

    # Test case 2: Node with decorator list (returns first decorator line)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list (returns node lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 15
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 15
```


# LLM-generated content at query #18
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

    # Case 2: Node with decorator_list but empty (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 30
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node where decorator_list is missing (should return node.lineno via getattr default)
    node_missing_attr = MagicMock()
    node_missing_attr.lineno = 40
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 40
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node with no decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with empty decorator list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return the lineno of the first decorator)
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #20
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


# LLM-generated content at query #21
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
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5
```


# LLM-generated content at query #23
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


# LLM-generated content at query #24
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

    # Case 2: Node with decorator_list attribute but empty (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators (should return first decorator's lineno)
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node where node.lineno is different from decorator lineno
    # (Verifying the logic specifically addresses the docstring's requirement)
    node_with_diff_lineno = MagicMock()
    decorator_top = MagicMock()
    decorator_top.lineno = 1
    node_with_diff_lineno.decorator_list = [decorator_top]
    node_with_diff_lineno.lineno = 5
    assert get_first_line_number(node_with_diff_lineno) == 1
```


# LLM-generated content at query #25
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


# LLM-generated content at query #26
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


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators (should return lineno of first decorator)
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    
    node_with_decorators = Mock()
    node_with_decorators.lineno = 10
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = Mock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 4: Node where decorator_list attribute is missing
    node_no_attr = Mock(spec=[])
    node_no_attr.lineno = 30
    # deleting lineno to simulate a raw object without the attribute if needed, 
    # but here we just ensure it falls back to node.lineno via getattr default
    assert get_first_line_number(node_no_attr) == 30
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

    # Case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #29
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
    # Case 1: No decorators, should return node.lineno
    node_no_decorators = MockNode(lineno=10)
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: With decorators, should return lineno of the first decorator
    decorator1 = MockDecorator(lineno=5)
    decorator2 = MockDecorator(lineno=6)
    node_with_decorators = MockNode(lineno=10, decorator_list=[decorator1, decorator2])
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Empty decorator list, should return node.lineno
    node_empty_decorators = MockNode(lineno=20, decorator_list=[])
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node with no decorator_list attribute at all
    del node_no_decorators.decorator_list
    assert get_first_line_number(node_no_decorators) == 10
```


# LLM-generated content at query #30
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
    node_with_decorators.lineno = 10
    node_with_decorators.decorator_list = [decorator1, decorator2]
    
    # Should return the lineno of the first decorator (5)
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator_list attribute
    node_empty_list = Mock()
    node_empty_list.lineno = 20
    node_empty_list.decorator_list = []
    assert get_first_line_number(node_empty_list) == 20

    # Case 4: Node without decorator_list attribute at all
    node_no_attr = Mock(spec=[])
    node_no_attr.lineno = 30
    # Using delattr to ensure the attribute doesn't exist if it was somehow injected
    if hasattr(node_no_attr, "decorator_list"):
        delattr(node_no_attr, "decorator_list")
    assert get_first_line_number(node_no_attr) == 30
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Case 1: Node has no decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators
    node_with_decorators = Mock()
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    # Even if node itself has a lineno, it should return the first decorator's lineno
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator_list
    node_empty_decorators = Mock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15

    # Case 4: Node has decorator_list as an empty list (edge case for getattr default)
    node_none_decorators = Mock()
    # Simulate attribute not present via AttributeError during getattr
    def side_effect(obj, attr, default):
        if attr == "decorator_list":
            return []
        return getattr(obj, attr)
    
    # Since we can't easily mock the built-in getattr globally without monkeypatching, 
    # we rely on the Mock behavior where we manually set it to an empty list.
    node_empty_list = Mock()
    node_empty_list.decorator_list = []
    node_empty_list.lineno = 20
    assert get_first_line_number(node_empty_list) == 20
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node with decorators - should return lineno of the first decorator
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 3
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 2: Node without decorator_list attribute - should return node.lineno
    node_without_attr = MagicMock()
    del node_without_attr.decorator_list
    node_without_attr.lineno = 10
    assert get_first_line_number(node_without_attr) == 10

    # Case 3: Node with empty decorator_list - should return node.lineno
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node with decorators where the first decorator is the only one
    node_single_decorator = MagicMock()
    decorator_single = MagicMock()
    decorator_single.lineno = 15
    node_single_decorator.decorator_list = [decorator_single]
    assert get_first_line_number(node_single_decorator) == 15

    # Case 5: Standard node with no decorators (lineno is the fallback)
    node_standard = MagicMock()
    node_standard.lineno = 100
    node_standard.decorator_list = []
    assert get_first_line_number(node_standard) == 100
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute is missing to test getattr default
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with empty decorator list (should return node.lineno)
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
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    
    # Should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node where decorator_list is an empty list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15

    # Case 4: Node where decorator_list attribute does not exist
    del node_no_decorators.decorator_list
    assert get_first_line_number(node_no_decorators) == 10
```


# LLM-generated content at query #35
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

    # Case 3: Node with empty decorator_list (should return node.lineno)
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #36
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

    # Case 3: Node with decorators (should return lineno of the first decorator)
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 30
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 7
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node where decorator_list attribute is missing entirely (fallback to lineno)
    node_missing_attr = MagicMock()
    node_missing_attr.lineno = 40
    if hasattr(node_missing_attr, "decorator_list"):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 40
```


# LLM-generated content at query #37
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
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 4: Node where decorator_list is present but node.lineno is different
    node_mixed = MagicMock()
    node_mixed.lineno = 15
    decorator_top = MagicMock()
    decorator_top.lineno = 12
    node_mixed.decorator_list = [decorator_top]
    assert get_first_line_number(node_mixed) == 12
```


# LLM-generated content at query #38
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
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 10
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node where decorators attribute is missing entirely
    node_missing_attr = MagicMock(spec=[])
    node_missing_attr.lineno = 30
    # Using a fresh mock to ensure no decorator_list exists
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #39
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
```


# LLM-generated content at query #40
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Case 1: Node has no decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node has decorators (should return first decorator's lineno)
    node_with_decorators = Mock()
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node has an empty decorator_list
    node_empty_decorators = Mock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #41
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
    del node_no_decorators.decorator_list
    assert get_first_line_number(node_no_decorators) == 10
```


# LLM-generated content at query #42
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist to test getattr default
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

    # Case 4: Node where decorator_list is present but node has its own lineno
    node_with_lineno = MagicMock()
    decorator = MagicMock()
    decorator.lineno = 1
    node_with_lineno.decorator_list = [decorator]
    node_with_lineno.lineno = 10
    assert get_first_line_number(node_with_lineno) == 1
```


# LLM-generated content at query #43
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

    # Test case 2: Node with decorators (should return first decorator's lineno)
    node_with_decorators = Mock()
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list (should return node.lineno)
    node_empty_decorators = Mock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 4: Node where decorator_list is present but contains only one element
    node_single_decorator = Mock()
    decorator_only = Mock()
    decorator_only.lineno = 1
    node_single_decorator.decorator_list = [decorator_only]
    node_single_decorator.lineno = 5
    assert get_first_line_number(node_single_decorator) == 1
```


# LLM-generated content at query #44
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    # Ensure decorator_list is not present or empty
    del node_no_decorators.decorator_list
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with an empty decorator list
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Test case 3: Node with decorators (should return the lineno of the first decorator)
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 10
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 4: Node where lineno is different from decorators' lineno
    node_complex = MagicMock()
    node_complex.lineno = 100
    decorator_top = MagicMock()
    decorator_top.lineno = 95
    node_complex.decorator_list = [decorator_top]
    assert get_first_line_number(node_complex) == 95
```


# LLM-generated content at query #45
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
    # The node's own lineno should be ignored if decorators exist
    node_with_decorators.lineno = 10 
    
    assert get_first_line_number(node_with_decorators) == 5

    # Case 2: Node without decorators
    node_without_decorators = MagicMock()
    node_without_decorators.decorator_list = []
    node_without_decorators.lineno = 15
    
    assert get_first_line_number(node_without_decorators) == 15

    # Case 3: Node where decorator_list attribute is missing
    node_missing_attr = MagicMock(spec=[])
    del node_missing_attr.decorator_list
    node_missing_attr.lineno = 20
    
    assert get_first_line_number(node_missing_attr) == 20
```


# LLM-generated content at query #46
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

    # Case 2: Node with decorators (should return the first decorator's lineno)
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


# LLM-generated content at query #47
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
```


# LLM-generated content at query #48
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node has no decorators, should return node.lineno
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list
    node_no_decorators.lineno = 10
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
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node has decorator_list but it's not present (handled by getattr default)
    node_no_attr = MagicMock(spec=[])
    del node_no_attr.lineno
    # We must ensure lineno is added back for the fallback logic to work if no decorators exist
    node_no_attr.lineno = 30
    assert get_first_line_number(node_no_attr) == 30
```


# LLM-generated content at query #49
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    # Ensure decorator_list is not present or empty
    del node_no_decorators.decorator_list
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return the first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node with empty decorator_list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


# LLM-generated content at query #50
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node with no decorators - should return node.lineno
    node_no_decorators = MagicMock()
    node_no_decorators.lineno = 10
    del node_no_decorators.decorator_list  # Ensure attribute is missing for getattr check
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with empty decorator list - should return node.lineno
    node_empty_decorators = MagicMock()
    node_empty_decorators.lineno = 20
    node_empty_decorators.decorator_list = []
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 3: Node with decorators - should return lineno of the first decorator
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    
    node_with_decorators = MagicMock()
    node_with_decorators.lineno = 15
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 4: Node where lineno and first decorator lineno are the same
    node_same_line = MagicMock()
    node_same_line.lineno = 10
    decorator_same_line = MagicMock()
    decorator_same_line.lineno = 10
    node_same_line.decorator_list = [decorator_same_line]
    assert get_first_line_number(node_same_line) == 10
```


# LLM-generated content at query #51
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

    # Case 2: Node with decorators (should return the first decorator's lineno)
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


# LLM-generated content at query #52
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


# LLM-generated content at query #53
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

    # Case 4: Node where decorator_list is missing but lineno exists
    node_missing_attr = MagicMock()
    node_missing_attr.lineno = 30
    if hasattr(node_missing_attr, 'decorator_list'):
        del node_missing_attr.decorator_list
    assert get_first_line_number(node_missing_attr) == 30
```


# LLM-generated content at query #54
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_get_first_line_number():
    # Case 1: Node without decorators (should return node.lineno)
    node_no_decorators = MagicMock()
    del node_no_decorators.decorator_list  # Ensure attribute doesn't exist
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Case 2: Node with decorators (should return first decorator's lineno)
    node_with_decorators = MagicMock()
    decorator1 = MagicMock()
    decorator1.lineno = 5
    decorator2 = MagicMock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    assert get_first_line_number(node_with_decorators) == 5

    # Case 3: Node where decorator_list is an empty list
    node_empty_decorators = MagicMock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20

    # Case 4: Node with decorator_list attribute but no actual items (edge case)
    node_none_decorator_list = MagicMock()
    node_none_decorator_list.decorator_list = []
    node_none_decorator_list.lineno = 30
    assert get_first_line_number(node_none_decorator_list) == 30
```


# LLM-generated content at query #55
#--------------------------

```python
import pytest
from unittest.mock import Mock

def test_get_first_line_number():
    # Test case 1: Node without decorators
    node_no_decorators = Mock()
    node_no_decorators.lineno = 10
    assert get_first_line_number(node_no_decorators) == 10

    # Test case 2: Node with decorators
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 7
    
    node_with_decorators = Mock()
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    
    # Should return the lineno of the first decorator
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = Mock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 15
    assert get_first_line_number(node_empty_decorators) == 15

    # Test case 4: Node missing decorator_list attribute entirely
    node_no_attr = Mock(spec=[]) # No attributes
    node_no_attr.lineno = 20
    assert get_first_line_number(node_no_attr) == 20
```


# LLM-generated content at query #56
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


# LLM-generated content at query #57
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

    # Test case 2: Node with decorators (should return first decorator's lineno)
    node_with_decorators = Mock()
    decorator1 = Mock()
    decorator1.lineno = 5
    decorator2 = Mock()
    decorator2.lineno = 6
    node_with_decorators.decorator_list = [decorator1, decorator2]
    node_with_decorators.lineno = 10
    assert get_first_line_number(node_with_decorators) == 5

    # Test case 3: Node with empty decorator list
    node_empty_decorators = Mock()
    node_empty_decorators.decorator_list = []
    node_empty_decorators.lineno = 20
    assert get_first_line_number(node_empty_decorators) == 20
```


