####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pathlib import Path

def test_get_file_name():
    # Test with template name that does not have .json extension
    assert get_file_name("/tmp/replay", "my_template") == os.path.join("/tmp/replay", "my_template.json")
    
    # Test with template name that already has .json extension
    assert get_file_name("/tmp/replay", "my_template.json") == os.path.join("/tmp/replay", "my_template.json")
    
    # Test with Path object
    replay_path = Path("/tmp/replay")
    assert get_file_name(replay_path, "test") == os.path.join("/tmp/replay", "test.json")
    
    # Test with empty template name
    assert get_file_name("dir", "") == os.path.join("dir", ".json")
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pathlib import Path

def test_get_file_name():
    # Test case 1: template_name does not end with .json (should append .json)
    dir_path = "/tmp/replay"
    template = "my_template"
    expected = "/tmp/replay/my_template.json"
    assert get_file_name(dir_path, template) == expected

    # Test case 2: template_name already ends with .json (should not append .json)
    template_with_ext = "my_template.json"
    expected_with_ext = "/tmp/replay/my_template.json"
    assert get_file_name(dir_path, template_with_ext) == expected_with_ext

    # Test case 3: Using Path object instead of string
    path_obj = Path("/tmp/replay")
    assert get_file_name(path_obj, "test") == "/tmp/replay/test.json"

    # Test case 4: Empty template name
    assert get_file_name(dir_path, "") == "/tmp/replay/.json"
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {"project_name": "test_project"},
        "other_key": "value"
    }
    
    invalid_context = {
        "no_cookiecutter_key": True
    }

    # Test successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing 'cookiecutter' key raises ValueError
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_, "invalid_template")

    # Test load with non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test template name with .json suffix already present
    template_with_ext = "template.json"
    replay_file_ext = replay_dir / "template.json"
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context_ext = load(replay_dir, template_with_ext)
    assert loaded_context_ext == valid_context
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pathlib import Path
import json
import os

def test_dump(tmp_path):
    replay_dir = tmp_path / "replays"
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "extra_data": 123
    }

    # Test successful dump
    dump(replay_dir, template_name, context)

    # Verify file exists with correct name (suffix .json added)
    expected_file = replay_dir / "my_template.json"
    assert expected_file.exists()

    # Verify content integrity
    with open(expected_file, 'r', encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context

    # Test dump with template name already containing .json
    dump(replay_dir, "already_has.json", context)
    expected_file_no_double_ext = replay_dir / "already_has.json"
    assert expected_file_no_double_ext.exists()

    # Test ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, "invalid_template", invalid_context)

    # Test that make_sure_path_exists logic is invoked (directory creation)
    new_deep_dir = tmp_path / "a" / "b" / "c"
    dump(new_deep_dir, "nested_test", context)
    assert new_deep_dir.exists()
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    json_content = json.dumps(context, indent=2)

    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir:
        with patch("builtins.open", mock_open()) as m_open:
            # Test successful dump
            dump(replay_dir, template_name, context)
            
            mock_mkdir.assert_called_once_with(replay_dir)
            m_open.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
            
            # Verify content written to file
            handle = m_open()
            # Collect all arguments passed to write calls
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)
            assert json.loads(written_data) == context

    # Test ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "oops"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)

    # Test suffix handling for template_name already ending in .json
    template_with_ext = "test.json"
    expected_file_path_ext = "/tmp/replay/test.json"
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with patch("builtins.open", mock_open()):
            dump(replay_dir, template_with_ext, context)
            m_open.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from pathlib import Path
import json

def test_load(tmp_path):
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    # Test Case 1: Successful load
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Test Case 2: Missing 'cookiecutter' key raises ValueError
    invalid_context = {"not_cookiecutter": "data"}
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'wrap_encoding="utf-8"', encoding="utf-8") as f:
        # Manually overwrite with invalid data
        import json
        with open(invalid_file, 'w', encoding="utf-8") as f_write:
            json.dump(invalid_context, f_write)
            
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Test Case 3: File does not exist raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test Case 4: Template name with .json suffix (verifying get_file_name logic inside load)
    json_template_name = "explicit_name.json"
    with open(replay_dir / "explicit_name.json", 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_explicit = load(replay_dir, json_template_name)
    assert loaded_explicit == valid_context
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = f"{replay_dir}/{template_name}.json"

    # Test case 1: Successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)

        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test case 2: Template name already ends with .json
    template_name_json = "my_template.json"
    expected_file_path_json = f"{replay_dir}/{template_name_json}"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_name_json, context)
        mocked_file.assert_called_once_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test case 3: Missing 'cookiecutter' key in context should raise ValueError
    invalid_context = {"not_cookiecutter": "data"}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import json
import os
from pathlib import Path

def test_load(tmp_path):
    # Setup
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }
    
    replay_file = os_path_join(replay_dir, f"{template_name}.json")

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load failure due to missing 'cookiecutter' key
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

    # Test Case 3: Load failure due to non-existent file
    non_existent_template = "ghost_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

def os_path_join(a, b):
    return os.path.join(a, b)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = f"{replay_dir}/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        m_json_dump.assert_called_once_with(context, m_open(), indent=2)

    # Test dump with .json extension in template name
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump"):
        
        template_with_ext = "my_template.json"
        expected_file_path_ext = f"{replay_dir}/my_template.json"
        
        dump(replay_dir, template_with_ext, context)
        m_open.assert_called_once_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump with missing cookiecutter key (raises ValueError)
    invalid_context = {"not_cookiecutter": "data"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "test_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "tester"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }

    # Case 1: Successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "my_project"

    # Case 2: Load failure due to missing 'cookiecutter' key
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")

    # Case 3: File does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        from cookiecutter.replay import dump
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "data"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        from cookiecutter.replay import dump
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)

    # Test handling of template_name already containing .json
    template_with_ext = "my_template.json"
    expected_file_path_with_ext = "/tmp/replay/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        from cookiecutter.replay import dump
        dump(replay_dir, template_with_ext, context)
        mocked_file.assert_called_once_with(expected_file_path_with_ext, 'w', encoding="utf-8")
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    json_content = json.dumps(context, indent=2)

    with patch("cookiecutter.replay.make_sure_path_exists") as mock_make_dir, \
         patch("builtins.open", mock_open()) as mocked_file:
        
        # Test successful dump
        dump(replay_dir, template_name, context)
        
        mock_make_dir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        
        # Verify content written to file
        handle = mocked_file()
        # We check if any call to write contained the expected JSON string
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written_data) == context

    # Test ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "oops"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)

    # Test template name with .json suffix already present
    template_with_suffix = "my_template.json"
    expected_file_path_suffix = "/tmp/replay/my_template.json"
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()):
        dump(replay_dir, template_with_suffix, context)
        mocked_file.assert_called_with(expected_file_path_suffix, 'w', encoding="utf-8")
```


# LLM-generated content at query #13
#--------------------------

```python
import json
import os
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    invalid_context = {"not_cookiecutter": True}
    
    replay_file = os_path_join_helper(replay_dir, template_name)

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Test Case 2: ValueError when 'cookiecutter' key is missing
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: FileNotFoundError when file does not exist
    non_existent_template = "missing_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

def os_path_join_helper(replay_dir, template_name):
    suffix = '.json' if not template_name.endswith('.json') else ''
    file_name = f'{template_name}{suffix}'
    return os.path.join(str(replay_dir), file_name)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = tmp_path / f"{template_name}.json"
    
    # Create dummy file for successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing 'cookiecutter' key in file
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)

    # Test load when file does not exist
    non_existent_template = "missing_template"
    with pytest.raises(FileNotFoundError):
        load(tmp_path, non_existent_template)

    # Test load with template name already including .json
    replay_file_suffix = tmp_path / f"{template_name}.json"
    with open(replay_file_suffix, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    loaded_suffix_context = load(tmp_path, f"{template_name}.json")
    assert loaded_suffix_context == context
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = f"{replay_dir}/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        m_json.assert_called_once()
        # Verify the content passed to json.dump matches context
        args, _ = m_json.call_args
        assert args[0] == context

    # Test dump with template name already having .json extension
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()), \
         patch("json.dump"):
        
        template_with_ext = "my_template.json"
        expected_file_path_ext = f"{replay_dir}/my_template.json"
        dump(replay_dir, template_with_ext, context)
        
        m_open.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "oops"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)

    # Test dump failure when directory creation fails (simulating permission error)
    with patch("cookiecutter.replay.make_sure_path_exists", side_effect=OSError("Permission denied")):
        with pytest.raises(OSError, match="Permission denied"):
            dump(replay_dir, template_name, context)
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = os.path.join(replay_dir, "my_template.json")

    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        m_json_dump.assert_called_once_with(context, m_open(), indent=2)

    # Test dump with template_name already having .json extension
    template_with_ext = "my_template.json"
    expected_file_path_ext = os.path.join(replay_dir, "my_template.json")
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump"):
        
        dump(replay_dir, template_with_ext, context)
        m_open.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump with missing 'cookiecutter' key in context
    invalid_context = {"not_cookiecutter": "data"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test dump with template name already containing .json
    template_with_ext = "my_template.json"
    expected_file_path_ext = "/tmp/replay/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_with_ext, context)
        mocked_file.assert_called_once_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump with missing 'cookiecutter' key in context
    invalid_context = {"not_cookiecutter": "oops"}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #18
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = tmp_path / f"{template_name}.json"
    
    # Create dummy file
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing cookiecutter key
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)

    # Test load for non-existent file
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test dump with template_name already ending in .json
    template_name_json = "my_template.json"
    expected_file_path_json = "/tmp/replay/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()), \
         patch("json.dump"):
        
        dump(replay_dir, template_name_json, context)
        # Verify path construction logic via the call to open
        # We can check if the argument passed to open was correct
        # (Re-checking the logic in get_file_name via dump)

    # Test ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "data"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = str(Path(replay_dir) / "my_template.json")
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        from cookiecutter.replay import dump
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test dump with template name already containing .json
    template_name_json = "my_template.json"
    expected_file_path_json = str(Path(replay_dir) / "my_template.json")
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        from cookiecutter.replay import dump
        dump(replay_dir, template_name_json, context)
        mocked_file.assert_called_once_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test dump with missing 'cookiecutter' key in context
    invalid_context = {"not_cookiecutter": "data"}
    from cookiecutter.replay import dump as dump_func
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump_func(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #21
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    # Case 1: Valid context with cookiecutter key
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other_data": 123}
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Case 2: Invalid context (missing cookiecutter key)
    invalid_context = {"not_cookiecutter": {}}
    replay_file_invalid = replay_dir / "invalid_template.json"
    with open(replay_file_invalid, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Case 3: File does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Case 4: template_name already ends with .json
    json_template_name = "already_has_extension.json"
    with open(replay_dir / json_template_name, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_json_context = load(replay_dir, json_template_name)
    assert loaded_json_context == valid_context
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from pathlib import Path
import json

def test_load(tmp_path):
    template_name = "test_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    # Case 1: Successful load
    valid_context = {"cookiecutter": {"project_name": "my_project"}, "other": "data"}
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Case 2: Load fails because 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "oops"}
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")

    # Case 3: Load fails because file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }

    # Test successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert "cookiecutter" in loaded_context

    # Test load with missing 'cookiecutter' key
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Test load with non-existent file
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test load with template name already including .json
    json_template_name = "already_has_extension.json"
    with open(replay_dir / json_template_name, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_json_context = load(replay_dir, json_template_name)
    assert loaded_json_context == valid_context
```


# LLM-generated content at query #24
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }

    # Test successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing 'cookiecutter' key (ValueError)
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

    # Test load with non-existent file (FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test suffix handling (.json extension in name)
    template_with_ext = "template.json"
    replay_file_ext = replay_dir / "template.json"
    with open(replay_file_ext, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context_ext = load(replay_dir, template_with_ext)
    assert loaded_context_ext == valid_context
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "no_cookiecutter_key": True
    }
    
    replay_file = replay_dir / f"{template_name}.json"

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load fails when 'cookiecutter' key is missing
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: Load fails when file does not exist
    non_existent_template = "non_existent"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}}
    expected_file_path = "/tmp/replay/my_template.json"

    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_exists, \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json_dump:
        
        from cookiecutter.replay import dump
        dump(replay_dir, template_name, context)

        mock_exists.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        m_json_dump.assert_called_once_with(context, m_open(), indent=2)

    # Test dump with template name already containing .json
    template_name_json = "my_template.json"
    expected_file_path_json = "/tmp/replay/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump"):
        
        from cookiecutter.replay import dump
        dump(replay_dir, template_name_json, context)
        m_open.assert_called_once_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        from cookiecutter.replay import dump
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    json_content = json.dumps(context, indent=2)

    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file:
        
        # Test successful dump
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        
        # Verify the content written to the file
        handle = mocked_file()
        # Check if any call to write contained our json string
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written_data) == context

    # Test ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "oops"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)

    # Test file name handling for template names already ending in .json
    template_with_ext = "template.json"
    expected_file_path_ext = "/tmp/replay/template.json"
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()):
        dump(replay_dir, template_with_ext, context)
        mocked_file.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = f"{replay_dir}/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        # Verify json.dump was called with the correct context
        args, _ = mock_json_dump.call_args
        assert args[0] == context

    # Test dump with existing .json extension in template name
    template_with_ext = "my_template.json"
    expected_file_path_ext = f"{replay_dir}/my_template.json"
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()), \
         patch("json.dump"):
        
        dump(replay_dir, template_with_ext, context)
        mocked_file.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "data"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #29
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {"project_name": "test_project"},
        "other_key": "value"
    }
    
    invalid_context = {
        "no_cookiecutter_key": True
    }

    # Test Case 1: Successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Raise ValueError when 'cookiecutter' key is missing
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Test Case 3: Raise FileNotFoundError when file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test Case 4: Ensure suffix logic handles .json extension correctly
    template_with_ext = "template.json"
    replay_file_ext = replay_dir / "template.json"
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
        
    loaded_data_ext = load(replay_dir, template_with_ext)
    assert loaded_data_ext == valid_context
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "test_template"
    context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "tester"
        },
        "other_key": "value"
    }
    
    # Create replay file manually for setup
    replay_file = tmp_path / f"{template_name}.json"
    import json
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"

    # Test load with existing .json extension in name
    template_with_ext = "test_template.json"
    replay_file_ext = tmp_path / "test_template.json" # same file
    loaded_context_ext = load(tmp_path, template_with_ext)
    assert loaded_context_ext == context

    # Test load failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, "invalid")

    # Test load failure when file does not exist
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = tmp_path / f"{template_name}.json"
    
    # Create dummy file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing cookiecutter key (should raise ValueError)
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

    # Test load with non-existent file (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from pathlib import Path
import json

def test_load(tmp_path):
    # Setup valid data
    template_name = "test_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_user"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }

    # Test Case 1: Successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "my_project"

    # Test Case 2: Load fails when 'cookiecutter' key is missing (ValueError)
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

    # Test Case 3: Load fails when file does not exist (FileNotFoundError)
    non_existent_template = "ghost_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {"project_name": "my_project"},
        "other_key": "value"
    }
    
    invalid_context = {
        "no_cookiecutter_key": "oops"
    }
    
    replay_file = replay_dir / f"{template_name}.json"

    # Test successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"

    # Test load with missing 'cookiecutter' key raises ValueError
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
        
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test load for non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = f"{replay_dir}/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test dump with template_name already having .json suffix
    template_with_ext = "my_template.json"
    expected_file_path_ext = f"{replay_dir}/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_with_ext, context)
        mocked_file.assert_called_once_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump with missing 'cookiecutter' key in context
    invalid_context = {"not_cookiecutter": "data"}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #35
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }

    # Test case 1: Successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test case 2: Load fails due to missing 'cookiecutter' key
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Test case 3: Load fails because file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test case 4: Verify suffix handling (template name with .json)
    suffix_template = "template.json"
    suffix_file = replay_dir / "template.json"
    with open(suffix_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_suffix_context = load(replay_dir, suffix_template)
    assert loaded_suffix_context == valid_context
```


# LLM-generated content at query #36
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = tmp_path / f"{template_name}.json"

    # Create the file manually to simulate a previously dumped state
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with template name already containing .json
    template_with_ext = "my_template.json"
    replay_file_ext = tmp_path / "my_template.json" # get_file_name handles suffix logic
    with open(replay_file_ext, "w", encoding="utf-8") as f:
        json.dump(context, f)
    
    loaded_context_ext = load(tmp_path, template_with_ext)
    assert loaded_context_ext == context

    # Test load failure - Missing cookiecutter key
    invalid_context = {"not_cookiecutter": {}}
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, "invalid")

    # Test load failure - File does not exist
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }

    # Test Case 1: Successful load
    file_path = replay_dir / f"{template_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load fails when 'cookiecutter' key is missing
    invalid_file_path = replay_dir / "invalid_template.json"
    with open(invalid_file_path, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")

    # Test Case 3: Load fails when file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test Case 4: Verify suffix handling via load (ensure .json is appended)
    # Using the previously created valid_context file with a name without extension
    loaded_data_no_ext = load(replay_dir, template_name) # Should point to same file
    assert loaded_data_no_ext == valid_context
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_make_dir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_make_dir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test dump with template name already containing .json
    template_name_json = "my_template.json"
    expected_file_path_json = "/tmp/replay/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_name_json, context)
        mocked_file.assert_called_once_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test dump raising ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #3
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "test_template"
    valid_context = {"cookiecutter": {"project_name": "my_project"}, "other": "data"}
    invalid_context = {"not_cookiecutter": "data"}
    replay_file = tmp_path / f"{template_name}.json"

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(tmp_path, template_name)
    assert loaded_data == valid_context

    # Test Case 2: ValueError when 'cookiecutter' key is missing
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)

    # Test Case 3: FileNotFoundError when file does not exist
    non_existent_template = "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load(tmp_path, non_existent_template)

    # Test Case 4: Verify handling of template names already ending in .json
    json_template_name = "test_template.json"
    replay_json_file = tmp_path / "test_template.json"
    with open(replay_json_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_json = load(tmp_path, json_template_name)
    assert loaded_data_json == valid_context
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump(tmp_path):
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_dir = tmp_path / "replay"
    expected_file_path = replay_dir / "my_template.json"

    # Test successful dump
    dump(replay_dir, template_name, context)

    assert expected_file_path.exists()
    with open(expected_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == context

    # Test dump with missing 'cookiecutter' key
    invalid_context = {"not_cookiecutter": {}}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)

    # Test dump with existing .json extension in template name
    template_with_ext = "my_template.json"
    expected_file_path_ext = replay_dir / "my_template.json"
    dump(replay_dir, template_with_ext, context)
    assert expected_file_path_ext.exists()

@patch("builtins.open", new_callable=mock_open)
@patch("cookiecutter.replay.make_sure_path_exists")
def test_dump_calls_dependencies(mock_make_path, mock_file, tmp_path):
    template_name = "test"
    context = {"cookiecutter": {}}
    replay_dir = str(tmp_path)

    dump(replay_dir, template_name, context)

    mock_make_path.assert_called_once_with(replay_dir)
    mock_file.assert_called_once()
    
    # Verify json.dump was called via the file handle
    handle = mock_file()
    # Check if content written to file contains expected JSON structure
    written_content = "".join(call.args[0] for call in handle.write.call_args_list)
    assert '"cookiecutter"' in written_content
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }

    # Test case 1: Successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test case 2: Load failure due to missing 'cookiecutter' key
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

    # Test case 3: Load failure due to non-existent file
    non_existent_template = "missing_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test case 4: Ensure suffix handling works (passing name with .json)
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_explicit = load(replay_dir, f"{template_name}.json")
    assert loaded_data_explicit == valid_context
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    json_content = json.dumps(context, indent=2)

    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file:
        
        # Test successful dump
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        
        # Verify the content written to the file
        handle = mocked_file()
        # We combine all write calls to verify the full JSON string
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written_content) == context

    # Test failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "data"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)

    # Test template name with .json extension already present
    template_with_ext = "my_template.json"
    expected_file_path_ext = "/tmp/replay/my_template.json"
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file_ext:
        
        dump(replay_dir, template_with_ext, context)
        mocked_file_ext.assert_called_once_with(expected_file_path_ext, 'w', encoding="utf-8")
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = f"{replay_dir}/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_make_dir, \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_make_dir.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        m_json_dump.assert_called_once()
        # Verify the content passed to json.dump matches context
        args, _ = m_json_dump.call_args
        assert args[0] == context

    # Test dump with template name already including .json
    template_with_ext = "my_template.json"
    expected_file_path_ext = f"{replay_dir}/my_template.json"
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()), \
         patch("json.dump"):
        
        dump(replay_dir, template_with_ext, context)
        m_open.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump failure when 'cookiecutter' key is missing
    invalid_context = {"no_key": "value"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)

    # Test dump failure with Path object
    replay_dir_path = Path("/tmp/replay")
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()), \
         patch("json.dump"):
        
        dump(replay_dir_path, template_name, context)
        # Ensure path joining works correctly with Path object
        m_open.assert_called_with(os.path.join(replay_dir_path, "my_template.json"), 'w', encoding="utf-8")
```


# LLM-generated content at query #8
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = tmp_path / f"{template_name}.json"

    # Create the file for loading
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"] == {"project_name": "test_project"}

    # Test load with missing 'cookiecutter' key raises ValueError
    invalid_context = {"not_cookiecutter": "data"}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)

    # Test load with non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")

    # Test load works when template_name already has .json extension
    template_name_ext = "my_template.json"
    replay_file_ext = tmp_path / "my_template.json"
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    loaded_context_ext = load(tmp_path, template_name_ext)
    assert loaded_context_ext == context
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from pathlib import Path
import json

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {"project_name": "test_project"},
        "other_key": "value"
    }
    
    invalid_context = {
        "not_cookiecutter": "oops"
    }

    # 1. Test successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # 2. Test load with missing 'cookiecutter' key (ValueError)
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # 3. Test load when file does not exist (FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # 4. Test get_file_name integration (handling .json suffix)
    template_with_ext = "test.json"
    expected_path = str(replay_dir / "test.json")
    actual_path = get_file_name(replay_dir, template_with_ext)
    assert actual_path == expected_path
```


# LLM-generated content at query #10
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {"key": "value"}
    }

    # Test successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing 'cookiecutter' key (should raise ValueError)
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-lan") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")

    # Test load with non-existent file (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test get_file_name logic integration within load
    # Ensure .json suffix is handled correctly by the filename getter
    template_with_suffix = "test_template.json"
    replay_file_suffix = replay_dir / "test_template.json"
    with open(replay_file_suffix, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_suffix = load(replay_dir, template_with_suffix)
    assert loaded_data_suffix == valid_context
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    json_output = json.dumps(context, indent=2)

    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file:
        
        # Test successful dump
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        
        # Verify the content written to the file
        handle = mocked_file()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        assert json.loads(written_content) == context

    # Test dump with missing 'cookiecutter' key
    invalid_context = {"not_cookiecutter": "data"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)

    # Test dump with template name already containing .json
    template_with_ext = "my_template.json"
    expected_file_path_no_double_ext = "/tmp/replay/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file_ext:
        
        dump(replay_dir, template_with_ext, context)
        mocked_file_ext.assert_called_once_with(expected_file_path_no_double_ext, 'w', encoding="utf-8")
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_key": "value"
    }
    
    invalid_context = {
        "not_cookiecutter": "missing_key"
    }

    # 1. Test successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # 2. Test load with missing 'cookiecutter' key raises ValueError
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

    # 3. Test load with non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # 4. Test get_file_name logic integration (ensure suffix handling)
    template_with_ext = "template.json"
    expected_path = str(replay_dir / "template.json")
    # Note: load calls get_file_name internally, so we test if it handles .json extension correctly
    # by creating the file with the exact expected name.
    with open(replay_dir / "template.json", 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_with_ext)
    assert loaded_data_ext == valid_context
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }
    
    file_path = replay_dir / f"{template_name}.json"

    # Test successful load
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing 'cookiecutter' key raises ValueError
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test load with non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test load with existing .json suffix in template name
    json_template_name = "data.json"
    json_file_path = replay_dir / "data.json"
    with open(json_file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
        
    loaded_json_context = load(replay_dir, json_template_name)
    assert loaded_json_context == valid_context
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    json_content = json.dumps(context, indent=2)

    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir:
        with patch("builtins.open", mock_open()) as mocked_file:
            # Test successful dump
            dump(replay_dir, template_name, context)
            
            mock_mkdir.assert_called_once_with(replay_dir)
            mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
            
            # Verify the content written to the file
            handle = mocked_file()
            # Check if write was called (we check all calls because json.dump might call write multiple times)
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)
            assert json.loads(written_data) == context

    # Test dump with missing 'cookiecutter' key
    invalid_context = {"not_cookiecutter": "value"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)

    # Test dump with .json extension in template name
    template_with_ext = "my_template.json"
    expected_file_path_ext = "/tmp/replay/my_template.json"
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with patch("builtins.open", mock_open()):
            dump(replay_dir, template_with_ext, context)
            mocked_file.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = str(Path(replay_dir) / "my_template.json")
    json_content = json.dumps(context, indent=2)

    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file:
        
        # Test successful dump
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        
        # Verify the content written to the file
        handle = mocked_file()
        args, _ = handle.write.call_args
        # Since json.dump might call write multiple times, we check if the context is in the output
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        assert '"cookiecutter"' in written_data
        assert '"project_name": "test_project"' in written_data

    # Test ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": True}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)

    # Test suffix handling for template_name already ending in .json
    template_with_ext = "template.json"
    expected_file_path_ext = str(Path(replay_dir) / "template.json")
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()):
        dump(replay_dir, template_with_ext, context)
        mocked_file.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    # Test Case 1: Successful load
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Test Case 2: ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "oops"}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: FileNotFoundError when file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup directory and file name
    replay_dir = tmp_path / "replays"
    template_name = "my_template"
    expected_file_path = replay_dir / f"{template_name}.json"
    
    # Test Case 1: Successful load
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_dir.mkdir()
    with open(expected_template_path := expected_file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
        
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Test Case 2: Load fails if 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "oops"}
    with open(expected_file_path, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
        
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: Load fails if file does not exist
    non_existent_template = "non_existent"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test Case 4: Verify template_name with .json suffix works same as without
    with open(expected_file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_with_suffix = load(replay_dir, f"{template_name}.json")
    assert loaded_with_suffix == valid_context
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
import json
import os
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    # Create the file manually for loading
    file_name = f"{template_name}.json"
    replay_file = replay_dir / file_name
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load failure due to missing 'cookiecutter' key
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

    # Test load failure due to non-existent file
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "test_user"
        }
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }

    # Test Case 1: Successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load with missing 'cookiecutter' key raises ValueError
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: File does not exist raises FileNotFoundError
    non_existent_template = "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test Case 4: template_name already ends with .json
    json_template_name = "template.json"
    replay_file_json = replay_dir / "template.json"
    with open(replay_file_json, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_json_context = load(replay_dir, json_template_name)
    assert loaded_json_context == valid_context
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_key": "value"
    }
    
    invalid_context = {
        "not_cookiecutter": "data"
    }

    # Test case 1: Successful load
    file_path = replay_dir / f"{template_name}.json"
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test case 2: Load with missing 'cookiecutter' key raises ValueError
    invalid_file_path = replay_dir / "invalid_template.json"
    with open(invalid_file_path, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Test case 3: Load non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test case 4: Verify template name with .json suffix works correctly
    template_with_ext = "test_ext.json"
    with open(replay_dir / "test_ext.json", 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_with_ext)
    assert loaded_data_ext == valid_context
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

def test_dump(tmp_path):
    replay_dir = tmp_path / "replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}

    # Test successful dump
    dump(replay_dir, template_name, context)
    
    expected_file = replay_dir / "my_template.json"
    assert expected_file.exists()
    
    with open(expected_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == context

    # Test dump with template name already containing .json
    template_name_json = "my_template.json"
    dump(replay_dir, template_name_json, context)
    expected_file_json = replay_dir / "my_template.json"
    assert expected_file_json.exists()

    # Test ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)

    # Test using mock to verify file writing logic specifically
    mock_context = {"cookiecutter": {}}
    with patch("builtins.open", mock_open()) as mocked_file:
        with patch("cookiecutter.replay.make_sure_path_exists"):
            dump("/fake/dir", "test", mock_context)
            mocked_file.assert_called_once_with(
                os.path.join("/fake/dir", "test.json"), "w", encoding="utf-8"
            )
            # Verify json.dump was called via the written content
            handle = mocked_file()
            args, _ = handle.write.call_args
            assert '"cookiecutter"' in args[0]
```


# LLM-generated content at query #22
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    invalid_context = {"no_cookiecutter_key": True}
    
    replay_file = replay_dir / f"{template_name}.json"

    # Test successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Test ValueError when 'cookiecutter' key is missing
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

    # Test FileNotFoundError when file does not exist
    non_existent_template = "missing"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)
```


# LLM-generated content at query #23
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "test_template"
    context = {"cookiecutter": {"project_name": "my_project"}, "other": "data"}
    replay_file = tmp_path / f"{template_name}.json"
    
    # Create dummy file
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"] == {"project_name": "my_project"}

    # Test load with invalid data (missing cookiecutter key)
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)

    # Test load for non-existent file
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #24
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = tmp_path / f"{template_name}.json"
    
    # Create dummy file
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing cookiecutter key (should raise ValueError)
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

    # Test load with non-existent file (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #25
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_key": "value"
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }

    # Test successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Test load with missing cookiecutter key (ValueError)
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'template_name_error.json" if False else invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Test load with non-existent file (FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test file name suffix logic (ensuring .json is handled)
    suffix_template = "test_with_extension.json"
    suffix_file = replay_dir / f"{suffix_template}"
    with open(suffix_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_suffix_context = load(replay_dir, suffix_template)
    assert loaded_suffix_context == valid_context
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = os.path.join(replay_dir, "my_template.json")
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test dump with template name already having .json extension
    template_name_json = "my_template.json"
    expected_file_path_json = os.path.join(replay_dir, "my_template.json")
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_name_json, context)
        mocked_file.assert_called_once_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test dump with missing 'cookiecutter' key in context
    invalid_context = {"not_cookiecutter": "data"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #27
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }
    
    replay_file = replay_dir / f"{template_name}.json"

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load fails if 'cookiecutter' key is missing
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: Load fails if file does not exist
    non_existent_template = "ghost_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test Case 4: Verify suffix handling (template name with .json)
    template_with_ext = "my_template.json"
    replay_file_ext = replay_dir / "my_template.json"
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context_ext = load(replay_dir, template_with_ext)
    assert loaded_context_ext == valid_context
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
import json
import os
from pathlib import Path

def test_load(tmp_path):
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_file = replay_dir / f"{template_name}.json"
    
    # Valid context case
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_dir.mkdir()
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    # Invalid context case (missing 'cookiecutter' key)
    invalid_context = {"not_cookiecutter": "data"}
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'wrap_encoding="utf-8"') as f: # Note: logic check below
        pass 
    # Re-writing properly for the test
    with open(replay_dir / "invalid_template.json", 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    # File does not exist case
    non_existent_file = replay_dir / "missing.json"

    # Assertions
    # 1. Test successful load
    assert load(replay_dir, template_name) == valid_context

    # 2. Test ValueError when 'cookiecutter' key is missing
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")

    # 3. Test FileNotFoundError when file doesn't exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "missing")

    # 4. Test suffix handling (template name without .json)
    # Ensure get_file_name logic is covered via load
    assert load(replay_dir, f"{template_name}.json") == valid_context
```


# LLM-generated content at query #29
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    context = {
        "cookiecutter": {
            "project_name": "test_project",
            "author": "tester"
        },
        "other_key": "value"
    }
    
    # Create the replay file manually for testing load
    replay_file = tmp_path / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing 'cookiecutter' key raises ValueError
    invalid_context = {"no_cookiecutter_here": True}
    invalid_file = tmp_path / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, "invalid_template")

    # Test load with non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    
    # Test Case 1: Successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_make_dir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_make_dir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test Case 2: Template name already has .json suffix
    template_with_ext = "my_template.json"
    expected_file_path_ext = "/tmp/replay/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_with_ext, context)
        mocked_file.assert_called_once_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test Case 3: Missing 'cookiecutter' key raises ValueError
    invalid_context = {"not_cookiecutter": "data"}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #31
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    expected_file_path = "/tmp/replay/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once()
        # Verify the content passed to json.dump is our context
        args, _ = mock_json_dump.call_args
        assert args[0] == context

    # Test dump with template name already having .json extension
    template_with_ext = "my_template.json"
    expected_file_path_ext = "/tmp/replay/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_with_ext, context)
        mocked_file.assert_called_once_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump raising ValueError when cookiecutter key is missing
    invalid_context = {"not_cookiecutter": {}}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #32
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "tester"
        }
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }
    
    replay_file = replay_dir / f"{template_name}.json"

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "my_project"

    # Test Case 2: Load fails due to missing 'cookiecutter' key
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: Load fails due to non-existent file
    non_existent_template = "missing_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test Case 4: Verify suffix handling (template name with .json)
    template_with_ext = "template.json"
    with open(replay_dir / template_with_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_with_ext)
    assert loaded_data_ext == valid_context
```


# LLM-generated content at query #33
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    
    valid_context = {"cookiecutter": {"project_name": "my_project"}, "other": "data"}
    invalid_context = {"no_cookiecutter_key": True}
    
    replay_file = replay_dir / f"{template_name}.json"

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Test Case 2: Load fails when 'cookiecutter' key is missing
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: Load fails when file does not exist
    non_existent_template = "non_existent"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest
import json
import os
from pathlib import Path

def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {
            "project_name": "my_project",
            "author": "test_user"
        },
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }

    replay_file = os.path.join(str(replay_dir), f"{template_name}.json")

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dict := str(replay_dir), template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "my_project"

    # Test Case 2: Raise ValueError when 'cookiecutter' key is missing
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: Raise FileNotFoundError when file does not exist
    non_existent_template = "non_existent"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test Case 4: Ensure suffix logic works (template name with .json extension)
    template_with_ext = "template.json"
    replay_file_ext = os.path.join(str(replay_dir), "template.json")
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_with_ext)
    assert loaded_data_ext == valid_context
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = tmp_path / f"{template_name}.json"

    # Test successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context

    # Test load with .json extension in template name
    template_with_ext = "my_template.json"
    replay_file_ext = tmp_path / "my_template.json"
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    loaded_context_ext = load(tmp_path, template_with_ext)
    assert loaded_context_ext == context

    # Test ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": True}
    replay_file_invalid = tmp_path / "invalid.json"
    with open(replay_file_invalid, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, "invalid")

    # Test FileNotFoundError when file does not exist
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


