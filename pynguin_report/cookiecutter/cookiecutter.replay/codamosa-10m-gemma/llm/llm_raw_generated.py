####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pathlib import Path

def test_get_file_name():
    replay_dir = "/tmp/replay"
    
    # Test with template name that doesn't have .json suffix
    template_no_suffix = "my_template"
    expected_no_suffix = "/tmp/replay/my_template.json"
    assert get_file_name(replay_dir, template_no_suffix) == expected_no_suffix

    # Test with template name that already has .json suffix
    template_with_suffix = "my_template.json"
    expected_with_suffix = "/tmp/template.json" # Wait, logic check: 
    # Logic: suffix = '.json' if not endswith('.json') else ''
    # If template_name is "my_template.json", suffix is ""
    # Result should be "/tmp/replay/my_template.json"
    assert get_file_name(replay_dir, template_with_suffix) == "/tmp/replay/my_template.json"

    # Test with Path object
    replay_path = Path("/tmp/replay")
    assert get_file_name(replay_path, "test") == str(Path("/tmp/replay/test.json"))

    # Test edge case: empty string
    assert get_file_name(replay_dir, "") == "/tmp/replay/.json"
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pathlib import Path

def test_get_file_name():
    replay_dir = "/tmp/replay"
    
    # Case 1: template_name does not have .json suffix
    template_no_suffix = "my_template"
    expected_no_suffix = os.path.join(replay_dir, "my_template.json")
    assert get_file_name(replay_dir, template_no_suffix) == expected_no_suffix

    # Case 2: template_name already has .json suffix
    template_with_suffix = "config.json"
    expected_with_suffix = os.path.join(replay_dir, "config.json")
    assert get_file_name(replay_dir, template_with_suffix) == expected_with_suffix

    # Case 3: Using Path object instead of string
    replay_dir_path = Path("/tmp/replay")
    expected_path_obj = os.path.join(replay_dir_path, "test.json")
    assert get_file_name(replay_dir_path, "test") == expected_path_obj
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from pathlib import Path

def test_get_file_name():
    replay_dir = "/tmp/replay"
    
    # Test case 1: template name without .json suffix
    template_no_suffix = "my_template"
    expected_path_1 = "/tmp/replay/my_template.json"
    assert get_file_name(replay_dir, template_no_suffix) == expected_path_1

    # Test case 2: template name with .json suffix
    template_with_suffix = "my_template.json"
    expected_path_2 = "/tmp/regex/my_template.json".replace("/regex/", "/replay/") # ensuring path logic
    expected_path_2 = "/tmp/replay/my_template.json"
    assert get_file_name(replay_dir, template_with_suffix) == expected_path_2

    # Test case 3: Using Path object instead of string
    replay_path_obj = Path("/tmp/replay")
    expected_path_3 = "/tmp/replay/test.json"
    assert get_file_name(replay_path_obj, "test") == expected_path_3

    # Test case 4: Empty template name
    assert get_file_name(replay_dir, "") == "/tmp/replay/.json"
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pathlib import Path

def test_get_file_name():
    replay_dir = "/tmp/replays"
    
    # Test case 1: template name without .json suffix
    template_no_suffix = "my_template"
    expected_1 = "/tmp/replays/my_template.json"
    assert get_file_name(replay_dir, template_no_suffix) == expected_1

    # Test case 2: template name already has .json suffix
    template_with_suffix = "config.json"
    expected_2 = "/tmp/replays/config.json"
    assert get_file_name(replay_dir, template_with_suffix) == expected_2

    # Test case 3: using Path object instead of string
    replay_path_obj = Path("/tmp/replays")
    expected_3 = "/tmp/replays/test.json"
    assert get_file_name(replay_path_obj, "test") == expected_3

    # Test case 4: empty template name (should still append .json)
    assert get_file_name(replay_dir, "") == "/tmp/replays/.json"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from pathlib import Path

def test_get_file_name():
    replay_dir = "/tmp/replay"
    
    # Test case 1: template name without .json suffix
    template_no_ext = "my_template"
    expected_1 = "/tmp/replay/my_template.json"
    assert get_file_name(replay_dir, template_no_ext) == expected_1

    # Test case 2: template name with .json suffix
    template_with_ext = "my_template.json"
    expected_2 = "/tmp/lag/my_template.json" # Note: get_file_name uses os.path.join
    # Re-calculating expected based on logic: 
    # suffix is '' if ends with .json. file_name = 'my_template.json' + ''
    assert get_file_name(replay_dir, template_with_ext) == "/tmp/replay/my_template.json"

    # Test case 3: Using Path object instead of string
    replay_path = Path("/tmp/replay")
    expected_3 = str(Path("/tmp/replay/other_template.json"))
    assert get_file_name(replay_path, "other_template") == expected_3

    # Test case 4: template name that is just an extension
    assert get_file_name(replay_dir, ".json") == "/tmp/replay/.json"
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
    expected_file_path = f"{replay_dir}/my_template.json"

    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)

        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test dump with existing .json extension in template name
    template_name_json = "my_template.json"
    expected_file_path_json = f"{replay_dir}/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_name_json, context)
        mocked_file.assert_called_once_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test dump failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #7
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

    # Test Case 1: Successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load fails when 'cookiecutter' key is missing
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")

    # Test Case 3: Load fails when file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test Case 4: Verify template name with .json suffix handling
    template_name_with_ext = "test_suffix.json"
    replay_file_ext = replay_dir / "test_suffix.json"
    with open(replay_file_ext, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_name_with_ext)
    assert loaded_data_ext == valid_context
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import json
import os
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
        "no_cookiecutter_key": True
    }

    # Test Case 1: Successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load fails due to missing 'cookiecutter' key
    invalid_file = replay_dir / "invalid.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid")

    # Test Case 3: Load fails because file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test Case 4: Verify template name with .json extension works correctly
    template_name_with_ext = "template.json"
    replay_file_ext = replay_dir / "template.json"
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_name_with_ext)
    assert loaded_data_ext == valid_context
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
    json_content = json.dumps(context, indent=2)

    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_exists, \
         patch("builtins.open", mock_open()) as mocked_file:
        
        dump(replay_dir, template_name, context)
        
        mock_exists.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        
        # Check if json.dump was called with correct content
        handle = mocked_file()
        # We verify the write call occurred. Since json.dump calls write multiple times, 
        # we check if the context is present in the arguments.
        args, _ = handle.write.call_args
        assert json.loads(args[0]) == context

    # Test dump with template name already containing .json
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()):
        
        dump(replay_dir, "existing.json", context)
        expected_json_path = f"{replay_dir}/existing.json"
        # Verify the file path logic via get_file_name side effect in dump
        args, _ = patch("builtins.open", mock_open()).call_args_list[0][0]
        # Note: In a real test environment, we'd inspect the actual path passed to open

    # Test dump failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "data"}
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, invalid_context)

    # Test dump failure when directory creation fails (simulating permission error)
    with patch("cookiecutter.replay.make_sure_path_exists", side_effect=OSError):
        with pytest.raises(OSError):
            dump(replay_dir, template_name, context)
```


# LLM-generated content at query #10
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
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"

    # Test load with missing 'cookiecutter' key raises ValueError
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

    # Test load with non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test get_file_name logic integration (suffix handling)
    template_with_ext = "test_template.json"
    expected_path = str(replay_dir / "test_template.json")
    # Since load calls get_file_name, we verify the path resolution works via the file creation
    with open(replay_dir / "test_template.json", "w", encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_ext_context = load(replay_dir, template_with_ext)
    assert loaded_ext_context == valid_context
```


# LLM-generated content at query #11
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
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing cookiecutter key (ValueError)
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test load with non-existent file (FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test get_file_name logic integration (handling .json suffix)
    replay_file_with_ext = replay_dir / "template.json"
    with open(replay_file_with_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_ext_context = load(replay_dir, "template.json")
    assert loaded_ext_context == valid_context
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    
    valid_context = {
        "cookiecutter": {"project_name": "my_project"},
        "other_data": 123
    }
    
    invalid_context = {
        "not_cookiecutter": "oops"
    }

    # Test case 1: Successful load
    file_path = replay_dir / f"{template_name}.json"
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "my_project"

    # Test case 2: Load with .json extension in name (should not double suffix)
    template_with_ext = "test_template.json"
    file_path_ext = replay_dir / template_with_ext
    with open(file_path_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_with_ext)
    assert loaded_data_ext == valid_context

    # Test case 3: Raise ValueError if 'cookiecutter' key is missing
    file_path_invalid = replay_dir / "invalid.json"
    with open(file_path_invalid, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid")

    # Test case 4: Raise FileNotFoundError if file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #13
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
        "other_data": 123
    }
    
    # Create replay directory and file
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_file = get_file_name(replay_dir, template_name)
    
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == context
    assert "cookiecutter" in loaded_context

    # Test load with missing 'cookiecutter' key
    invalid_context = {"no_key": "here"}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test load when file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #14
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
    
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    invalid_context = {"not_cookiecutter": "data"}
    
    replay_file = replay_dir / f"{template_name}.json"

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Raise ValueError if 'cookiecutter' key is missing
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: Raise FileNotFoundError if file does not exist
    non_existent_template = "missing_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test Case 4: Verify suffix handling (template name with .json extension)
    template_with_ext = "template.json"
    with open(replay_dir / "template.json", 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_with_ext)
    assert loaded_data_ext == valid_context
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
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_make_dir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)

        mock_make_dir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test dump with template name already containing .json
    template_with_ext = "my_template.json"
    expected_file_path_ext = f"{replay_dir}/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()), \
         patch("json.dump"):
        
        dump(replay_dir, template_with_ext, context)
        # Verify the logic inside get_file_name used in dump
        from cookiecutter.replay import get_file_name
        assert get_file_name(replay_dir, template_with_ext) == expected_file_path_ext

    # Test dump failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #16
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

    replay_file = replay_dir / f"{template_name}.json"

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load fails if 'cookiecutter' key is missing
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: Load fails if file does not exist
    non_existent_template = "missing_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test Case 4: Verify filename logic (handling .json suffix)
    template_with_suffix = "test_template.json"
    replay_file_suffix = replay_dir / "test_template.json"
    with open(replay_file_suffix, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_suffix = load(replay_dir, template_with_suffix)
    assert loaded_data_suffix == valid_context
```


# LLM-generated content at query #17
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
    replay_file = tmp_path / f"{template_name}.json"
    
    # Create dummy file for loading
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing cookiecutter key (should raise ValueError)
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)

    # Test load with non-existent file (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
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
    file_path = replay_dir / f"{template_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing 'cookiecutter' key raises ValueError
    invalid_file_path = replay_dir / "invalid_template.json"
    with open(invalid_file_path, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")

    # Test load with non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test load handles template_name with .json suffix correctly
    template_name_with_ext = "complete_name.json"
    file_path_ext = replay_dir / "complete_name.json"
    with open(file_path_ext, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context_ext = load(replay_dir, template_name_with_ext)
    assert loaded_context_ext == valid_context
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
    expected_file_path = f"{replay_dir}/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json:
        
        from cookiecutter.replay import dump
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        m_json.assert_called_once()
        # Verify the content passed to json.dump is our context
        args, _ = m_json.call_args
        assert args[0] == context

    # Test dump with missing 'cookiecutter' key in context
    invalid_context = {"not_cookiecutter": True}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        from cookiecutter.replay import dump
        dump(replay_dim, template_name, invalid_context)

    # Test dump with template name already containing .json
    template_with_ext = "my_template.json"
    expected_file_path_ext = f"{replay_dir}/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump"):
        
        from cookiecutter.replay import dump
        dump(replay_dir, template_with_ext, context)
        m_open.assert_called_once_with(expected_file_path_ext, 'w', encoding="utf-8")
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
    expected_file_path = "/tmp/replay/my_template.json"
    json_content = json.dumps(context, indent=2)

    with patch("cookiecutter.replay.make_sure_path_exists") as mock_make_dir, \
         patch("builtins.open", mock_open()) as mocked_file:
        
        # Test successful dump
        dump(replay_dir, template_name, context)
        
        mock_make_dir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        
        # Verify content written (checking the call to write)
        handle = mocked_file()
        # We check if any call to write contained our JSON string
        args, _ = handle.write.call_args
        assert json.loads(handle.write.call_args_list[0][0][0]) == context

    # Test ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "data"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)

    # Test suffix handling for template_name already ending in .json
    template_name_json = "my_template.json"
    expected_file_path_json = "/tmp/replay/my_template.json"
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()):
        dump(replay_dir, template_name_json, context)
        mocked_file.assert_called_with(expected_file_path_json, 'w', encoding="utf-8")
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "test_template"
    replay_dir = tmp_path / "replays"
    valid_context = {"cookiecutter": {"project_name": "my_project"}, "other": "data"}
    invalid_context = {"not_cookiecutter": "oops"}
    
    replay_dir.mkdir()
    file_path = replay_dir / f"{template_name}.json"

    # Test 1: Successful load
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "my_project"

    # Test 2: Load fails if 'cookiecutter' key is missing
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

    # Test 3: Load fails if file does not exist
    non_existent_template = "missing"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test 4: Verify suffix handling (template_name with .json)
    template_with_ext = "test_template.json"
    with open(replay_dir / template_with_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_with_ext)
    assert loaded_data_ext == valid_context
```


# LLM-generated content at query #22
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

    # Test load with existing .json suffix in name
    template_with_ext = "my_template.json"
    replay_file_ext = tmp_path / "my_template.json"
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    loaded_context_ext = load(tmp_path, template_with_ext)
    assert loaded_context_ext == context

    # Test load failure due to missing cookiecutter key
    invalid_context = {"no_key": "here"}
    replay_file_invalid = tmp_path / "invalid.json"
    with open(replay_file_invalid, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, "invalid")

    # Test load failure due to non-existent file
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #23
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

    # Test load with invalid content (missing cookiecutter key)
    invalid_context = {"no_key": "here"}
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, 'none', encoding="utf-8") as f: # This is a placeholder logic for the test setup
        pass 
    # Correct way to setup invalid file for testing:
    with open(tmp_path / "invalid.json", 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
        
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, "invalid")

    # Test load with non-existent file
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #24
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
    template_name_json = "my_template.json"
    expected_file_path_json = "/tmp/replay/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()), \
         patch("json.dump"):
        
        dump(replay_dir, template_name_json, context)
        # Verify the logic in get_file_name via the path passed to open
        # (The second call to open would use the same logic)

    # Test dump raising ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)

    # Test dump raising ValueError when 'cookiecutter' key is missing in empty context
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecall key'):
            dump(replay_dir, template_name, {})
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
import json
import os
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
    file_path = os.path.join(str(replay_dir), f"{template_name}.json")
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert "cookiecutter" in loaded_context

    # Test case 2: Load fails when 'cookiecutter' key is missing
    invalid_file_path = os.path.join(str(replay_dir), "invalid_template.json")
    with open(invalid_file_path, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Test case 3: Load fails when file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #26
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    template_name = "test_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {"cookiecutter": {"project_name": "my_project"}, "other": "data"}
    invalid_context = {"not_cookiecutter": {}}
    
    # Test successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Test load with .json extension in name
    template_with_ext = "test_template.json"
    replay_file_ext = replay_dir / "test_template.json"
    with open(replay_file_ext, 'ω', encoding="utf-8") as f:
        # Note: Using a separate file to ensure get_file_name logic works
        pass 
    
    # Re-run with explicit .json name
    with open(replay_dir / "test_template.json", 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    assert load(replay_dir, "test_template.json") == valid_context

    # Test load with missing cookiecutter key (ValueError)
    invalid_file = replay_dir / "invalid.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid")

    # Test load with non-existent file (FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
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
    expected_file_path = f"{replay_dir}/my_template.json"
    
    # Test successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_make_dir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_make_dir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        # Check if json.dump was called with correct context
        args, _ = mock_json_dump.call_args
        assert args[0] == context

    # Test dump with template name already containing .json
    template_with_ext = "my_template.json"
    expected_file_path_ext = f"{replay_dir}/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()), \
         patch("json.dump"):
        
        dump(replay_dir, template_with_ext, context)
        mocked_file = patch("builtins.open", mock_open()).start()
        mocked_file.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "oops"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #28
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
    
    # Create dummy replay file
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing cookiecutter key (should raise ValueError)
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(tmp_path, template_name)

    # Test load when file does not exist (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #29
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
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_make_dir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_make_dir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test dump with template_name already having .json extension
    template_with_ext = "my_template.json"
    expected_file_path_ext = f"{replay_dir}/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_with_ext, context)
        mocked_file.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump raising ValueError when cookiecutter key is missing
    invalid_context = {"not_cookiecutter": {}}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #30
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
    json_content = json.dumps(context, indent=2)

    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir:
        with patch("builtins.open", mock_open()) as mocked_file:
            # Test successful dump
            dump(replay_dir, template_name, context)
            
            mock_mkdir.assert_called_once_with(replay_dir)
            mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
            
            # Verify the content written to the file
            handle = mocked_file()
            args, _ = handle.write.call_args
            # Since json.dump might call write multiple times, we check if the final output matches
            # In a real mock_open scenario for json.dump, it's easier to verify via side effects or 
            # by intercepting the string buffer.
    
    # Test dump with missing 'cookiecutter' key
    invalid_context = {"not_cookiecutter": "oops"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_file=replay_dir, template_name=template_name, context=invalid_context)

    # Test dump with .json extension already in template_name
    template_with_ext = "config.json"
    expected_file_path_ext = f"{replay_dir}/config.json"
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with patch("builtins.open", mock_open()):
            dump(replay_dir, template_with_ext, context)
            mocked_file.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")
```


# LLM-generated content at query #31
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
    invalid_context = {"not_cookiecutter": "error"}
    
    # Test successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    
    # Test load with missing cookiecutter key raises ValueError
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)
        
    # Test load with non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test load with template name already containing .json
    template_with_ext = "my_template.json"
    replay_file_ext = replay_dir / template_with_ext
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context_ext = load(replay_dir, template_with_ext)
    assert loaded_context_ext == valid_context
```


# LLM-generated content at query #32
#--------------------------

```python
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
        "other_key": "value"
    }
    
    invalid_context = {
        "not_cookiecutter": {}
    }

    # Test Case 1: Successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load failure due to missing 'cookiecutter' key
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Test Case 3: Load failure due to non-existent file
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test Case 4: Verify suffix handling (template name with .json)
    template_with_ext = "explicit_name.json"
    with open(replay_dir / "explicit_name.json", 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_with_ext)
    assert loaded_data_ext == valid_context
```


# LLM-generated content at query #33
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
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_exists, \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json:
        
        dump(replay_dir, template_name, context)
        
        mock_exists.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        m_json.assert_called_once()
        # Verify the content passed to json.dump is our context
        args, _ = m_json.call_args
        assert args[0] == context

    # Test dump with template name already containing .json
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()), \
         patch("json.dump"):
        
        template_with_ext = "my_template.json"
        expected_file_path_ext = f"{replay_dir}/my_template.json"
        dump(replay_dir, template_with_ext, context)
        
        m_open.assert_called_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "value"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    invalid_context = {"no_cookiecutter_key": True}
    replay_file = tmp_path / f"{template_name}.json"

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(tmp_path, template_name)
    assert loaded_data == valid_context

    # Test Case 2: Load fails when 'cookiecutter' key is missing
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)

    # Test Case 3: Load fails when file does not exist
    non_existent_template = "missing_template"
    with pytest.raises(FileNotFoundError):
        load(tmp_path, non_existent_template)

    # Test Case 4: Verify it works with .json extension in template name (no double suffix)
    json_template_name = "test.json"
    json_replay_file = tmp_path / "test.json"
    with open(json_replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_json = load(tmp_path, json_template_name)
    assert loaded_data_json == valid_context
```


# LLM-generated content at query #35
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

@pytest.mark.parametrize(
    "replay_dir, template_name, context, expected_path",
    [
        ("/tmp/replay", "my_template", {"cookiecutter": {"key": "val"}}, "/tmp/replay/my_template.json"),
        ("/tmp/replay", "my_template.json", {"cookiecutter": {"key": "val"}}, "/tmp/replay/my_template.json"),
        (Path("/tmp/replay"), "my_template", {"cookiecutter": {}}, "/tmp/replay/my_template.json"),
    ],
)
def test_dump(replay_dir, template_name, context, expected_path):
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir:
        with patch("builtins.open", mock_open()) as m_open:
            with patch("json.dump") as m_json_dump:
                from cookiecutter.replay import dump

                dump(replay_dir, template_name, context)

                mock_mkdir.assert_called_once_with(replay_dir)
                m_open.assert_called_once_with(expected_path, "w", encoding="utf-8")
                m_json_dump.assert_called_once_with(context, m_open(), indent=2)

def test_dump_missing_cookiecutter_key():
    from cookiecutter.replay import dump
    
    invalid_context = {"not_cookiecutter": {}}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump("/tmp/replay", "template", invalid_context)
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {"cookiecutter": {"project_name": "my_project"}, "other": "data"}
    invalid_context = {"not_cookiecutter": "data"}
    
    file_path = replay_dir / f"{template_name}.json"

    # Test 1: Successful load
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Test 2: Load fails if 'cookiecutter' key is missing
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test 3: Load fails if file does not exist
    non_existent_template = "missing"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pathlib import Path
import json

def test_load(tmp_path):
    # Setup data
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

    # Test successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"

    # Test load with missing 'cookiecutter' key raises ValueError
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Test load with non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test load with template_name already containing .json extension
    json_template_name = "already_has_extension.json"
    with open(replay_dir / json_template_name, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_json_context = load(replay_dir, json_template_name)
    assert loaded_json_context == valid_context
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

@pytest.mark.parametrize(
    "replay_dir, template_name, context, expected_path",
    [
        ("/tmp/replay", "my_template", {"cookiecutter": {"project": "test"}}, "/tmp/replay/my_template.json"),
        ("/tmp/replay", "my_template.json", {"cookiecutter": {}}, "/tmp/replay/my_template.json"),
    ],
)
def test_dump(replay_dir, template_name, context, expected_path):
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json_dump:
        
        from cookiecutter.replay import dump
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_path, 'w', encoding="utf-8")
        m_json_dump.assert_called_once_with(context, m_open(), indent=2)

def test_dump_missing_cookiecutter_key():
    from cookiecutter.replay import dump
    
    invalid_context = {"not_cookiecutter": {}}
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump("/tmp/replay", "template", invalid_context)
```


# LLM-generated content at query #4
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
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        m_json_dump.assert_called_once_with(context, m_open(), indent=2)

    # Test dump with template name already having .json extension
    template_name_json = "my_template.json"
    expected_file_path_json = f"{replay_dir}/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump"):
        
        dump(replay_dir, template_name_json, context)
        m_open.assert_called_once_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test dump failure when 'cookiecutter' key is missing in context
    invalid_context = {"not_cookiecutter": {}}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #5
#--------------------------

```python
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_dump(tmp_path):
    # Setup data
    replay_dir = tmp_path / "replays"
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other_key": 123}

    # Test successful dump
    dump(replay_dir, template_name, context)

    # Verify file existence and content
    expected_file = replay_dir / "my_template.json"
    assert expected_file.exists()
    
    with open(expected_file, 'r', encoding="utf-8") as f:
        loaded_data = json.load(f)
    assert loaded_data == context

    # Test dump with template name already containing .json
    template_name_json = "my_template.json"
    dump(replay_dir, template_name_json, context)
    expected_file_json = replay_dir / "my_template.json"
    assert expected_file_json.exists()

    # Test ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump(replay_dir, template_name, invalid_context)

    # Test with mock to verify make_sure_path_exists is called
    with patch('cookiecutter.replay.make_sure_path_exists') as mock_make_exists:
        dump(replay_dir, template_name, context)
        mock_make_exists.assert_called_once_with(replay_dir)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "test_template"
    replay_dir = tmp_path / "replays"
    replay_file = replay_dir / f"{template_name}.json"
    
    valid_context = {"cookiecutter": {"project_name": "my_project"}, "other": "data"}
    invalid_context = {"not_cookiecutter": True}

    # Create directory
    replay_dir.mkdir()

    # Test case 1: Successful load
    with open(replay_template_file := replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context

    # Test case 2: Load failure due to missing 'cookiecutter' key
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test case 3: Load failure due to non-existent file
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from pathlib import Path
import json

def test_load(tmp_path):
    # Setup
    template_name = "test_template"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    
    valid_context = {"cookiecutter": {"project_name": "my_project"}, "other": "data"}
    invalid_context = {"not_cookiecutter": "oops"}
    file_path = replay_dir / f"{template_name}.json"

    # Test 1: Successful load
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context

    # Test 2: Load fails if 'cookiecutter' key is missing
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test 3: Load fails if file does not exist
    non_existent_template = "missing"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

@pytest.mark.parametrize(
    "replay_dir, template_name, context, expected_path, should_raise_error",
    [
        (
            "/tmp/replay",
            "my_template",
            {"cookiecutter": {"project_name": "test"}},
            "/tmp/replay/my_template.json",
            False,
        ),
        (
            "/tmp/replay",
            "my_template.json",
            {"cookiecutter": {"project_name": "test"}},
            "/tmp/replay/my_template.json",
            False,
        ),
        (
            "/tmp/replay",
            "my_template",
            {"not_cookiecutter": "data"},
            None,
            True,
        ),
    ],
)
def test_dump(replay_dir, template_name, context, expected_path, should_raise_error):
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_make_dir:
        if should_raise_error:
            with pytest.raises(ValueError, match="Context is required to contain a cookiecard key" if "cookiecutter" not in context else "Context is required to contain a cookiecutter key"):
                # The error message in the source code is 'Context is required to contain a cookiecutter key'
                # Note: there is a typo in the original source code provided (msg uses 'cookiecutter')
                # I will match the actual string in the provided code.
                dump(replay_dir, template_name, context)
        else:
            m = mock_open()
            with patch("builtins.open", m):
                import json
                dump(replay_dir, template_name, context)
                
                mock_make_dir.assert_called_once_with(replay_dir)
                m.assert_called_once_with(expected_path, "w", encoding="utf-8")
                
                # Verify json.dump was called with correct data
                handle = m()
                args, _ = json.dump.call_args if hasattr(json, 'dump') else (None, None) 
                # Since we can't easily intercept json.dump arguments without complex patching, 
                # we verify the call to open and the logic flow.
```

Wait, I should provide a cleaner, more standard version of `test_dump` that follows best practices for testing this specific function:

```python
import pytest
from unittest.mock import patch, mock_open
import json

def test_dump():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"cookiecutter": {"name": "test"}}
    expected_file_path = "/tmp/replay/my_template.json"

    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)

        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, "w", encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

def test_dump_missing_cookiecutter_key():
    replay_dir = "/tmp/replay"
    template_name = "my_template"
    context = {"wrong_key": "data"}

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        dump(replay_dir, template_name, context)

def test_dump_handles_json_extension_correctly():
    replay_dir = "/tmp/replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {}}
    expected_file_path = "/tmp/replay/my_template.json"

    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_name, context)
        mocked_file.assert_called_once_with(expected_file_path, "w", encoding="utf-8")
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

@pytest.mark.parametrize(
    "replay_dir, template_name, context, expected_path",
    [
        ("/tmp/replays", "my_template", {"cookiecutter": {"key": "val"}}, "/tmp/replays/my_template.json"),
        ("/tmp/replays", "my_template.json", {"cookiecutter": {"key": "val"}}, "/tmp/replays/my_template.json"),
        (Path("/tmp/replays"), "my_template", {"cookiecutter": {}}, "/tmp/replays/my_template.json"),
    ],
)
def test_dump(replay_dir, template_name, context, expected_path):
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json:
        
        dump(replay_dir, template_name, context)

        mock_mkdir.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_path, 'w', encoding="utf-8")
        m_json.assert_called_once_with(context, m_open(), indent=2)

def test_dump_raises_value_error_on_missing_key():
    invalid_context = {"not_cookiecutter": {}}
    replay_dir = "/tmp/replays"
    template_name = "test"

    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #10
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
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"

    # Case 2: Load fails when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
        
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Case 3: Load fails when file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Case 4: Verify template name with .json extension works correctly
    template_name_with_ext = "suffix_test.json"
    with open(replay_dir / "suffix_test.json", 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_ext_context = load(replay_dir, template_name_with_ext)
    assert loaded_ext_context == valid_context
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
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_make_dir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_make_dir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once()
        # Verify context passed to json.dump is correct
        args, _ = mock_json_dump.call_args
        assert args[0] == context

    # Test dump with template name already containing .json
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()), \
         patch("json.dump"):
        
        template_with_ext = "my_template.json"
        expected_file_path_ext = "/tmp/replay/my_template.json"
        
        # We check get_file_name logic via dump's side effect on open
        with patch("os.path.join", return_value=expected_file_path_ext):
            dump(replay_dir, template_with_ext, context)
            # Verification is implicit in the call to open with the correct path

    # Test dump failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)
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
    json_output = json.dumps(context, indent=2)

    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir:
        with patch("builtins.open", mock_open()) as mocked_file:
            # Test successful dump
            dump(replay_dir, template_name, context)
            
            mock_mkdir.assert_called_once_with(replay_dir)
            mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
            
            # Verify content written to file
            handle = mocked_file()
            # Collect all calls to write to check if the json was dumped correctly
            written_content = "".join(call.args[0] for call in handle.write.call_args_list)
            assert json.loads(written_content) == context

    # Test dump with missing 'cookiecutter' key
    invalid_context = {"not_cookiecutter": "data"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)

    # Test dump with .json extension already in template name
    template_with_ext = "my_template.json"
    expected_file_path_ext = "/tmp/replay/my_template.json"
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with patch("builtins.open", mock_open()) as mocked_file_ext:
            dump(replay_dir, template_with_ext, context)
            mocked_file_ext.assert_called_once_with(expected_file_path_ext, 'w', encoding="utf-8")
```


# LLM-generated content at query #13
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

    # Test successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"] == {"project_name": "test_project"}

    # Test load with missing cookiecutter key raises ValueError
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, 'arg', encoding="utf-8") as f: # overwriting existing
        pass 
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)

    # Test load with non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
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

    # Test case 1: Successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)

        mock_mkdir.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once_with(context, mocked_file(), indent=2)

    # Test case 2: Template name already has .json
    template_name_json = "my_template.json"
    expected_file_path_json = "/tmp/replay/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_name_json, context)
        mocked_file.assert_called_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test case 3: Missing 'cookiecutter' key in context
    invalid_context = {"not_cookiecutter": "data"}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #15
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
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "my_project"

    # Test load with missing 'cookiecutter' key raises ValueError
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, template_name)

    # Test load with non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test get_file_name logic integration (suffix handling)
    template_with_ext = "test_template.json"
    expected_path = str(replay_dir / "test_template.json")
    # If we load a file that already has .json, it shouldn't append another .json
    with open(replay_dir / "test_template.json", 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    assert load(replay_dir, template_with_ext) == valid_context
```


# LLM-generated content at query #16
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
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        m_json.assert_called_once()
        # Verify the content passed to json.dump matches context
        args, _ = m_json.call_args
        assert args[0] == context

    # Test dump with template name already containing .json
    template_name_json = "my_template.json"
    expected_file_path_json = "/tmp/replay/my_template.json"
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump"):
        
        dump(replay_dir, template_name_json, context)
        m_open.assert_called_once_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test dump with missing 'cookiecutter' key in context
    invalid_context = {"not_cookiecutter": "data"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)

    # Test dump with Path object
    replay_path = Path("/tmp/replay")
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump"):
        
        dump(replay_path, template_name, context)
        # Check if the path was resolved correctly in join
        m_open.assert_called_once()
        args, _ = m_open.call_args
        assert str(args[0]) == expected_file_path
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
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_exists, \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump") as mock_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_exists.assert_called_once_with(replay_dir)
        mocked_file.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        mock_json_dump.assert_called_once()
        # Verify the content passed to json.dump is our context
        args, _ = mock_json_dump.call_args
        assert args[0] == context

    # Test dump with template_name already having .json extension
    template_with_ext = "my_template.json"
    expected_file_path_ext = "/tmp/replay/my_template.json"
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_with_ext, context)
        mocked_file.assert_called_once_with(expected_file_path_ext, 'w', encoding="utf-8")

    # Test dump raising ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "data"}
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #18
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

    # Test dump with template_name already having .json extension
    template_name_json = "my_template.json"
    expected_file_path_json = "/tmp/replay/my_template.json"
    
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


# LLM-generated content at query #19
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
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load failure due to missing 'cookiecutter' key
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Test Case 3: File does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup directory and file name
    replay_dir = tmp_path / "replays"
    template_name = "my_template"
    replay_file = replay_dir / "my_template.json"
    replay_dir.mkdir()

    # Test case 1: Successful load with valid context
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context

    # Test case 2: Load fails when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "data"}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test case 3: Load fails when file does not exist
    non_existent_template = "missing_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test case 4: Verify template name with .json suffix works correctly
    suffix_template = "my_template.json"
    with open(replay_dir / "my_template.json", 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_suffix_context = load(replay_dir, suffix_template)
    assert loaded_suffix_context == valid_context
```


# LLM-generated content at query #21
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

    # Test dump with template name already having .json extension
    template_name_json = "my_template.json"
    expected_file_path_json = "/tmp/replay/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        dump(replay_dir, template_name_json, context)
        mocked_file.assert_called_once_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test dump failure when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "data"}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #22
#--------------------------

```python
import json
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    replay_dir = tmp_path / "replay"
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
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Raise ValueError when 'cookiecutter' key is missing
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")

    # Test Case 3: Raise FileNotFoundError when file does not exist
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test Case 4: Verify template name with .json suffix works correctly
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_suffix_context = load(replay_dir, f"{template_name}.json")
    assert loaded_suffix_context == valid_context
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from pathlib import Path
import json

def test_load(tmp_path):
    # Setup directory and template name
    replay_dir = tmp_path / "replays"
    template_name = "my_template"  # function will append .json
    replay_file = replay_dir / "my_template.json"
    
    # Create valid context
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    
    # Create invalid context (missing 'cookiecutter' key)
    invalid_context = {"not_cookiecutter": True}

    # 1. Test successful load
    replay_dir.mkdir()
    with open(replay_template_file := replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # 2. Test load with invalid context (missing key)
    with open(replay_template_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # 3. Test load when file does not exist
    non_existent_file = replay_dir / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "does_not_exist")

    # 4. Test load with template name already containing .json
    template_with_ext = "template.json"
    replay_file_ext = replay_dir / "template.json"
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
        
    loaded_ext_context = load(replay_dir, template_with_ext)
    assert loaded_ext_context == valid_context
```


# LLM-generated content at query #24
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
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load failure due to missing 'cookiecutter' key
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")

    # Test Case 3: Load failure due to missing file
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
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
        "other_key": "value"
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
    assert "cookiecutter" in loaded_data

    # Test Case 2: Load failure due to missing 'cookiecutter' key
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: File does not exist
    non_existent_template = "missing_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test Case 4: Template name with .json extension already present
    template_name_ext = "test_template.json"
    replay_file_ext = replay_dir / "test_template.json"
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_name_ext)
    assert loaded_data_ext == valid_context
```


# LLM-generated content at query #26
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

    # Test case 1: Successful load
    file_path = replay_dir / f"{template_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "test_project"

    # Test case 2: Load with missing 'cookiecutter' key (ValueError)
    invalid_file_path = replay_dir / "invalid_template.json"
    with open(invalid_file_path, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")

    # Test case 3: File does not exist (FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = tmp_path / f"{template_name}.json"
    
    # Create the dummy file for loading
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with missing cookiecutter key (should raise ValueError)
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)

    # Test load when file does not exist (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #28
#--------------------------

```python
import json
import os
import pytest
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
    invalid_context = {"not_cookiecutter": "data"}
    
    file_path = os.path.join(str(replay_dir), f"{template_name}.json")

    # Test 1: Successful load
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data = load(replay_dir, template_name)
    assert loaded_data == valid_context
    assert loaded_data["cookiecutter"]["project_name"] == "my_project"

    # Test 2: Load fails when 'cookiecutter' key is missing
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test 3: Load fails when file does not exist
    non_existent_template = "non_existent"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test 4: Verify suffix handling (template name with .json)
    template_with_ext = "template.json"
    with open(os.path.join(str(replay_dir), "template.json"), 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_data_ext = load(replay_dir, template_with_ext)
    assert loaded_data_ext == valid_context
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from pathlib import Path

def test_load(tmp_path):
    # Setup data
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = tmp_path / f"{template_name}.json"
    
    # Create the file for loading
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with template name already containing .json
    template_with_ext = "my_template.json"
    replay_file_ext = tmp_path / "my_template.json"
    with open(replay_file_ext, "w", encoding="utf-8") as f:
        json.dump(context, f)
    
    loaded_context_ext = load(tmp_path, template_with_ext)
    assert loaded_context_ext == context

    # Test load error when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": {}}
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, "invalid")

    # Test load error when file does not exist
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest
import json
from pathlib import Path

def test_load(tmp_path):
    # Setup valid data
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    replay_file = tmp_path / f"{template_name}.json"
    
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test load with existing .json extension in name
    template_with_ext = "my_template.json"
    replay_file_ext = tmp_path / "my_template.json"
    with open(replay_file_ext, 'whe', encoding="utf-8") as f:
        json.dump(context, f)
    
    loaded_context_ext = load(tmp_path, template_with_ext)
    assert loaded_context_ext == context

    # Test load with missing cookiecutter key (should raise ValueError)
    invalid_context = {"not_cookiecutter": {}}
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, "invalid")

    # Test load with non-existent file (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
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
    expected_file_path = f"{replay_dir}/my_template.json"
    
    # Test 1: Successful dump
    with patch("cookiecutter.replay.make_sure_path_exists") as mock_mkdir, \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump") as m_json_dump:
        
        dump(replay_dir, template_name, context)
        
        mock_mkdir.assert_called_once_with(replay_dir)
        m_open.assert_called_once_with(expected_file_path, 'w', encoding="utf-8")
        m_json_dump.assert_called_once_with(context, m_open(), indent=2)

    # Test 2: Successful dump with .json extension already in template name
    template_name_json = "my_template.json"
    expected_file_path_json = f"{replay_dir}/my_template.json"
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as m_open, \
         patch("json.dump"):
        
        dump(replay_dir, template_name_json, context)
        m_open.assert_called_once_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test 3: Missing 'cookiecutter' key in context raises ValueError
    invalid_context = {"not_cookiecutter": "data"}
    
    with patch("cookiecutter.replay.make_sure_path_exists"):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            dump(replay_dir, template_name, invalid_context)
```


# LLM-generated content at query #32
#--------------------------

```python
import pytest
from pathlib import Path
import json

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

    # Test load with missing cookiecutter key (ValueError)
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)

    # Test load with non-existent file (FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")
```


# LLM-generated content at query #33
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

    replay_file = os.path.join(str(replay_dir), f"{template_name}.json")

    # Test Case 1: Successful load
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test Case 2: Load fails due to missing 'cookiecutter' key
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, template_name)

    # Test Case 3: Load fails because file does not exist
    non_existent_template = "ghost_template"
    with pytest.raises(FileNotFoundError):
        load(replay_dir, non_existent_template)

    # Test Case 4: Works with .json extension already in template name
    json_template_name = "template.json"
    with open(os.path.join(str(replay_dir), json_template_name), 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_json_context = load(replay_dir, json_template_name)
    assert loaded_json_context == valid_context
```


# LLM-generated content at query #34
#--------------------------

```python
import pytest
from pathlib import Path
import json

def test_load(tmp_path):
    template_name = "my_template"
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    
    valid_context = {"cookiecutter": {"project_name": "test_project"}, "other": "data"}
    invalid_context = {"not_cookiecutter": "data"}
    
    # Test successful load
    file_path = replay_dir / f"{template_name}.json"
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    
    # Test load with missing cookiecutter key (ValueError)
    invalid_file_path = replay_dir / "invalid_template.json"
    with open(invalid_file_path, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(replay_dir, "invalid_template")
        
    # Test load with non-existent file (FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")

    # Test load with template name already containing .json
    json_template_name = "already_has_extension.json"
    with open(replay_dir / json_template_name, 'w', encoding="utf-8") as f:
        json.dump(valid_context, f)
    
    loaded_ext_context = load(replay_dir, json_template_name)
    assert loaded_ext_context == valid_context
```


# LLM-generated content at query #35
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

    # Test case 1: Successful load
    replay_file = replay_dir / f"{template_name}.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(valid_context, f)

    loaded_context = load(replay_dir, template_name)
    assert loaded_context == valid_context
    assert loaded_context["cookiecutter"]["project_name"] == "test_project"

    # Test case 2: Load failure due to missing 'cookiecutter' key
    invalid_file = replay_dir / "invalid_template.json"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)

    with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
        load(replay_dir, "invalid_template")

    # Test case 3: Load failure due to non-existent file
    with pytest.raises(FileNotFoundError):
        load(replay_dir, "non_existent_template")
```


# LLM-generated content at query #36
#--------------------------

```python
import json
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
    
    # Create replay file manually for valid case
    replay_file = tmp_path / f"{template_name}.json"
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(context, f)

    # Test successful load
    loaded_context = load(tmp_path, template_name)
    assert loaded_context == context
    assert loaded_context["cookiecutter"]["project_name"] == "my_project"

    # Test load with missing 'cookiecutter' key (should raise ValueError)
    invalid_context = {"not_cookiecutter": {}}
    with open(replay_file, 'w', encoding="utf-8") as f:
        json.dump(invalid_context, f)
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        load(tmp_path, template_name)

    # Test load with non-existent file (should raise FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load(tmp_path, "non_existent_template")

    # Test load with filename already containing .json
    template_with_ext = "test_template.json"
    replay_file_ext = tmp_path / "test_template.json"
    with open(replay_file_ext, 'w', encoding="utf-8") as f:
        json.dump(context, f)
    
    loaded_ext_context = load(tmp_path, template_with_ext)
    assert loaded_ext_context == context
```


# LLM-generated content at query #37
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
        mock_json_dump.assert_called_once()
        # Verify the content passed to json.dump matches context
        args, _ = mock_json_dump.call_args
        assert args[0] == context

    # Test dump with template name already containing .json
    template_name_json = "my_template.json"
    expected_file_path_json = str(Path(replay_dir) / "my_template.json")
    
    with patch("cookiecutter.replay.make_sure_path_exists"), \
         patch("builtins.open", mock_open()) as mocked_file, \
         patch("json.dump"):
        
        from cookiecutter.replay import dump
        dump(replay_dir, template_name_json, context)
        mocked_file.assert_called_once_with(expected_file_path_json, 'w', encoding="utf-8")

    # Test dump raising ValueError when 'cookiecutter' key is missing
    invalid_context = {"not_cookiecutter": "data"}
    from cookiecutter.replay import dump as dump_func
    
    with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
        dump_func(replay_dir, template_name, invalid_context)
```


