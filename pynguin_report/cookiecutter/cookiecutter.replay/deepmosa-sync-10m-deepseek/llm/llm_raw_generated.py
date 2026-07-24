####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_load_valid_json_with_cookiecutter_key():
    import json
    import tempfile
    import os
    from pathlib import Path
    test_data = {"cookiecutter": {"project_name": "test"}}
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "template.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        result = load(tmpdir, "template")
        assert result == test_data

def test_load_valid_json_without_json_extension():
    import json
    import tempfile
    import os
    from pathlib import Path
    test_data = {"cookiecutter": {"project_name": "test"}}
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "template.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        result = load(tmpdir, "template.json")
        assert result == test_data

def test_load_missing_cookiecutter_key():
    import json
    import tempfile
    import os
    from pathlib import Path
    test_data = {"project_name": "test"}
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "template.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        try:
            load(tmpdir, "template")
            assert False
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_file_not_found():
    import tempfile
    import os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            load(tmpdir, "nonexistent")
            assert False
        except FileNotFoundError:
            assert True

def test_load_with_path_object():
    import json
    import tempfile
    import os
    from pathlib import Path
    test_data = {"cookiecutter": {"project_name": "test"}}
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "template.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f)
        result = load(Path(tmpdir), "template")
        assert result == test_data


# LLM-generated content at query #2
#--------------------------

def test_get_file_name_with_path_object_and_no_json_suffix():
    replay_dir = Path("/some/dir")
    template_name = "template"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join("/some/dir", "template.json")
    assert result == expected

def test_get_file_name_with_path_object_and_json_suffix():
    replay_dir = Path("/another/dir")
    template_name = "template.json"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join("/another/dir", "template.json")
    assert result == expected

def test_get_file_name_with_str_dir_and_no_json_suffix():
    replay_dir = "/str/dir"
    template_name = "my_template"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join("/str/dir", "my_template.json")
    assert result == expected

def test_get_file_name_with_str_dir_and_json_suffix():
    replay_dir = "/str/dir"
    template_name = "my_template.json"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join("/str/dir", "my_template.json")
    assert result == expected

def test_get_file_name_with_empty_template_name():
    replay_dir = Path("/empty")
    template_name = ""
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join("/empty", ".json")
    assert result == expected

def test_get_file_name_with_template_name_already_having_dot_json():
    replay_dir = "/test"
    template_name = "data.json"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join("/test", "data.json")
    assert result == expected

def test_get_file_name_with_template_name_having_other_suffix():
    replay_dir = Path("/mixed")
    template_name = "file.txt"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join("/mixed", "file.txt.json")
    assert result == expected


# LLM-generated content at query #3
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import json
    from unittest.mock import mock_open, patch
    mock_data = {}
    mock_json = json.dumps(mock_data)
    with patch('builtins.open', mock_open(read_data=mock_json)):
        with patch('pathlib.Path.is_file', return_value=True):
            try:
                load(Path('fake_dir'), 'fake_template')
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #4
#--------------------------

def test_dump_creates_directory_and_file():
    replay_dir = "test_replay"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = os.path.join(replay_dir, "my_template.json")
    assert os.path.exists(replay_dir)
    assert os.path.exists(expected_file)
    with open(expected_file, "r", encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context
    os.remove(expected_file)
    os.rmdir(replay_dir)

def test_dump_with_existing_json_extension():
    replay_dir = "test_replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = os.path.join(replay_dir, "my_template.json")
    assert os.path.exists(expected_file)
    with open(expected_file, "r", encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context
    os.remove(expected_file)
    os.rmdir(replay_dir)

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "my_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_dump_handles_os_error_from_make_sure_path_exists():
    replay_dir = "/invalid_root_path/test_replay"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except OSError:
        assert True


# LLM-generated content at query #5
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import json
    import tempfile
    import os
    from unittest.mock import mock_open, patch

    def get_file_name_mock(replay_dir, template_name):
        return "fake_file.json"

    with patch('__main__.get_file_name', side_effect=get_file_name_mock):
        with patch('builtins.open', mock_open(read_data=json.dumps({"not_cookiecutter": "value"}))):
            try:
                load("fake_dir", "fake_template")
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #6
#--------------------------

def test_load_context_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_data = {'cookiecutter': {'project_name': 'test'}}
    mock_json_load = lambda x: test_data
    with patch('builtins.open', mock_open(read_data='')) as mock_file:
        with patch('json.load', mock_json_load):
            from cookiecutter.replay import load
            result = load(Path('/fake/dir'), 'template')
    assert 'cookiecutter' in result


# LLM-generated content at query #7
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import json
    from cookiecutter.replay import dump
    from cookiecutter.utils import make_sure_path_exists
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'not_cookiecutter': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #8
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    open_mock = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', open_mock):
        result = load(replay_dir, template_name)
    open_mock.assert_called_once_with(expected_file, encoding='utf-8')
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_file = '/tmp/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    open_mock = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', open_mock):
        result = load(replay_dir, template_name)
    open_mock.assert_called_once_with(expected_file, encoding='utf-8')
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    invalid_context = {'key': 'value'}
    open_mock = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch('builtins.open', open_mock):
        try:
            load(replay_dir, template_name)
            assert False
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    with mock.patch('builtins.open', side_effect=FileNotFoundError):
        try:
            load(replay_dir, template_name)
            assert False
        except FileNotFoundError:
            assert True

def test_load_json_decode_error():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    with mock.patch('builtins.open', side_effect=json.JSONDecodeError('msg', 'doc', 0)):
        try:
            load(replay_dir, template_name)
            assert False
        except json.JSONDecodeError:
            assert True


# LLM-generated content at query #9
#--------------------------

def test_load_context_missing_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    mock_json_data = {"some_key": "some_value"}
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_json_data))):
        try:
            load(test_replay_dir, test_template_name)
            assert False
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #10
#--------------------------

def test_load_context_without_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    context_without_cookiecutter = {}
    mock_json_data = json.dumps(context_without_cookiecutter)
    mock_file_path = Path('test_replay.json')
    with patch('pathlib.Path.open', mock_open(read_data=mock_json_data)):
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pathlib.Path.exists', return_value=True):
                try:
                    load(mock_file_path, 'test_template')
                except ValueError as e:
                    assert str(e) == 'Context is required to contain a cookiecutter key'
                else:
                    assert False


# LLM-generated content at query #11
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = 'some_dir'
    template_name = 'some_template'
    context = {'not_cookiecutter': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #12
#--------------------------

def test_load_returns_context_with_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_data = {'cookiecutter': {'project_name': 'test'}}
    mock_json_load = lambda x: test_data
    with patch('builtins.open', mock_open(read_data='{}')), patch('json.load', mock_json_load), patch('path.to.get_file_name', return_value='dummy_path'):
        result = load('dummy_dir', 'dummy_template')
        assert 'cookiecutter' in result


# LLM-generated content at query #13
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = Path("/fake/dir/cookiecutter-test_template.json")
    mock_json_data = {"some_key": "some_value"}
    with patch('pathlib.Path.is_file', return_value=True):
        with patch('__main__.get_file_name', return_value=test_file_name) as mock_get_file_name:
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_json_data))) as mock_file:
                try:
                    result = load(test_replay_dir, test_template_name)
                except ValueError as e:
                    assert str(e) == 'Context is required to contain a cookiecutter key'
                    return
                assert False, "Expected ValueError not raised"


# LLM-generated content at query #14
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context_without_cookiecutter = {'key': 'value'}
    try:
        dump(replay_dir, template_name, context_without_cookiecutter)
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #15
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from cookiecutter.replay import dump
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {"not_cookiecutter": "value"}
        try:
            dump(replay_dir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #16
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = Path('some_dir')
    template_name = 'some_template'
    context = {'not_cookiecutter': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #17
#--------------------------

def test_load_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_data = {'cookiecutter': {'project_name': 'test'}}
    mock_file_content = json.dumps(test_data)
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('pathlib.Path.is_file', return_value=True):
            from cookiecutter.replay import get_file_name
            with patch('cookiecutter.replay.get_file_name', return_value='dummy_path'):
                from cookiecutter.replay import load
                result = load('dummy_dir', 'dummy_template')
                assert 'cookiecutter' in result


# LLM-generated content at query #18
#--------------------------

def test_load_context_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_context = {'cookiecutter': {'project_name': 'test'}}
    mock_json_load = lambda x: test_context
    with patch('builtins.open', mock_open(read_data='{}')):
        with patch('json.load', mock_json_load):
            result = load(Path('test_dir'), 'test_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #19
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'template'
    with mock.patch('builtins.open', side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load(replay_dir, template_name)

def test_load_missing_cookiecutter_key():
    replay_dir = '/tmp'
    template_name = 'template'
    invalid_context = {'key': 'value'}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch('builtins.open', mock_open):
        with pytest.raises(ValueError, match='Context is required to contain a cookiecutter key'):
            load(replay_dir, template_name)

def test_load_with_json_extension():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_without_json_extension():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context


# LLM-generated content at query #20
#--------------------------

def test_load_success():
    replay_dir = "/tmp"
    template_name = "template"
    expected_context = {"cookiecutter": {"key": "value"}}
    with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data='{"cookiecutter": {"key": "value"}}')):
        with unittest.mock.patch("os.path.join", return_value="/tmp/template.json"):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = "/tmp"
    template_name = "template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data='{"cookiecutter": {"key": "value"}}')):
        with unittest.mock.patch("os.path.join", return_value="/tmp/template.json"):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = "/tmp"
    template_name = "template"
    with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data='{"key": "value"}')):
        with unittest.mock.patch("os.path.join", return_value="/tmp/template.json"):
            try:
                load(replay_dir, template_name)
                assert False
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_file_not_found():
    replay_dir = "/tmp"
    template_name = "template"
    with unittest.mock.patch("builtins.open", side_effect=FileNotFoundError):
        with unittest.mock.patch("os.path.join", return_value="/tmp/template.json"):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True

def test_load_invalid_json():
    replay_dir = "/tmp"
    template_name = "template"
    with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data="invalid json")):
        with unittest.mock.patch("os.path.join", return_value="/tmp/template.json"):
            try:
                load(replay_dir, template_name)
                assert False
            except json.JSONDecodeError:
                assert True


# LLM-generated content at query #21
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from cookiecutter.replay import dump
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {"not_cookiecutter": {}}
        try:
            dump(replay_dir, template_name, context)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #22
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = Path("/fake/dir/test_template.json")
    with patch('pathlib.Path.open', mock_open(read_data='{"not_cookiecutter": {}}')):
        with patch('json.load', return_value={"not_cookiecutter": {}}):
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #23
#--------------------------

def test_load_success():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file = "/tmp/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context


def test_load_with_json_extension():
    replay_dir = "/tmp"
    template_name = "template.json"
    expected_file = "/tmp/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context


def test_load_missing_cookiecutter():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file = "/tmp/template.json"
    invalid_context = {"key": "value"}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file):
            try:
                load(replay_dir, template_name)
                assert False
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"


def test_load_file_not_found():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file = "/tmp/template.json"
    with mock.patch("os.path.join", return_value=expected_file):
        with mock.patch("builtins.open", side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True


def test_load_json_decode_error():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file = "/tmp/template.json"
    mock_open = mock.mock_open(read_data="invalid json")
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file):
            try:
                load(replay_dir, template_name)
                assert False
            except json.JSONDecodeError:
                assert True


# LLM-generated content at query #24
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path('/fake/dir')
    test_template_name = 'test_template'
    test_file_name = test_replay_dir / 'test_template.json'
    test_context_without_cookiecutter = {'some_key': 'some_value'}
    with patch('__main__.get_file_name', return_value=test_file_name) as mock_get_file_name:
        with patch('builtins.open', mock_open(read_data=json.dumps(test_context_without_cookiecutter))) as mock_file:
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'
                mock_get_file_name.assert_called_once_with(test_replay_dir, test_template_name)
                mock_file.assert_called_once_with(test_file_name, encoding='utf-8')


# LLM-generated content at query #25
#--------------------------

def test_dump_creates_directory_if_not_exists():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    assert Path(replay_dir).exists()
    Path(replay_dir).rmdir()

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_dump_writes_correct_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    replay_file = Path(replay_dir) / f"{template_name}.json"
    assert replay_file.exists()
    with open(replay_file, encoding="utf-8") as infile:
        data = json.load(infile)
    assert data == context
    replay_file.unlink()
    Path(replay_dir).rmdir()

def test_dump_appends_json_suffix_only_if_missing():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    replay_file = Path(replay_dir) / template_name
    assert replay_file.exists()
    replay_file.unlink()
    Path(replay_dir).rmdir()


# LLM-generated content at query #26
#--------------------------

def test_cookiecutter_key_present():
    context = {'cookiecutter': {}}
    result = 'cookiecutter' not in context
    assert result == False


# LLM-generated content at query #27
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_data = {"some_key": "some_value"}
    mock_file = mock_open(read_data=json.dumps(test_data))
    with patch("builtins.open", mock_file):
        with patch("pathlib.Path.is_file", return_value=True):
            try:
                load("fake_dir", "fake_template")
                assert False
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #28
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'not_cookiecutter': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #29
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = test_replay_dir / f"{test_template_name}.json"
    test_context = {"some_key": "some_value"}
    with patch('pathlib.Path.__truediv__', return_value=test_file_name):
        with patch('builtins.open', mock_open(read_data=json.dumps(test_context))):
            try:
                load(test_replay_dir, test_template_name)
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #30
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = Path('some_dir')
    template_name = 'test_template'
    context = {'not_cookiecutter': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #31
#--------------------------

def test_load_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_context = {'cookiecutter': {'project_name': 'test'}}
    json_data = json.dumps(test_context)
    mock_file = mock_open(read_data=json_data)
    with patch('builtins.open', mock_file):
        with patch('pathlib.Path.is_file', return_value=True):
            result = load(Path('fake_dir'), 'fake_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #32
#--------------------------

def test_load_returns_context_with_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_context = {'cookiecutter': {'project_name': 'test'}}
    json_data = json.dumps(test_context)
    with patch('builtins.open', mock_open(read_data=json_data)):
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pathlib.Path.exists', return_value=True):
                result = load(Path('fake_dir'), 'fake_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #33
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path('/fake/dir')
    test_template_name = 'fake_template'
    fake_file_path = Path('/fake/dir/fake_template.json')
    fake_context = {'some_key': 'some_value'}
    with patch('pathlib.Path.is_file', return_value=True):
        with patch('builtins.open', mock_open(read_data=json.dumps(fake_context))):
            with patch('cookiecutter.replay.get_file_name', return_value=fake_file_path):
                try:
                    from cookiecutter.replay import load
                    result = load(test_replay_dir, test_template_name)
                except ValueError as e:
                    assert str(e) == 'Context is required to contain a cookiecutter key'
                else:
                    assert False, "Expected ValueError but none was raised"


# LLM-generated content at query #34
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'not_cookiecutter': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #35
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'test_template'
    expected_file = '/tmp/test_template.json'
    mock_data = {'cookiecutter': {'key': 'value'}}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(mock_data))
    with unittest.mock.patch('builtins.open', mock_open):
        with unittest.mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == mock_data
    mock_open.assert_called_once_with(expected_file, encoding='utf-8')


def test_load_with_json_extension():
    replay_dir = '/tmp'
    template_name = 'test_template.json'
    expected_file = '/tmp/test_template.json'
    mock_data = {'cookiecutter': {'key': 'value'}}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(mock_data))
    with unittest.mock.patch('builtins.open', mock_open):
        with unittest.mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == mock_data
    mock_open.assert_called_once_with(expected_file, encoding='utf-8')


def test_load_missing_cookiecutter():
    replay_dir = '/tmp'
    template_name = 'test_template'
    expected_file = '/tmp/test_template.json'
    mock_data = {'key': 'value'}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(mock_data))
    with unittest.mock.patch('builtins.open', mock_open):
        with unittest.mock.patch('os.path.join', return_value=expected_file):
            try:
                load(replay_dir, template_name)
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'test_template'
    expected_file = '/tmp/test_template.json'
    with unittest.mock.patch('os.path.join', return_value=expected_file):
        with unittest.mock.patch('builtins.open', side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True


def test_load_json_decode_error():
    replay_dir = '/tmp'
    template_name = 'test_template'
    expected_file = '/tmp/test_template.json'
    mock_open = unittest.mock.mock_open(read_data='invalid json')
    with unittest.mock.patch('builtins.open', mock_open):
        with unittest.mock.patch('os.path.join', return_value=expected_file):
            try:
                load(replay_dir, template_name)
                assert False
            except json.JSONDecodeError:
                assert True


# LLM-generated content at query #36
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    context_without_cookiecutter = {"some_key": "some_value"}
    json_data = json.dumps(context_without_cookiecutter)
    with patch("builtins.open", mock_open(read_data=json_data)):
        try:
            load(Path("fake_dir"), "fake_template")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #37
#--------------------------

def test_dump_raises_error_when_cookiecutter_key_missing():
    from cookiecutter.replay import dump
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context_without_key = {"other_key": "value"}
        try:
            dump(replay_dir, template_name, context_without_key)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #38
#--------------------------

def test_load_context_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_context = {'cookiecutter': {'project_name': 'test'}}
    mock_json_load = lambda x: test_context
    with patch('builtins.open', mock_open(read_data='')):
        with patch('json.load', mock_json_load):
            from cookiecutter.replay import load
            result = load(Path('test_dir'), 'test_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #39
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    context_without_cookiecutter = {"some_key": "some_value"}
    json_data = json.dumps(context_without_cookiecutter)
    with patch("builtins.open", mock_open(read_data=json_data)):
        with patch("pathlib.Path.is_file", return_value=True):
            try:
                load(Path("fake_dir"), "fake_template")
                assert False
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #40
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_replay_file = Path("/fake/dir/test_template.json")
    test_context_without_cookiecutter = {"some_key": "some_value"}
    with patch('pathlib.Path.open', mock_open(read_data=json.dumps(test_context_without_cookiecutter))):
        with patch('cookiecutter.replay.get_file_name', return_value=test_replay_file):
            try:
                from cookiecutter.replay import load
                result = load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_get_file_name_with_path_and_no_json_suffix():
    replay_dir = Path("/some/dir")
    template_name = "template"
    result = get_file_name(replay_dir, template_name)
    assert result == "/some/dir/template.json"

def test_get_file_name_with_path_and_json_suffix():
    replay_dir = Path("/another/dir")
    template_name = "template.json"
    result = get_file_name(replay_dir, template_name)
    assert result == "/another/dir/template.json"

def test_get_file_name_with_str_and_no_json_suffix():
    replay_dir = "/str/dir"
    template_name = "my_template"
    result = get_file_name(replay_dir, template_name)
    assert result == "/str/dir/my_template.json"

def test_get_file_name_with_str_and_json_suffix():
    replay_dir = "/str/another"
    template_name = "my_template.json"
    result = get_file_name(replay_dir, template_name)
    assert result == "/str/another/my_template.json"

def test_get_file_name_with_empty_template_name():
    replay_dir = Path("/empty")
    template_name = ""
    result = get_file_name(replay_dir, template_name)
    assert result == "/empty/.json"

def test_get_file_name_with_template_name_already_having_dot_json():
    replay_dir = "/path"
    template_name = "file.json"
    result = get_file_name(replay_dir, template_name)
    assert result == "/path/file.json"

def test_get_file_name_with_template_name_having_other_suffix():
    replay_dir = Path("/other")
    template_name = "file.txt"
    result = get_file_name(replay_dir, template_name)
    assert result == "/other/file.txt.json"


# LLM-generated content at query #2
#--------------------------

def test_dump_creates_directory_if_not_exists():
    replay_dir = 'test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert Path(replay_dir).exists()
    Path(replay_dir).rmdir()

def test_dump_raises_error_without_cookiecutter_key():
    replay_dir = 'test_replay'
    template_name = 'test_template'
    context = {'key': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_dump_writes_correct_json_file():
    replay_dir = 'test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'project': 'test'}}
    dump(replay_dir, template_name, context)
    file_path = Path(replay_dir) / f'{template_name}.json'
    assert file_path.exists()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data == context
    file_path.unlink()
    Path(replay_dir).rmdir()

def test_dump_handles_template_name_with_json_extension():
    replay_dir = 'test_replay'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    file_path = Path(replay_dir) / template_name
    assert file_path.exists()
    file_path.unlink()
    Path(replay_dir).rmdir()


# LLM-generated content at query #3
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_file = '/tmp/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    invalid_context = {'key': 'value'}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            try:
                load(replay_dir, template_name)
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    with mock.patch('os.path.join', return_value=expected_file):
        with mock.patch('builtins.open', side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True

def test_load_json_decode_error():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    mock_open = mock.mock_open(read_data='invalid json')
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            try:
                load(replay_dir, template_name)
                assert False
            except json.JSONDecodeError:
                assert True


# LLM-generated content at query #4
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'not_cookiecutter': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #5
#--------------------------

def test_load_success():
    replay_dir = "/tmp"
    template_name = "template"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_suffix():
    replay_dir = "/tmp"
    template_name = "template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = "/tmp"
    template_name = "template"
    invalid_context = {"key": "value"}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch("builtins.open", mock_open):
        try:
            load(replay_dir, template_name)
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_file_not_found():
    replay_dir = "/tmp"
    template_name = "template"
    with mock.patch("builtins.open", side_effect=FileNotFoundError):
        try:
            load(replay_dir, template_name)
        except FileNotFoundError:
            pass

def test_load_json_decode_error():
    replay_dir = "/tmp"
    template_name = "template"
    mock_open = mock.mock_open(read_data="invalid json")
    with mock.patch("builtins.open", mock_open):
        try:
            load(replay_dir, template_name)
        except json.JSONDecodeError:
            pass


# LLM-generated content at query #6
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_context = {"some_key": "some_value"}
    json_data = json.dumps(test_context)
    with patch("builtins.open", mock_open(read_data=json_data)):
        try:
            load(Path("fake_dir"), "fake_template")
            assert False
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #7
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from cookiecutter.replay import dump
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context_without_cookiecutter = {"key": "value"}
        try:
            dump(replay_dir, template_name, context_without_cookiecutter)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #8
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from cookiecutter.replay import dump
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context = {"not_cookiecutter": "value"}
        try:
            dump(replay_dir, template_name, context)
            assert False
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #9
#--------------------------

def test_load_success():
    replay_dir = "/tmp/replay"
    template_name = "template"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(expected_context))
    with unittest.mock.patch("builtins.open", mock_open):
        with unittest.mock.patch("os.path.join", return_value="/tmp/replay/template.json"):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_suffix():
    replay_dir = "/tmp/replay"
    template_name = "template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(expected_context))
    with unittest.mock.patch("builtins.open", mock_open):
        with unittest.mock.patch("os.path.join", return_value="/tmp/replay/template.json"):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = "/tmp/replay"
    template_name = "template"
    invalid_context = {"key": "value"}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(invalid_context))
    with unittest.mock.patch("builtins.open", mock_open):
        with unittest.mock.patch("os.path.join", return_value="/tmp/replay/template.json"):
            try:
                load(replay_dir, template_name)
                assert False
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_file_not_found():
    replay_dir = "/tmp/replay"
    template_name = "template"
    with unittest.mock.patch("os.path.join", return_value="/tmp/replay/template.json"):
        with unittest.mock.patch("builtins.open", side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True

def test_load_invalid_json():
    replay_dir = "/tmp/replay"
    template_name = "template"
    mock_open = unittest.mock.mock_open(read_data="invalid json")
    with unittest.mock.patch("builtins.open", mock_open):
        with unittest.mock.patch("os.path.join", return_value="/tmp/replay/template.json"):
            try:
                load(replay_dir, template_name)
                assert False
            except json.JSONDecodeError:
                assert True


# LLM-generated content at query #10
#--------------------------

def test_load_context_missing_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    replay_dir = Path("test_dir")
    template_name = "test_template"
    expected_file = Path("test_dir/test_template.json")
    mock_json_data = {}
    with patch("pathlib.Path.open", mock_open(read_data=json.dumps(mock_json_data))):
        with patch("pathlib.Path.is_file", return_value=True):
            with patch("pathlib.Path.exists", return_value=True):
                try:
                    load(replay_dir, template_name)
                    assert False
                except ValueError as e:
                    assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #11
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_dir = Path("test_dir")
    test_template = "test_template"
    test_file_path = test_dir / f"{test_template}.json"
    test_context = {"some_key": "some_value"}
    with patch("pathlib.Path.mkdir") as mock_mkdir, patch("pathlib.Path.is_file", return_value=True) as mock_is_file, patch("builtins.open", mock_open(read_data=json.dumps(test_context))) as mock_file:
        from cookiecutter.replay import load
        try:
            result = load(test_dir, test_template)
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"
        else:
            assert False, "Expected ValueError not raised"


# LLM-generated content at query #12
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    context_without_cookiecutter = {"some_key": "some_value"}
    json_data = json.dumps(context_without_cookiecutter)
    with patch("builtins.open", mock_open(read_data=json_data)):
        with patch("pathlib.Path.is_file", return_value=True):
            try:
                load(Path("fake_dir"), "fake_template")
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #13
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import json
    from cookiecutter.replay import dump
    from cookiecutter.utils import make_sure_path_exists
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'not_cookiecutter': 'value'}
    try:
        dump(replay_dir, template_name, context)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #14
#--------------------------

def test_dump_creates_directory_and_writes_file():
    import tempfile
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    context = {'cookiecutter': {'project_name': 'Test'}}
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir) / 'subdir'
        template_name = 'my_template'
        dump(replay_dir, template_name, context)
        expected_file = replay_dir / f'{template_name}.json'
        assert expected_file.exists()
        with open(expected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data == context

def test_dump_raises_value_error_without_cookiecutter_key():
    import tempfile
    from pathlib import Path
    from cookiecutter.replay import dump
    context = {'project_name': 'Test'}
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = 'my_template'
        try:
            dump(replay_dir, template_name, context)
            assert False
        except ValueError as e:
            assert 'Context is required to contain a cookiecutter key' in str(e)

def test_dump_handles_template_name_with_json_extension():
    import tempfile
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    context = {'cookiecutter': {'project_name': 'Test'}}
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = 'my_template.json'
        dump(replay_dir, template_name, context)
        expected_file = replay_dir / template_name
        assert expected_file.exists()
        with open(expected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data == context

def test_dump_writes_indented_json():
    import tempfile
    from pathlib import Path
    from cookiecutter.replay import dump
    context = {'cookiecutter': {'key': 'value'}}
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = 'template'
        dump(replay_dir, template_name, context)
        expected_file = replay_dir / f'{template_name}.json'
        with open(expected_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '\n' in content
        assert '  ' in content


# LLM-generated content at query #15
#--------------------------

def test_load_context_missing_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    replay_dir = Path("test_dir")
    template_name = "test_template"
    expected_file = replay_dir / f"{template_name}.json"
    mock_context = {"some_key": "some_value"}
    with patch("pathlib.Path.__truediv__", return_value=expected_file), \
         patch("builtins.open", mock_open(read_data=json.dumps(mock_context))):
        try:
            load(replay_dir, template_name)
            assert False
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #16
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = Path('test_replay')
    template_name = 'test_template'
    context = {'not_cookiecutter': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #17
#--------------------------

def test_load_success():
    replay_dir = '/tmp/test'
    template_name = 'template'
    expected_file = '/tmp/test/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = '/tmp/test'
    template_name = 'template.json'
    expected_file = '/tmp/test/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = '/tmp/test'
    template_name = 'template'
    expected_file = '/tmp/test/template.json'
    invalid_context = {'key': 'value'}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            try:
                load(replay_dir, template_name)
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_not_found():
    replay_dir = '/tmp/test'
    template_name = 'template'
    expected_file = '/tmp/test/template.json'
    with mock.patch('os.path.join', return_value=expected_file):
        with mock.patch('builtins.open', side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True

def test_load_invalid_json():
    replay_dir = '/tmp/test'
    template_name = 'template'
    expected_file = '/tmp/test/template.json'
    mock_open = mock.mock_open(read_data='invalid json')
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            try:
                load(replay_dir, template_name)
                assert False
            except json.JSONDecodeError:
                assert True


# LLM-generated content at query #18
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import json
    import tempfile
    import os
    from cookiecutter.replay import dump
    from cookiecutter.utils import make_sure_path_exists
    replay_dir = Path(tempfile.mkdtemp())
    template_name = "test_template"
    context_without_cookiecutter = {"key": "value"}
    try:
        dump(replay_dir, template_name, context_without_cookiecutter)
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"
    else:
        assert False, "Expected ValueError not raised"


# LLM-generated content at query #19
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = test_replay_dir / f"{test_template_name}.json"
    mock_json_data = {"some_key": "some_value"}
    with patch('pathlib.Path.open', mock_open(read_data=json.dumps(mock_json_data))):
        with patch('pathlib.Path.__truediv__', return_value=test_file_name):
            try:
                load(test_replay_dir, test_template_name)
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #20
#--------------------------

def test_context_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    context_with_cookiecutter = {'cookiecutter': {'project_name': 'test'}}
    mock_data = json.dumps(context_with_cookiecutter)
    with patch('builtins.open', mock_open(read_data=mock_data)):
        with patch('pathlib.Path.is_file', return_value=True):
            from cookiecutter.replay import load
            result = load(Path('test_dir'), 'test_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #21
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = Path("/fake/dir/test_template.json")
    test_context_without_cookiecutter = {"some_key": "some_value"}
    with patch('__main__.get_file_name', return_value=test_file_name) as mock_get_file_name:
        with patch('builtins.open', mock_open(read_data=json.dumps(test_context_without_cookiecutter))) as mock_file:
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'
                mock_get_file_name.assert_called_once_with(test_replay_dir, test_template_name)
                mock_file.assert_called_once_with(test_file_name, encoding="utf-8")


# LLM-generated content at query #22
#--------------------------

def test_load_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_data = {'cookiecutter': {'project_name': 'test'}}
    mock_json_load = lambda x: test_data
    with patch('builtins.open', mock_open()) as mock_file, patch('json.load', mock_json_load):
        result = load(Path('fake_dir'), 'fake_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #23
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = test_replay_dir / f"{test_template_name}.json"
    test_context = {"some_key": "some_value"}
    with patch('pathlib.Path.open', mock_open(read_data=json.dumps(test_context))):
        with patch('pathlib.Path.is_file', return_value=True):
            try:
                load(test_replay_dir, test_template_name)
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #24
#--------------------------

def test_load_context_missing_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    mock_data = json.dumps({"some_key": "some_value"})
    with patch("builtins.open", mock_open(read_data=mock_data)), patch("pathlib.Path.is_file", return_value=True):
        try:
            load(Path("fake_dir"), "fake_template")
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"
        else:
            assert False, "Expected ValueError not raised"


# LLM-generated content at query #25
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_replay_file = Path("/fake/dir/test_template.json")
    test_context_without_cookiecutter = {"some_key": "some_value"}
    with patch('path.to.module.get_file_name', return_value=test_replay_file) as mock_get_file_name:
        with patch('builtins.open', mock_open(read_data=json.dumps(test_context_without_cookiecutter))) as mock_file:
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'
                mock_get_file_name.assert_called_once_with(test_replay_dir, test_template_name)
                mock_file.assert_called_once_with(test_replay_file, encoding="utf-8")


# LLM-generated content at query #26
#--------------------------

def test_dump_creates_directory_and_writes_file():
    replay_dir = "test_replay"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = os.path.join(replay_dir, "my_template.json")
    assert os.path.exists(expected_file)
    with open(expected_file, "r", encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context
    os.remove(expected_file)
    os.rmdir(replay_dir)

def test_dump_raises_error_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "my_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_dump_handles_existing_json_suffix():
    replay_dir = "test_replay"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = os.path.join(replay_dir, "my_template.json")
    assert os.path.exists(expected_file)
    os.remove(expected_file)
    os.rmdir(replay_dir)

def test_dump_handles_nested_directory_creation():
    replay_dir = "nested/test/replay"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = os.path.join(replay_dir, "template.json")
    assert os.path.exists(expected_file)
    os.remove(expected_file)
    os.rmdir(os.path.join("nested", "test", "replay"))
    os.rmdir(os.path.join("nested", "test"))
    os.rmdir("nested")


# LLM-generated content at query #27
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_file = '/tmp/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    invalid_context = {'key': 'value'}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            try:
                load(replay_dir, template_name)
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'
            else:
                assert False

def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    with mock.patch('os.path.join', return_value=expected_file):
        with mock.patch('builtins.open', side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
            except FileNotFoundError:
                pass
            else:
                assert False

def test_load_invalid_json():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    mock_open = mock.mock_open(read_data='invalid json')
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            try:
                load(replay_dir, template_name)
            except json.JSONDecodeError:
                pass
            else:
                assert False


# LLM-generated content at query #28
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"cookiecutter": {"key": "value"}}')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_suffix():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"cookiecutter": {"key": "value"}}')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"other": "data"}')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            try:
                load(replay_dir, template_name)
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', side_effect=FileNotFoundError):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True

def test_load_json_decode_error():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='invalid json')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            try:
                load(replay_dir, template_name)
                assert False
            except json.JSONDecodeError:
                assert True


# LLM-generated content at query #29
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path('/fake/dir')
    test_template_name = 'test_template'
    test_file_name = Path('/fake/dir/test_template.json')
    test_context_without_cookiecutter = {'some_key': 'some_value'}
    with patch('__main__.get_file_name', return_value=test_file_name) as mock_get_file_name:
        with patch('builtins.open', mock_open(read_data=json.dumps(test_context_without_cookiecutter))) as mock_file:
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'
                mock_get_file_name.assert_called_once_with(test_replay_dir, test_template_name)
                mock_file.assert_called_once_with(test_file_name, encoding='utf-8')


# LLM-generated content at query #30
#--------------------------

def test_dump_creates_directory_if_not_exists():
    replay_dir = 'test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(replay_dir)
    os.rmdir(replay_dir)

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = 'test_replay'
    template_name = 'test_template'
    context = {'key': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_dump_writes_correct_json_file():
    replay_dir = 'test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    file_path = os.path.join(replay_dir, 'test_template.json')
    with open(file_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    assert data == context
    os.remove(file_path)
    os.rmdir(replay_dir)

def test_dump_handles_template_name_with_json_extension():
    replay_dir = 'test_replay'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    file_path = os.path.join(replay_dir, 'test_template.json')
    assert os.path.exists(file_path)
    os.remove(file_path)
    os.rmdir(replay_dir)


# LLM-generated content at query #31
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_data = {"some_key": "some_value"}
    mock_file_content = json.dumps(test_data)
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('pathlib.Path.is_file', return_value=True):
            try:
                load(Path("fake_dir"), "fake_template")
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #32
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = Path("/fake/dir/test_template.json")
    mock_json_data = {"some_key": "some_value"}
    with patch('pathlib.Path.is_file', return_value=True), \
         patch('__main__.get_file_name', return_value=test_file_name), \
         patch('builtins.open', mock_open(read_data=json.dumps(mock_json_data))):
        try:
            load(test_replay_dir, test_template_name)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #33
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = test_replay_dir / "test_template.json"
    test_context_without_cookiecutter = {"some_key": "some_value"}
    with patch("pathlib.Path.__truediv__", return_value=test_file_name):
        with patch("builtins.open", mock_open(read_data=json.dumps(test_context_without_cookiecutter))):
            try:
                load(test_replay_dir, test_template_name)
                assert False
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #34
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import json
    from cookiecutter.replay import dump
    from cookiecutter.utils import make_sure_path_exists
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context_without_cookiecutter = {'key': 'value'}
    try:
        dump(replay_dir, template_name, context_without_cookiecutter)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #35
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"cookiecutter": {"key": "value"}}')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"cookiecutter": {"key": "value"}}')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"other": "data"}')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            try:
                load(replay_dir, template_name)
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', side_effect=FileNotFoundError):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True

def test_load_invalid_json():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='invalid json')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            try:
                load(replay_dir, template_name)
                assert False
            except json.JSONDecodeError:
                assert True


# LLM-generated content at query #36
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from cookiecutter.replay import dump
    from pathlib import Path
    import tempfile
    import pytest
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context_without_cookiecutter = {"key": "value"}
        with pytest.raises(ValueError, match="Context is required to contain a cookiecutter key"):
            dump(replay_dir, template_name, context_without_cookiecutter)


# LLM-generated content at query #37
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = '/tmp'
    template_name = 'template'
    invalid_context = {'key': 'value'}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch('builtins.open', mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'template'
    with mock.patch('builtins.open', side_effect=FileNotFoundError):
        try:
            load(replay_dir, template_name)
            assert False
        except FileNotFoundError:
            assert True

def test_load_json_decode_error():
    replay_dir = '/tmp'
    template_name = 'template'
    mock_open = mock.mock_open(read_data='invalid json')
    with mock.patch('builtins.open', mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except json.JSONDecodeError:
            assert True


# LLM-generated content at query #38
#--------------------------

def test_load_success():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file = "/tmp/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    with open(expected_file, "w", encoding="utf-8") as f:
        json.dump(expected_context, f)
    result = load(replay_dir, template_name)
    assert result == expected_context
    os.remove(expected_file)

def test_load_with_json_extension():
    replay_dir = "/tmp"
    template_name = "template.json"
    expected_file = "/tmp/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    with open(expected_file, "w", encoding="utf-8") as f:
        json.dump(expected_context, f)
    result = load(replay_dir, template_name)
    assert result == expected_context
    os.remove(expected_file)

def test_load_missing_cookiecutter_key():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file = "/tmp/template.json"
    invalid_context = {"key": "value"}
    with open(expected_file, "w", encoding="utf-8") as f:
        json.dump(invalid_context, f)
    try:
        load(replay_dir, template_name)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"
    os.remove(expected_file)

def test_load_file_not_found():
    replay_dir = "/tmp"
    template_name = "nonexistent"
    try:
        load(replay_dir, template_name)
        assert False
    except FileNotFoundError:
        assert True

def test_load_with_path_object():
    replay_dir = Path("/tmp")
    template_name = "template"
    expected_file = "/tmp/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    with open(expected_file, "w", encoding="utf-8") as f:
        json.dump(expected_context, f)
    result = load(replay_dir, template_name)
    assert result == expected_context
    os.remove(expected_file)


# LLM-generated content at query #39
#--------------------------

def test_dump_creates_directory_if_not_exists():
    replay_dir = '/tmp/test_replay'
    template_name = 'my_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(replay_dir)

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = '/tmp/test_replay'
    template_name = 'my_template'
    context = {'key': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_dump_writes_correct_json_file():
    replay_dir = '/tmp/test_replay'
    template_name = 'my_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    file_path = os.path.join(replay_dir, 'my_template.json')
    with open(file_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    assert data == context

def test_dump_appends_json_suffix_if_missing():
    replay_dir = '/tmp/test_replay'
    template_name = 'my_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    file_path = os.path.join(replay_dir, 'my_template.json')
    assert os.path.exists(file_path)

def test_dump_does_not_append_json_suffix_if_present():
    replay_dir = '/tmp/test_replay'
    template_name = 'my_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    file_path = os.path.join(replay_dir, 'my_template.json')
    assert os.path.exists(file_path)


# LLM-generated content at query #40
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_key_missing():
    from pathlib import Path
    import json
    from cookiecutter.replay import dump
    from cookiecutter.utils import make_sure_path_exists
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context_without_cookiecutter = {'other_key': 'value'}
    try:
        dump(replay_dir, template_name, context_without_cookiecutter)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #41
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_key_missing():
    from pathlib import Path
    import json
    from cookiecutter.replay import dump
    from unittest.mock import patch, mock_open, MagicMock

    replay_dir = Path('/fake/replay')
    template_name = 'test_template'
    context_without_cookiecutter = {'other_key': 'value'}

    with patch('cookiecutter.replay.make_sure_path_exists') as mock_make_sure, \
         patch('cookiecutter.replay.get_file_name') as mock_get_file_name, \
         patch('builtins.open', mock_open()) as mock_file:
        mock_get_file_name.return_value = Path('/fake/replay/test_template.json')
        try:
            dump(replay_dir, template_name, context_without_cookiecutter)
            assert False, "Expected ValueError but none was raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'
            mock_make_sure.assert_called_once_with(replay_dir)
            mock_get_file_name.assert_not_called()
            mock_file.assert_not_called()


# LLM-generated content at query #42
#--------------------------

def test_dump_creates_directory_if_not_exists():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    assert Path(replay_dir).exists()
    Path(replay_dir).rmdir()

def test_dump_raises_error_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_dump_writes_correct_json_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    replay_file = Path(replay_dir) / f"{template_name}.json"
    assert replay_file.exists()
    with open(replay_file, encoding="utf-8") as infile:
        data = json.load(infile)
    assert data == context
    replay_file.unlink()
    Path(replay_dir).rmdir()

def test_dump_handles_template_name_with_json_extension():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    replay_file = Path(replay_dir) / template_name
    assert replay_file.exists()
    replay_file.unlink()
    Path(replay_dir).rmdir()


# LLM-generated content at query #43
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = test_replay_dir / f"{test_template_name}.json"
    test_context_without_cookiecutter = {"some_key": "some_value"}
    with patch('pathlib.Path.__truediv__', return_value=test_file_name):
        with patch('builtins.open', mock_open(read_data=json.dumps(test_context_without_cookiecutter))):
            try:
                load(test_replay_dir, test_template_name)
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #44
#--------------------------

def test_dump_creates_file_with_utf8_encoding():
    replay_dir = Path('test_replay')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'r', encoding='utf-8') as infile:
        content = infile.read()
    assert content.strip() == json.dumps(context, indent=2)
    replay_file.unlink()
    replay_dir.rmdir()


# LLM-generated content at query #45
#--------------------------

def test_dump_creates_directory_if_not_exists():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    assert Path(replay_dir).exists()
    Path(replay_dir).rmdir()

def test_dump_writes_correct_file():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = Path(replay_dir) / "test_template.json"
    assert expected_file.exists()
    with open(expected_file, "r", encoding="utf-8") as f:
        content = json.load(f)
    assert content == context
    expected_file.unlink()
    Path(replay_dir).rmdir()

def test_dump_handles_template_name_with_json_extension():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = Path(replay_dir) / "test_template.json"
    assert expected_file.exists()
    expected_file.unlink()
    Path(replay_dir).rmdir()

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_dump_creates_file_with_proper_indentation():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = Path(replay_dir) / "test_template.json"
    with open(expected_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) > 1
    assert lines[0].strip() == "{"
    expected_file.unlink()
    Path(replay_dir).rmdir()


