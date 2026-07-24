####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_get_file_name_with_path_and_no_json_suffix():
    replay_dir = Path("/some/dir")
    template_name = "template"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join(replay_dir, "template.json")
    assert result == expected

def test_get_file_name_with_path_and_json_suffix():
    replay_dir = Path("/some/dir")
    template_name = "template.json"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join(replay_dir, "template.json")
    assert result == expected

def test_get_file_name_with_str_and_no_json_suffix():
    replay_dir = "/some/dir"
    template_name = "template"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join(replay_dir, "template.json")
    assert result == expected

def test_get_file_name_with_str_and_json_suffix():
    replay_dir = "/some/dir"
    template_name = "template.json"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join(replay_dir, "template.json")
    assert result == expected

def test_get_file_name_with_empty_template_name():
    replay_dir = Path("/some/dir")
    template_name = ""
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join(replay_dir, ".json")
    assert result == expected

def test_get_file_name_with_template_name_already_ending_with_json():
    replay_dir = Path("/some/dir")
    template_name = "file.json"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join(replay_dir, "file.json")
    assert result == expected

def test_get_file_name_with_template_name_ending_with_dot_json_but_extra():
    replay_dir = Path("/some/dir")
    template_name = "file.json.txt"
    result = get_file_name(replay_dir, template_name)
    expected = os.path.join(replay_dir, "file.json.txt.json")
    assert result == expected


# LLM-generated content at query #2
#--------------------------

def test_load_success():
    replay_dir = '/tmp/test'
    template_name = 'template'
    expected_file = '/tmp/test/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context
    mock_open.assert_called_once_with(expected_file, encoding='utf-8')

def test_load_with_json_extension():
    replay_dir = '/tmp/test'
    template_name = 'template.json'
    expected_file = '/tmp/test/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context
    mock_open.assert_called_once_with(expected_file, encoding='utf-8')

def test_load_missing_cookiecutter():
    replay_dir = '/tmp/test'
    template_name = 'template'
    expected_file = '/tmp/test/template.json'
    invalid_context = {'key': 'value'}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch('builtins.open', mock_open):
        try:
            load(replay_dir, template_name)
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #3
#--------------------------

def test_dump_creates_directory_and_file():
    replay_dir = "test_replay"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = os.path.join(replay_dir, "template.json")
    assert os.path.exists(replay_dir)
    assert os.path.exists(expected_file)
    with open(expected_file, "r", encoding="utf-8") as infile:
        loaded = json.load(infile)
    assert loaded == context
    os.remove(expected_file)
    os.rmdir(replay_dir)

def test_dump_raises_error_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_dump_handles_existing_json_suffix():
    replay_dir = "test_replay"
    template_name = "template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = os.path.join(replay_dir, "template.json")
    assert os.path.exists(expected_file)
    with open(expected_file, "r", encoding="utf-8") as infile:
        loaded = json.load(infile)
    assert loaded == context
    os.remove(expected_file)
    os.rmdir(replay_dir)

def test_dump_creates_nested_directory():
    replay_dir = "nested/test/replay"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = os.path.join(replay_dir, "template.json")
    assert os.path.exists(replay_dir)
    assert os.path.exists(expected_file)
    os.remove(expected_file)
    os.rmdir("nested/test/replay")
    os.rmdir("nested/test")
    os.rmdir("nested")


# LLM-generated content at query #4
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import json
    from unittest.mock import mock_open, patch
    mock_context = {}
    mock_file_content = json.dumps(mock_context)
    mock_replay_file = Path("fake_file.json")
    with patch("pathlib.Path.open", mock_open(read_data=mock_file_content)):
        with patch("cookiecutter.replay.get_file_name", return_value=mock_replay_file):
            try:
                load(mock_replay_file, "template")
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #5
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import json
    import tempfile
    import os
    from unittest.mock import mock_open, patch

    test_dir = tempfile.mkdtemp()
    test_template = "test_template"
    test_file_path = Path(test_dir) / f"{test_template}.json"
    test_context = {"not_cookiecutter": "some_value"}
    mock_file_content = json.dumps(test_context)

    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.is_file', return_value=True):
                try:
                    import sys
                    sys.modules[__name__].load(test_dir, test_template)
                    assert False, "Expected ValueError"
                except ValueError as e:
                    assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #6
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


# LLM-generated content at query #7
#--------------------------

def test_load_context_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_context = {'cookiecutter': {'project_name': 'test'}}
    mock_data = json.dumps(test_context)
    with patch('builtins.open', mock_open(read_data=mock_data)):
        result = load('fake_dir', 'fake_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #8
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import tempfile
    from cookiecutter.replay import dump
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context_without_cookiecutter = {"key": "value"}
        try:
            dump(replay_dir, template_name, context_without_cookiecutter)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #9
#--------------------------

def test_context_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    context_with_cookiecutter = {'cookiecutter': {'project_name': 'test'}}
    mock_json_load = lambda x: context_with_cookiecutter
    with patch('builtins.open', mock_open(read_data='{}')):
        with patch('json.load', mock_json_load):
            result = load('fake_dir', 'fake_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #10
#--------------------------

def test_load_successful():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file_path = "/tmp/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file_path):
            result = load(replay_dir, template_name)
    assert result == expected_context


def test_load_with_json_extension():
    replay_dir = "/tmp"
    template_name = "template.json"
    expected_file_path = "/tmp/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file_path):
            result = load(replay_dir, template_name)
    assert result == expected_context


def test_load_missing_cookiecutter_key():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file_path = "/tmp/template.json"
    invalid_context = {"key": "value"}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file_path):
            try:
                load(replay_dir, template_name)
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"
            else:
                assert False


def test_load_file_not_found():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file_path = "/tmp/template.json"
    with mock.patch("os.path.join", return_value=expected_file_path):
        with mock.patch("builtins.open", side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
            except FileNotFoundError:
                pass
            else:
                assert False


# LLM-generated content at query #11
#--------------------------

def test_load_context_missing_cookiecutter_key():
    from pathlib import Path
    import json
    from unittest.mock import mock_open, patch

    mock_replay_dir = Path('/fake/dir')
    mock_template_name = 'test_template'
    mock_file_path = Path('/fake/dir/test_template.json')
    mock_context = {'some_key': 'some_value'}

    with patch('pathlib.Path.is_file', return_value=True):
        with patch('json.load', return_value=mock_context):
            with patch('builtins.open', mock_open()):
                try:
                    load(mock_replay_dir, mock_template_name)
                except ValueError as e:
                    assert str(e) == 'Context is required to contain a cookiecutter key'
                else:
                    assert False, "Expected ValueError not raised"


# LLM-generated content at query #12
#--------------------------

def test_load_valid_file():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"cookiecutter": {"key": "value"}}')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_file_without_json_extension():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"cookiecutter": {"key": "value"}}')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter_key():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{"key": "value"}')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            try:
                load(replay_dir, template_name)
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', side_effect=FileNotFoundError):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            try:
                load(replay_dir, template_name)
            except FileNotFoundError:
                pass

def test_load_invalid_json():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='invalid json')):
        with unittest.mock.patch('os.path.join', return_value='/tmp/template.json'):
            try:
                load(replay_dir, template_name)
            except json.JSONDecodeError:
                pass


# LLM-generated content at query #13
#--------------------------

def test_load_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_data = {'cookiecutter': {'project_name': 'test'}}
    mock_file_content = json.dumps(test_data)
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pathlib.Path.exists', return_value=True):
                from cookiecutter.replay import get_file_name
                with patch('cookiecutter.replay.get_file_name', return_value=Path('dummy.json')):
                    from cookiecutter.replay import load
                    result = load('dummy_dir', 'dummy_template')
                    assert 'cookiecutter' in result


# LLM-generated content at query #14
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    from pathlib import Path
    import json
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = Path("/fake/dir/test_template.json")
    mock_json_data = {"some_key": "some_value"}
    with patch('pathlib.Path.open', mock_open(read_data=json.dumps(mock_json_data))):
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pathlib.Path.exists', return_value=True):
                try:
                    load(test_replay_dir, test_template_name)
                    assert False, "Expected ValueError was not raised"
                except ValueError as e:
                    assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

def test_load_context_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_data = {'cookiecutter': {'project_name': 'test'}}
    mock_file_content = json.dumps(test_data)
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pathlib.Path.exists', return_value=True):
                from cookiecutter.replay import get_file_name
                with patch('cookiecutter.replay.get_file_name', return_value='dummy_path'):
                    from cookiecutter.replay import load
                    result = load('dummy_dir', 'dummy_template')
                    assert 'cookiecutter' in result


# LLM-generated content at query #17
#--------------------------

def test_load_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_data = {'cookiecutter': {'project_name': 'test'}}
    mock_file = mock_open(read_data=json.dumps(test_data))
    with patch('builtins.open', mock_file):
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pathlib.Path.exists', return_value=True):
                from cookiecutter.replay import get_file_name
                with patch('cookiecutter.replay.get_file_name', return_value=Path('dummy.json')):
                    from cookiecutter.replay import load
                    result = load('dummy_dir', 'dummy_template')
                    assert 'cookiecutter' in result


# LLM-generated content at query #18
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import tempfile
    from cookiecutter.replay import dump
    from cookiecutter.utils import make_sure_path_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context_without_cookiecutter = {"key": "value"}
        try:
            dump(replay_dir, template_name, context_without_cookiecutter)
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #19
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = Path('/tmp/test_replay')
    template_name = 'test_template'
    context = {'not_cookiecutter': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #20
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
    else:
        assert False, "Expected ValueError not raised"


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

def test_load_context_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_context = {'cookiecutter': {'project_name': 'test'}}
    mock_json_load = lambda x: test_context
    with patch('builtins.open', mock_open(read_data='{}')), patch('json.load', mock_json_load), patch('path.to.get_file_name', return_value='dummy_path'):
        result = load('dummy_dir', 'dummy_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #23
#--------------------------

def test_load_success():
    replay_dir = "/tmp/test"
    template_name = "template"
    expected_file = "/tmp/test/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context


def test_load_with_json_extension():
    replay_dir = "/tmp/test"
    template_name = "template.json"
    expected_file = "/tmp/test/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context


def test_load_missing_cookiecutter():
    replay_dir = "/tmp/test"
    template_name = "template"
    expected_file = "/tmp/test/template.json"
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
    replay_dir = "/tmp/test"
    template_name = "template"
    expected_file = "/tmp/test/template.json"
    with mock.patch("os.path.join", return_value=expected_file):
        with mock.patch("builtins.open", side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True


def test_load_json_decode_error():
    replay_dir = "/tmp/test"
    template_name = "template"
    expected_file = "/tmp/test/template.json"
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

def test_load_success():
    replay_dir = "/tmp/replay"
    template_name = "template"
    expected_file = "/tmp/replay/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = "/tmp/replay"
    template_name = "template.json"
    expected_file = "/tmp/replay/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = "/tmp/replay"
    template_name = "template"
    expected_file = "/tmp/replay/template.json"
    invalid_context = {"key": "value"}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file):
            try:
                load(replay_dir, template_name)
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_file_not_found():
    replay_dir = "/tmp/replay"
    template_name = "template"
    expected_file = "/tmp/replay/template.json"
    with mock.patch("os.path.join", return_value=expected_file):
        with mock.patch("builtins.open", side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
            except FileNotFoundError:
                pass

def test_load_invalid_json():
    replay_dir = "/tmp/replay"
    template_name = "template"
    expected_file = "/tmp/replay/template.json"
    mock_open = mock.mock_open(read_data="invalid json")
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.path.join", return_value=expected_file):
            try:
                load(replay_dir, template_name)
            except json.JSONDecodeError:
                pass


# LLM-generated content at query #25
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    from pathlib import Path
    import json
    import tempfile
    from unittest.mock import mock_open, patch

    test_dir = Path(tempfile.mkdtemp())
    test_template = "test_template"
    test_file_path = test_dir / f"{test_template}.json"
    test_context = {"some_key": "some_value"}

    with patch('json.load', return_value=test_context):
        with patch('builtins.open', mock_open()):
            with patch('pathlib.Path.open', mock_open()):
                with patch('__main__.get_file_name', return_value=test_file_path):
                    try:
                        load(test_dir, test_template)
                        assert False
                    except ValueError as e:
                        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #26
#--------------------------

def test_load_returns_context_with_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    mock_data = '{"cookiecutter": {"project_name": "test"}}'
    with patch('builtins.open', mock_open(read_data=mock_data)):
        with patch('pathlib.Path.is_file', return_value=True):
            from cookiecutter.replay import get_file_name
            with patch('cookiecutter.replay.get_file_name', return_value=Path('dummy.json')):
                from cookiecutter.replay import load
                result = load('dummy_dir', 'dummy_template')
                assert 'cookiecutter' in result


# LLM-generated content at query #27
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path('/fake/dir')
    test_template_name = 'test_template'
    fake_file_path = test_replay_dir / f'{test_template_name}.json'
    fake_json_content = '{"some_key": "some_value"}'
    with patch('pathlib.Path.__truediv__', return_value=fake_file_path):
        with patch('builtins.open', mock_open(read_data=fake_json_content)):
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #28
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import tempfile
    import json
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


# LLM-generated content at query #29
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path('/fake/dir')
    test_template_name = 'test_template'
    test_file_name = Path('/fake/dir/test_template.json')
    mock_json_data = {'some_key': 'some_value'}
    with patch('path.to.module.get_file_name', return_value=test_file_name) as mock_get_file_name:
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_json_data))) as mock_file:
            try:
                result = load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError but none was raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'
                mock_get_file_name.assert_called_once_with(test_replay_dir, test_template_name)
                mock_file.assert_called_once_with(test_file_name, encoding='utf-8')


# LLM-generated content at query #30
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    open_mock = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', open_mock):
        result = load(replay_dir, template_name)
    assert result == expected_context
    open_mock.assert_called_once_with(expected_file, encoding='utf-8')

def test_load_with_json_extension():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_file = '/tmp/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    open_mock = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', open_mock):
        result = load(replay_dir, template_name)
    assert result == expected_context
    open_mock.assert_called_once_with(expected_file, encoding='utf-8')

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
            pass

def test_load_json_decode_error():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    with mock.patch('builtins.open', side_effect=json.JSONDecodeError('msg', 'doc', 0)):
        try:
            load(replay_dir, template_name)
            assert False
        except json.JSONDecodeError:
            pass


# LLM-generated content at query #31
#--------------------------

def test_dump_creates_directory_and_writes_file():
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

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "my_template"
    context = {"other_key": "value"}
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
    with open(expected_file, "r", encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context
    os.remove(expected_file)
    os.rmdir(replay_dir)


# LLM-generated content at query #32
#--------------------------

def test_context_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_context = {'cookiecutter': {'project_name': 'test'}}
    mock_file_content = json.dumps(test_context)
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('pathlib.Path.is_file', return_value=True):
            from cookiecutter.replay import load
            result = load(Path('test_dir'), 'test_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #33
#--------------------------

def test_load_success():
    replay_dir = '/tmp/replay'
    template_name = 'template'
    expected_file = '/tmp/replay/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = '/tmp/replay'
    template_name = 'template.json'
    expected_file = '/tmp/replay/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = '/tmp/replay'
    template_name = 'template'
    expected_file = '/tmp/replay/template.json'
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
    replay_dir = '/tmp/replay'
    template_name = 'template'
    expected_file = '/tmp/replay/template.json'
    with mock.patch('os.path.join', return_value=expected_file):
        with mock.patch('builtins.open', side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True

def test_load_json_decode_error():
    replay_dir = '/tmp/replay'
    template_name = 'template'
    expected_file = '/tmp/replay/template.json'
    mock_open = mock.mock_open(read_data='invalid json')
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            try:
                load(replay_dir, template_name)
                assert False
            except json.JSONDecodeError:
                assert True


# LLM-generated content at query #34
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    mock_data = json.dumps({})
    with patch('builtins.open', mock_open(read_data=mock_data)), \
         patch('pathlib.Path.is_file', return_value=True), \
         patch('pathlib.Path.exists', return_value=True):
        try:
            load(Path('fake_dir'), 'fake_template')
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #35
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_data = {"some_key": "some_value"}
    mock_file = mock_open(read_data=json.dumps(test_data))
    with patch("builtins.open", mock_file):
        try:
            load(Path("fake_dir"), "fake_template")
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #36
#--------------------------

def test_cookiecutter_key_missing_raises_value_error():
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

def test_dump_creates_directory_if_not_exists():
    import tempfile
    import json
    import os
    from pathlib import Path
    from cookiecutter.replay import dump
    replay_dir = Path(tempfile.mkdtemp()) / "new_subdir"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    with open(expected_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == context

def test_dump_uses_existing_directory():
    import tempfile
    import json
    import os
    from pathlib import Path
    from cookiecutter.replay import dump
    replay_dir = Path(tempfile.mkdtemp())
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    with open(expected_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == context

def test_dump_raises_value_error_without_cookiecutter_key():
    import tempfile
    from pathlib import Path
    from cookiecutter.replay import dump
    replay_dir = Path(tempfile.mkdtemp())
    template_name = "my_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_dump_handles_template_name_with_json_extension():
    import tempfile
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    replay_dir = Path(tempfile.mkdtemp())
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = replay_dir / template_name
    assert expected_file.exists()
    with open(expected_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == context

def test_dump_writes_indented_json():
    import tempfile
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    replay_dir = Path(tempfile.mkdtemp())
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = replay_dir / f"{template_name}.json"
    with open(expected_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert content.strip().startswith("{")
    assert "\n  " in content
    loaded = json.loads(content)
    assert loaded == context


# LLM-generated content at query #38
#--------------------------

def test_load_context_missing_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    mock_file_name = Path("/fake/dir/test_template.json")
    with patch('__main__.get_file_name', return_value=mock_file_name) as mock_get_file:
        mock_data = json.dumps({"some_key": "some_value"})
        with patch('builtins.open', mock_open(read_data=mock_data)):
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #39
#--------------------------

def test_load_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_data = {'cookiecutter': {'project_name': 'test'}}
    mock_file = mock_open(read_data=json.dumps(test_data))
    with patch('builtins.open', mock_file):
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pathlib.Path.exists', return_value=True):
                from cookiecutter.replay import get_file_name
                with patch('cookiecutter.replay.get_file_name', return_value=Path('fake_path')):
                    from cookiecutter.replay import load
                    result = load('fake_dir', 'fake_template')
                    assert 'cookiecutter' in result


# LLM-generated content at query #40
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import json
    from unittest.mock import mock_open, patch
    test_dir = Path("test_dir")
    test_template = "test_template"
    test_file_path = test_dir / f"{test_template}.json"
    test_context = {"some_key": "some_value"}
    with patch("pathlib.Path.open", mock_open(read_data=json.dumps(test_context))):
        try:
            load(test_dir, test_template)
            assert False
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #41
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


# LLM-generated content at query #42
#--------------------------

def test_dump_creates_directory_and_writes_file():
    import tempfile
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    from cookiecutter.utils import make_sure_path_exists
    replay_dir = Path(tempfile.mkdtemp())
    template_name = "my_template"
    context = {"cookiecutter": {"project_name": "Test"}}
    dump(replay_dir, template_name, context)
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    with open(expected_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    assert content == context

def test_dump_raises_value_error_without_cookiecutter_key():
    import tempfile
    from pathlib import Path
    from cookiecutter.replay import dump
    replay_dir = Path(tempfile.mkdtemp())
    template_name = "my_template"
    context = {"project_name": "Test"}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_dump_handles_existing_json_extension():
    import tempfile
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    replay_dir = Path(tempfile.mkdtemp())
    template_name = "my_template.json"
    context = {"cookiecutter": {"project_name": "Test"}}
    dump(replay_dir, template_name, context)
    expected_file = replay_dir / template_name
    assert expected_file.exists()
    with open(expected_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    assert content == context

def test_dump_creates_nested_directories():
    import tempfile
    import json
    from pathlib import Path
    from cookiecutter.replay import dump
    base_dir = Path(tempfile.mkdtemp())
    replay_dir = base_dir / "nested" / "deep"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = replay_dir / f"{template_name}.json"
    assert expected_file.exists()
    with open(expected_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    assert content == context


# LLM-generated content at query #43
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_dir = Path("test_dir")
    test_name = "test_template"
    test_file_path = test_dir / f"{test_name}.json"
    mock_json_content = {}
    with patch('pathlib.Path.open', mock_open(read_data=json.dumps(mock_json_content))):
        with patch('pathlib.Path.is_file', return_value=True):
            try:
                load(test_dir, test_name)
                assert False, "Expected ValueError"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #44
#--------------------------

def test_load_returns_context_with_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_context = {'cookiecutter': {'project_name': 'test'}}
    json_data = json.dumps(test_context)
    mock_file = mock_open(read_data=json_data)
    with patch('builtins.open', mock_file):
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pathlib.Path.exists', return_value=True):
                result = load(Path('test_dir'), 'test_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #45
#--------------------------

def test_load_success():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file = "/tmp/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context
    mock_open.assert_called_once_with(expected_file, encoding="utf-8")

def test_load_with_json_extension():
    replay_dir = "/tmp"
    template_name = "template.json"
    expected_file = "/tmp/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context
    mock_open.assert_called_once_with(expected_file, encoding="utf-8")

def test_load_missing_cookiecutter():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file = "/tmp/template.json"
    invalid_context = {"key": "value"}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch("builtins.open", mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_file_not_found():
    replay_dir = "/tmp"
    template_name = "template"
    expected_file = "/tmp/template.json"
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
        try:
            load(replay_dir, template_name)
            assert False
        except json.JSONDecodeError:
            assert True


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

def test_get_file_name_with_str_and_no_json_suffix():
    replay_dir = "/some/dir"
    template_name = "template"
    result = get_file_name(replay_dir, template_name)
    assert result == "/some/dir/template.json"

def test_get_file_name_with_json_suffix():
    replay_dir = Path("/another/dir")
    template_name = "template.json"
    result = get_file_name(replay_dir, template_name)
    assert result == "/another/dir/template.json"

def test_get_file_name_with_dot_json_in_middle():
    replay_dir = "/path"
    template_name = "my.template.json"
    result = get_file_name(replay_dir, template_name)
    assert result == "/path/my.template.json"

def test_get_file_name_empty_template():
    replay_dir = Path("/empty")
    template_name = ""
    result = get_file_name(replay_dir, template_name)
    assert result == "/empty/.json"


# LLM-generated content at query #2
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
    file_path = Path(replay_dir) / "test_template.json"
    assert file_path.exists()
    with open(file_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    assert data == context
    file_path.unlink()
    Path(replay_dir).rmdir()

def test_dump_handles_template_name_with_json_extension():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    file_path = Path(replay_dir) / "test_template.json"
    assert file_path.exists()
    file_path.unlink()
    Path(replay_dir).rmdir()


# LLM-generated content at query #3
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


# LLM-generated content at query #4
#--------------------------

def test_load_success():
    replay_dir = '/tmp/replay'
    template_name = 'template'
    expected_file = '/tmp/replay/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = '/tmp/replay'
    template_name = 'template.json'
    expected_file = '/tmp/replay/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = '/tmp/replay'
    template_name = 'template'
    expected_file = '/tmp/replay/template.json'
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
    replay_dir = '/tmp/replay'
    template_name = 'template'
    expected_file = '/tmp/replay/template.json'
    with mock.patch('os.path.join', return_value=expected_file):
        with mock.patch('builtins.open', side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True

def test_load_json_decode_error():
    replay_dir = '/tmp/replay'
    template_name = 'template'
    expected_file = '/tmp/replay/template.json'
    mock_open = mock.mock_open(read_data='invalid json')
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file):
            try:
                load(replay_dir, template_name)
                assert False
            except json.JSONDecodeError:
                assert True


# LLM-generated content at query #5
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context
    mock_open.assert_called_once_with(expected_file, encoding='utf-8')

def test_load_with_json_extension():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_file = '/tmp/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context
    mock_open.assert_called_once_with(expected_file, encoding='utf-8')

def test_load_missing_cookiecutter():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_file = '/tmp/template.json'
    invalid_context = {'key': 'value'}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch('builtins.open', mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_path_object():
    replay_dir = Path('/tmp')
    template_name = 'template'
    expected_file = '/tmp/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context
    mock_open.assert_called_once_with(expected_file, encoding='utf-8')


# LLM-generated content at query #6
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
        with patch('builtins.open', mock_open(read_data=json.dumps(test_context_without_cookiecutter))):
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'
                mock_get_file_name.assert_called_once_with(test_replay_dir, test_template_name)


# LLM-generated content at query #7
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path('/fake/dir')
    test_template_name = 'test_template'
    fake_file_path = Path('/fake/dir/test_template.json')
    fake_context = {'some_key': 'some_value'}
    with patch('__main__.get_file_name', return_value=fake_file_path) as mock_get_file_name:
        with patch('builtins.open', mock_open(read_data=json.dumps(fake_context))) as mock_file:
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'
                mock_get_file_name.assert_called_once_with(test_replay_dir, test_template_name)
                mock_file.assert_called_once_with(fake_file_path, encoding='utf-8')


# LLM-generated content at query #8
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_key_missing():
    from cookiecutter.replay import dump
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context_without_key = {"some_key": "value"}
        try:
            dump(replay_dir, template_name, context_without_key)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #9
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = Path("/fake/dir/test_template.json")
    mock_json_data = {"some_key": "some_value"}
    with patch('pathlib.Path.open', mock_open(read_data=json.dumps(mock_json_data))):
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pathlib.Path.__str__', return_value=str(test_file_name)):
                try:
                    load(test_replay_dir, test_template_name)
                    assert False, "Expected ValueError was not raised"
                except ValueError as e:
                    assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_path = Path("/fake/dir/test_template.json")
    test_context_without_cookiecutter = {"some_key": "some_value"}
    with patch('pathlib.Path.is_file', return_value=True), \
         patch('__main__.get_file_name', return_value=test_file_path), \
         patch('builtins.open', mock_open(read_data=json.dumps(test_context_without_cookiecutter))):
        try:
            load(test_replay_dir, test_template_name)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #12
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = Path('test_replay')
    template_name = 'test_template'
    context = {'not_cookiecutter': {}}
    try:
        dump(replay_dir, template_name, context)
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #13
#--------------------------

def test_dump_raises_value_error_when_cookiecutter_not_in_context():
    replay_dir = Path('test_replay')
    template_name = 'test_template'
    context = {'not_cookiecutter': {}}
    try:
        dump(replay_dir, template_name, context)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #14
#--------------------------

def test_load_returns_context_with_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_context = {'cookiecutter': {'project_name': 'Test'}}
    json_data = json.dumps(test_context)
    with patch('builtins.open', mock_open(read_data=json_data)):
        from cookiecutter.replay import load
        result = load(Path('test_dir'), 'test_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #15
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'test_template'
    expected_file_path = '/tmp/test_template.json'
    mock_json_content = '{"cookiecutter": {"key": "value"}}'
    mock_open = mock.mock_open(read_data=mock_json_content)
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file_path):
            result = load(replay_dir, template_name)
    assert result == {"cookiecutter": {"key": "value"}}
    mock_open.assert_called_once_with(expected_file_path, encoding='utf-8')


def test_load_with_json_suffix():
    replay_dir = '/tmp'
    template_name = 'test_template.json'
    expected_file_path = '/tmp/test_template.json'
    mock_json_content = '{"cookiecutter": {"key": "value"}}'
    mock_open = mock.mock_open(read_data=mock_json_content)
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file_path):
            result = load(replay_dir, template_name)
    assert result == {"cookiecutter": {"key": "value"}}
    mock_open.assert_called_once_with(expected_file_path, encoding='utf-8')


def test_load_missing_cookiecutter():
    replay_dir = '/tmp'
    template_name = 'test_template'
    expected_file_path = '/tmp/test_template.json'
    mock_json_content = '{"other_key": "value"}'
    mock_open = mock.mock_open(read_data=mock_json_content)
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file_path):
            try:
                load(replay_dir, template_name)
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'test_template'
    expected_file_path = '/tmp/test_template.json'
    with mock.patch('os.path.join', return_value=expected_file_path):
        with mock.patch('builtins.open', side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True


def test_load_json_decode_error():
    replay_dir = '/tmp'
    template_name = 'test_template'
    expected_file_path = '/tmp/test_template.json'
    mock_json_content = 'invalid json'
    mock_open = mock.mock_open(read_data=mock_json_content)
    with mock.patch('builtins.open', mock_open):
        with mock.patch('os.path.join', return_value=expected_file_path):
            with mock.patch('json.load', side_effect=json.JSONDecodeError('', '', 0)):
                try:
                    load(replay_dir, template_name)
                    assert False
                except json.JSONDecodeError:
                    assert True


# LLM-generated content at query #16
#--------------------------

def test_load_success():
    replay_dir = "/tmp/replay"
    template_name = "template"
    expected_file = "/tmp/replay/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context
    mock_open.assert_called_once_with(expected_file, encoding="utf-8")

def test_load_with_json_extension():
    replay_dir = "/tmp/replay"
    template_name = "template.json"
    expected_file = "/tmp/replay/template.json"
    expected_context = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch("builtins.open", mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context
    mock_open.assert_called_once_with(expected_file, encoding="utf-8")

def test_load_missing_cookiecutter():
    replay_dir = "/tmp/replay"
    template_name = "template"
    expected_file = "/tmp/replay/template.json"
    invalid_context = {"key": "value"}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch("builtins.open", mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_file_not_found():
    replay_dir = "/tmp/replay"
    template_name = "template"
    expected_file = "/tmp/replay/template.json"
    with mock.patch("builtins.open", side_effect=FileNotFoundError):
        try:
            load(replay_dir, template_name)
            assert False
        except FileNotFoundError:
            pass

def test_load_invalid_json():
    replay_dir = "/tmp/replay"
    template_name = "template"
    expected_file = "/tmp/replay/template.json"
    mock_open = mock.mock_open(read_data="invalid json")
    with mock.patch("builtins.open", mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except json.JSONDecodeError:
            pass


# LLM-generated content at query #17
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
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


# LLM-generated content at query #18
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = Path("/fake/dir/test_template.json")
    mock_json_data = {"some_key": "some_value"}
    with patch('pathlib.Path.is_file', return_value=True):
        with patch('__main__.get_file_name', return_value=test_file_name) as mock_get_file:
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_json_data))) as mock_file:
                try:
                    load(test_replay_dir, test_template_name)
                    assert False, "Expected ValueError was not raised"
                except ValueError as e:
                    assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

def test_load_success():
    replay_dir = "/tmp/replay"
    template_name = "template"
    expected_data = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_data))
    with mock.patch("builtins.open", mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_data

def test_load_with_json_extension():
    replay_dir = "/tmp/replay"
    template_name = "template.json"
    expected_data = {"cookiecutter": {"key": "value"}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_data))
    with mock.patch("builtins.open", mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_data

def test_load_missing_cookiecutter():
    replay_dir = "/tmp/replay"
    template_name = "template"
    invalid_data = {"key": "value"}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_data))
    with mock.patch("builtins.open", mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_file_not_found():
    replay_dir = "/tmp/replay"
    template_name = "template"
    with mock.patch("builtins.open", side_effect=FileNotFoundError):
        try:
            load(replay_dir, template_name)
            assert False
        except FileNotFoundError:
            assert True

def test_load_json_decode_error():
    replay_dir = "/tmp/replay"
    template_name = "template"
    mock_open = mock.mock_open(read_data="invalid json")
    with mock.patch("builtins.open", mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except json.JSONDecodeError:
            assert True


# LLM-generated content at query #21
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_dir = Path("/fake/dir")
    test_template = "test_template"
    fake_json_content = {"not_cookiecutter": "some_value"}
    with patch("builtins.open", mock_open(read_data=json.dumps(fake_json_content))):
        with patch("pathlib.Path.is_file", return_value=True):
            try:
                load(test_dir, test_template)
                assert False, "Expected ValueError was not raised"
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
    test_context = {"some_key": "some_value"}
    with patch('__main__.get_file_name', return_value=test_file_name) as mock_get_file_name:
        with patch('builtins.open', mock_open(read_data=json.dumps(test_context))) as mock_file:
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'
                mock_get_file_name.assert_called_once_with(test_replay_dir, test_template_name)
                mock_file.assert_called_once_with(test_file_name, encoding="utf-8")


# LLM-generated content at query #23
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    from pathlib import Path
    import json
    import tempfile
    import os
    from unittest.mock import patch, mock_open

    test_dir = Path(tempfile.mkdtemp())
    test_template = "test_template"
    expected_file = test_dir / f"{test_template}.json"
    test_data = {"some_key": "some_value"}
    json_content = json.dumps(test_data)

    with patch('builtins.open', mock_open(read_data=json_content)):
        with patch('pathlib.Path.exists', return_value=True):
            try:
                result = load(test_dir, test_template)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #24
#--------------------------

def test_dump_creates_directory_if_not_exists():
    from pathlib import Path
    import tempfile
    import json
    from cookiecutter.replay import dump
    from cookiecutter.utils import make_sure_path_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir) / "subdir"
        template_name = "my_template"
        context = {"cookiecutter": {"key": "value"}}
        dump(replay_dir, template_name, context)
        expected_file = replay_dir / f"{template_name}.json"
        assert expected_file.exists()
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == context

def test_dump_raises_value_error_without_cookiecutter_key():
    from pathlib import Path
    import tempfile
    from cookiecutter.replay import dump
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "my_template"
        context = {"key": "value"}
        try:
            dump(replay_dir, template_name, context)
            assert False
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'

def test_dump_writes_correct_json_content():
    from pathlib import Path
    import tempfile
    import json
    from cookiecutter.replay import dump
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "template"
        context = {"cookiecutter": {"project": "test", "version": "1.0"}}
        dump(replay_dir, template_name, context)
        expected_file = replay_dir / f"{template_name}.json"
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == context

def test_dump_handles_template_name_with_json_extension():
    from pathlib import Path
    import tempfile
    import json
    from cookiecutter.replay import dump
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "template.json"
        context = {"cookiecutter": {"data": "test"}}
        dump(replay_dir, template_name, context)
        expected_file = replay_dir / template_name
        assert expected_file.exists()
        with open(expected_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == context


# LLM-generated content at query #25
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    context_without_cookiecutter = {"some_key": "some_value"}
    json_data = json.dumps(context_without_cookiecutter)
    with patch("builtins.open", mock_open(read_data=json_data)), patch("pathlib.Path.is_file", return_value=True), patch("pathlib.Path.exists", return_value=True):
        try:
            load(Path("fake_dir"), "fake_template")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


# LLM-generated content at query #26
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(expected_context))
    with unittest.mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(expected_context))
    with unittest.mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = '/tmp'
    template_name = 'template'
    invalid_context = {'key': 'value'}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(invalid_context))
    with unittest.mock.patch('builtins.open', mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', side_effect=FileNotFoundError):
        try:
            load(replay_dir, template_name)
            assert False
        except FileNotFoundError:
            assert True

def test_load_json_decode_error():
    replay_dir = '/tmp'
    template_name = 'template'
    mock_open = unittest.mock.mock_open(read_data='invalid json')
    with unittest.mock.patch('builtins.open', mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except json.JSONDecodeError:
            assert True


# LLM-generated content at query #27
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = test_replay_dir / f"{test_template_name}.json"
    test_context = {"some_key": "some_value"}
    with patch('__main__.get_file_name', return_value=test_file_name) as mock_get_file_name:
        with patch('builtins.open', mock_open(read_data=json.dumps(test_context))) as mock_file:
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'
                mock_get_file_name.assert_called_once_with(test_replay_dir, test_template_name)
                mock_file.assert_called_once_with(test_file_name, encoding="utf-8")


# LLM-generated content at query #28
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
            with patch('pathlib.Path.__truediv__', return_value=test_file_name):
                try:
                    load(test_replay_dir, test_template_name)
                    assert False
                except ValueError as e:
                    assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #29
#--------------------------

def test_dump_creates_directory_if_not_exists():
    replay_dir = "test_replay"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    assert Path(replay_dir).exists()
    Path(replay_dir).rmdir()

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"

def test_dump_writes_correct_json_file():
    replay_dir = "test_replay"
    template_name = "template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    replay_file = Path(replay_dir) / "template.json"
    assert replay_file.exists()
    with open(replay_file, encoding="utf-8") as infile:
        data = json.load(infile)
    assert data == context
    replay_file.unlink()
    Path(replay_dir).rmdir()

def test_dump_handles_template_name_with_json_extension():
    replay_dir = "test_replay"
    template_name = "template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    replay_file = Path(replay_dir) / "template.json"
    assert replay_file.exists()
    replay_file.unlink()
    Path(replay_dir).rmdir()


# LLM-generated content at query #30
#--------------------------

def test_load_success():
    replay_dir = '/tmp/test'
    template_name = 'template'
    expected_file = '/tmp/test/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context
    mock_open.assert_called_once_with(expected_file, encoding='utf-8')

def test_load_with_json_extension():
    replay_dir = '/tmp/test'
    template_name = 'template.json'
    expected_file = '/tmp/test/template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    mock_open = mock.mock_open(read_data=json.dumps(expected_context))
    with mock.patch('builtins.open', mock_open):
        result = load(replay_dir, template_name)
    assert result == expected_context
    mock_open.assert_called_once_with(expected_file, encoding='utf-8')

def test_load_missing_cookiecutter():
    replay_dir = '/tmp/test'
    template_name = 'template'
    expected_file = '/tmp/test/template.json'
    invalid_context = {'key': 'value'}
    mock_open = mock.mock_open(read_data=json.dumps(invalid_context))
    with mock.patch('builtins.open', mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_not_found():
    replay_dir = '/tmp/test'
    template_name = 'template'
    expected_file = '/tmp/test/template.json'
    with mock.patch('builtins.open', side_effect=FileNotFoundError):
        try:
            load(replay_dir, template_name)
            assert False
        except FileNotFoundError:
            pass

def test_load_json_decode_error():
    replay_dir = '/tmp/test'
    template_name = 'template'
    expected_file = '/tmp/test/template.json'
    with mock.patch('builtins.open', side_effect=json.JSONDecodeError('msg', 'doc', 0)):
        try:
            load(replay_dir, template_name)
            assert False
        except json.JSONDecodeError:
            pass


# LLM-generated content at query #31
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
            with patch('pathlib.Path.exists', return_value=True):
                try:
                    load(test_replay_dir, test_template_name)
                    assert False
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
    test_file_name = test_replay_dir / f"{test_template_name}.json"
    mock_json_data = {"some_key": "some_value"}
    with patch('pathlib.Path.open', mock_open(read_data=json.dumps(mock_json_data))):
        with patch('pathlib.Path.__truediv__', return_value=test_file_name):
            try:
                load(test_replay_dir, test_template_name)
                assert False
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #33
#--------------------------

def test_context_contains_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_context = {'cookiecutter': {'project_name': 'test'}}
    mock_file_content = json.dumps(test_context)
    with patch('builtins.open', mock_open(read_data=mock_file_content)):
        with patch('pathlib.Path.is_file', return_value=True):
            result = load('fake_dir', 'fake_template')
    assert 'cookiecutter' in result


# LLM-generated content at query #34
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


# LLM-generated content at query #35
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
        with patch('pathlib.Path.__truediv__', return_value=test_file_name):
            try:
                load(test_replay_dir, test_template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #36
#--------------------------

def test_load_raises_value_error_when_cookiecutter_not_in_context():
    from pathlib import Path
    import json
    import tempfile
    import os
    from unittest.mock import patch

    def mock_get_file_name(replay_dir, template_name):
        return str(Path(replay_dir) / f"{template_name}.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        replay_file = replay_dir / f"{template_name}.json"
        context_without_cookiecutter = {"some_key": "some_value"}
        with open(replay_file, "w", encoding="utf-8") as f:
            json.dump(context_without_cookiecutter, f)
        with patch('__main__.get_file_name', side_effect=mock_get_file_name):
            try:
                load(replay_dir, template_name)
                assert False, "Expected ValueError was not raised"
            except ValueError as e:
                assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #37
#--------------------------

def test_load_returns_context_with_cookiecutter_key():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    context_data = {'cookiecutter': {'project_name': 'test'}}
    mock_json_load = lambda x: context_data
    with patch('builtins.open', mock_open(read_data='{}')), patch('json.load', mock_json_load), patch('test_module.get_file_name', return_value='dummy_path'):
        from test_module import load
        result = load('dummy_dir', 'dummy_template')
        assert 'cookiecutter' in result


# LLM-generated content at query #38
#--------------------------

def test_load_raises_value_error_when_cookiecutter_key_missing():
    import json
    from pathlib import Path
    from unittest.mock import mock_open, patch
    test_replay_dir = Path("/fake/dir")
    test_template_name = "test_template"
    test_file_name = "/fake/dir/test_template.json"
    test_context_without_cookiecutter = {"some_key": "some_value"}
    with patch('pathlib.Path.is_file', return_value=True), \
         patch('json.load', return_value=test_context_without_cookiecutter), \
         patch('builtins.open', mock_open()):
        try:
            load(test_replay_dir, test_template_name)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'


# LLM-generated content at query #39
#--------------------------

def test_load_success():
    replay_dir = "/tmp"
    template_name = "test_template"
    expected_file = "/tmp/test_template.json"
    mock_data = {"cookiecutter": {"project_name": "Test"}}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(mock_data))
    with unittest.mock.patch("builtins.open", mock_open):
        result = load(replay_dir, template_name)
    assert result == mock_data
    mock_open.assert_called_once_with(expected_file, encoding="utf-8")

def test_load_with_json_extension():
    replay_dir = "/tmp"
    template_name = "test_template.json"
    expected_file = "/tmp/test_template.json"
    mock_data = {"cookiecutter": {"key": "value"}}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(mock_data))
    with unittest.mock.patch("builtins.open", mock_open):
        result = load(replay_dir, template_name)
    assert result == mock_data
    mock_open.assert_called_once_with(expected_file, encoding="utf-8")

def test_load_missing_cookiecutter():
    replay_dir = "/tmp"
    template_name = "invalid_template"
    mock_data = {"some_key": "some_value"}
    mock_open = unittest.mock.mock_open(read_data=json.dumps(mock_data))
    with unittest.mock.patch("builtins.open", mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_file_not_found():
    replay_dir = "/tmp"
    template_name = "nonexistent"
    with unittest.mock.patch("builtins.open", side_effect=FileNotFoundError):
        try:
            load(replay_dir, template_name)
            assert False
        except FileNotFoundError:
            assert True

def test_load_json_decode_error():
    replay_dir = "/tmp"
    template_name = "corrupted"
    mock_open = unittest.mock.mock_open(read_data="invalid json")
    with unittest.mock.patch("builtins.open", mock_open):
        try:
            load(replay_dir, template_name)
            assert False
        except json.JSONDecodeError:
            assert True


# LLM-generated content at query #40
#--------------------------

def test_dump_creates_directory():
    replay_dir = "/tmp/test_replay"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    assert Path(replay_dir).exists()


def test_dump_writes_correct_file():
    replay_dir = "/tmp/test_replay2"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = Path(replay_dir) / f"{template_name}.json"
    assert expected_file.exists()


def test_dump_handles_existing_json_extension():
    replay_dir = "/tmp/test_replay3"
    template_name = "my_template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = Path(replay_dir) / template_name
    assert expected_file.exists()


def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = "/tmp/test_replay4"
    template_name = "my_template"
    context = {"not_cookiecutter": {"key": "value"}}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert "cookiecutter" in str(e)


def test_dump_writes_correct_json_content():
    replay_dir = "/tmp/test_replay5"
    template_name = "my_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    expected_file = Path(replay_dir) / f"{template_name}.json"
    with open(expected_file, "r", encoding="utf-8") as infile:
        content = json.load(infile)
    assert content == context


# LLM-generated content at query #41
#--------------------------

def test_dump_creates_file_with_cookiecutter_key():
    replay_dir = Path('test_replay')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()
    with open(replay_file, 'r', encoding='utf-8') as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context
    replay_file.unlink()
    replay_dir.rmdir()


# LLM-generated content at query #42
#--------------------------

def test_load_success():
    replay_dir = "/tmp"
    template_name = "test_template"
    expected_content = {"cookiecutter": {"key": "value"}}
    with patch("builtins.open", mock_open(read_data=json.dumps(expected_content))):
        with patch("os.path.join", return_value="/tmp/test_template.json"):
            result = load(replay_dir, template_name)
    assert result == expected_content

def test_load_with_json_extension():
    replay_dir = "/tmp"
    template_name = "test_template.json"
    expected_content = {"cookiecutter": {"key": "value"}}
    with patch("builtins.open", mock_open(read_data=json.dumps(expected_content))):
        with patch("os.path.join", return_value="/tmp/test_template.json"):
            result = load(replay_dir, template_name)
    assert result == expected_content

def test_load_missing_cookiecutter():
    replay_dir = "/tmp"
    template_name = "test_template"
    invalid_content = {"key": "value"}
    with patch("builtins.open", mock_open(read_data=json.dumps(invalid_content))):
        with patch("os.path.join", return_value="/tmp/test_template.json"):
            try:
                load(replay_dir, template_name)
                assert False
            except ValueError as e:
                assert str(e) == "Context is required to contain a cookiecutter key"

def test_load_file_not_found():
    replay_dir = "/tmp"
    template_name = "test_template"
    with patch("os.path.join", return_value="/tmp/test_template.json"):
        with patch("builtins.open", side_effect=FileNotFoundError):
            try:
                load(replay_dir, template_name)
                assert False
            except FileNotFoundError:
                assert True

def test_load_json_decode_error():
    replay_dir = "/tmp"
    template_name = "test_template"
    with patch("builtins.open", mock_open(read_data="invalid json")):
        with patch("os.path.join", return_value="/tmp/test_template.json"):
            try:
                load(replay_dir, template_name)
                assert False
            except json.JSONDecodeError:
                assert True


# LLM-generated content at query #43
#--------------------------

def test_dump_creates_directory_and_writes_file():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    expected_file = '/tmp/test_replay/test_template.json'
    assert os.path.exists(replay_dir)
    assert os.path.exists(expected_file)
    with open(expected_file, 'r', encoding='utf-8') as infile:
        content = json.load(infile)
    assert content == context

def test_dump_raises_error_without_cookiecutter_key():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template'
    context = {'key': 'value'}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == 'Context is required to contain a cookiecutter key'

def test_dump_handles_existing_json_extension():
    replay_dir = '/tmp/test_replay'
    template_name = 'test_template.json'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    expected_file = '/tmp/test_replay/test_template.json'
    assert os.path.exists(expected_file)
    with open(expected_file, 'r', encoding='utf-8') as infile:
        content = json.load(infile)
    assert content == context


# LLM-generated content at query #44
#--------------------------

def test_load_success():
    replay_dir = '/tmp'
    template_name = 'template'
    expected_context = {'cookiecutter': {'key': 'value'}}
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(expected_context))):
        result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_with_json_extension():
    replay_dir = '/tmp'
    template_name = 'template.json'
    expected_context = {'cookiecutter': {'key': 'value'}}
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(expected_context))):
        result = load(replay_dir, template_name)
    assert result == expected_context

def test_load_missing_cookiecutter():
    replay_dir = '/tmp'
    template_name = 'template'
    invalid_context = {'key': 'value'}
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(invalid_context))):
        try:
            load(replay_dir, template_name)
        except ValueError as e:
            assert str(e) == 'Context is required to contain a cookiecutter key'

def test_load_file_not_found():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', side_effect=FileNotFoundError):
        try:
            load(replay_dir, template_name)
        except FileNotFoundError:
            pass

def test_load_json_decode_error():
    replay_dir = '/tmp'
    template_name = 'template'
    with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='invalid json')):
        try:
            load(replay_dir, template_name)
        except json.JSONDecodeError:
            pass


# LLM-generated content at query #45
#--------------------------

def test_dump_creates_file_with_cookiecutter_key():
    replay_dir = Path('test_replay')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()
    with open(replay_file, 'r', encoding='utf-8') as infile:
        loaded = json.load(infile)
    assert loaded == context
    replay_file.unlink()
    replay_dir.rmdir()


# LLM-generated content at query #46
#--------------------------

def test_dump_creates_directory_if_not_exists():
    replay_dir = "test_replay"
    template_name = "test_template"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    assert os.path.exists(replay_dir)
    os.rmdir(replay_dir)

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
    file_path = os.path.join(replay_dir, "test_template.json")
    with open(file_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    assert data == context
    os.remove(file_path)
    os.rmdir(replay_dir)

def test_dump_handles_template_name_with_json_extension():
    replay_dir = "test_replay"
    template_name = "test_template.json"
    context = {"cookiecutter": {"key": "value"}}
    dump(replay_dir, template_name, context)
    file_path = os.path.join(replay_dir, "test_template.json")
    assert os.path.exists(file_path)
    os.remove(file_path)
    os.rmdir(replay_dir)


# LLM-generated content at query #47
#--------------------------

def test_dump_creates_directory_and_writes_file():
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

def test_dump_raises_value_error_without_cookiecutter_key():
    replay_dir = "test_replay"
    template_name = "my_template"
    context = {"key": "value"}
    try:
        dump(replay_dir, template_name, context)
        assert False
    except ValueError as e:
        assert str(e) == "Context is required to contain a cookiecutter key"
    assert not os.path.exists(replay_dir)

def test_dump_handles_existing_json_extension():
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

def test_dump_uses_existing_directory():
    replay_dir = "test_replay"
    os.makedirs(replay_dir, exist_ok=True)
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


# LLM-generated content at query #48
#--------------------------

def test_dump_creates_file_with_cookiecutter_key():
    replay_dir = Path('test_replay')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()
    with open(replay_file, 'r', encoding='utf-8') as infile:
        loaded_context = json.load(infile)
    assert loaded_context == context
    replay_file.unlink()
    replay_dir.rmdir()


# LLM-generated content at query #49
#--------------------------

def test_dump_creates_file_with_cookiecutter_key():
    replay_dir = Path('test_replay')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    assert replay_file.exists()
    with open(replay_file, 'r', encoding='utf-8') as infile:
        loaded = json.load(infile)
    assert loaded == context
    replay_file.unlink()
    replay_dir.rmdir()


# LLM-generated content at query #50
#--------------------------

def test_dump_creates_file_with_utf8_encoding():
    replay_dir = Path('test_replay')
    template_name = 'test_template'
    context = {'cookiecutter': {'key': 'value'}}
    dump(replay_dir, template_name, context)
    replay_file = get_file_name(replay_dir, template_name)
    with open(replay_file, 'r', encoding='utf-8') as infile:
        content = infile.read()
    assert content == '{\n  "cookiecutter": {\n    "key": "value"\n  }\n}'
    replay_file.unlink()
    replay_dir.rmdir()


# LLM-generated content at query #51
#--------------------------

```python
def test_dump_raises_value_error_when_cookiecutter_key_missing():
    from cookiecutter.replay import dump
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        replay_dir = Path(tmpdir)
        template_name = "test_template"
        context_without_cookiecutter = {"some_key": "some_value"}
        try:
            dump(replay_dir, template_name, context_without_cookiecutter)
            assert False, "Expected ValueError was not raised"
        except ValueError as e:
            assert str(e) == "Context is required to contain a cookiecutter key"


