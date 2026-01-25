####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_path_constructor_default_platform():
    path = Path()
    assert path.platform == sys.platform
    assert path.seed is not None or path.seed is MissingSeed
    assert path.random is not None


def test_path_constructor_linux_platform():
    path = Path(platform="linux")
    assert path.platform == "linux"
    assert isinstance(path._pathlib_home, PurePosixPath)


def test_path_constructor_darwin_platform():
    path = Path(platform="darwin")
    assert path.platform == "darwin"
    assert isinstance(path._pathlib_home, PurePosixPath)


def test_path_constructor_win32_platform():
    path = Path(platform="win32")
    assert path.platform == "win32"
    assert isinstance(path._pathlib_home, PureWindowsPath)


def test_path_constructor_win64_platform():
    path = Path(platform="win64")
    assert path.platform == "win64"
    assert isinstance(path._pathlib_home, PureWindowsPath)


def test_path_constructor_freebsd_platform():
    path = Path(platform="freebsd11")
    assert path.platform == "freebsd"
    assert isinstance(path._pathlib_home, PurePosixPath)


def test_path_constructor_with_seed():
    path = Path(platform="linux", seed=42)
    assert path.platform == "linux"
    assert path.seed == 42


def test_path_constructor_with_custom_random():
    custom_random = _random.Random()
    path = Path(platform="linux", random=custom_random)
    assert path.platform == "linux"
    assert path.random is custom_random


def test_path_constructor_meta_name():
    path = Path()
    assert path.Meta.name == "path"


def test_path_constructor_pathlib_home_linux():
    path = Path(platform="linux")
    assert str(path._pathlib_home) == "/home"


def test_path_constructor_pathlib_home_windows():
    path = Path(platform="win32")
    assert "Users" in str(path._pathlib_home)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_path_constructor_default_platform():
    path = Path()
    assert path.platform == sys.platform
    assert path.seed is MissingSeed
    assert path.random is not None


def test_path_constructor_linux_platform():
    path = Path(platform="linux")
    assert path.platform == "linux"
    assert isinstance(path._pathlib_home, PurePosixPath)


def test_path_constructor_darwin_platform():
    path = Path(platform="darwin")
    assert path.platform == "darwin"
    assert isinstance(path._pathlib_home, PurePosixPath)


def test_path_constructor_win32_platform():
    path = Path(platform="win32")
    assert path.platform == "win32"
    assert isinstance(path._pathlib_home, PureWindowsPath)


def test_path_constructor_win64_platform():
    path = Path(platform="win64")
    assert path.platform == "win64"
    assert isinstance(path._pathlib_home, PureWindowsPath)


def test_path_constructor_freebsd_platform():
    path = Path(platform="freebsd11")
    assert path.platform == "freebsd"
    assert isinstance(path._pathlib_home, PurePosixPath)


def test_path_constructor_with_seed():
    path = Path(platform="linux", seed=42)
    assert path.platform == "linux"
    assert path.seed == 42


def test_path_constructor_with_custom_random():
    custom_random = _random.Random()
    path = Path(platform="linux", random=custom_random)
    assert path.random is custom_random
    assert path.platform == "linux"


def test_path_constructor_home_path_linux():
    path = Path(platform="linux")
    home_str = str(path._pathlib_home)
    assert "home" in home_str


def test_path_constructor_home_path_windows():
    path = Path(platform="win32")
    home_str = str(path._pathlib_home)
    assert "Users" in home_str


def test_path_constructor_meta_name():
    path = Path(platform="linux")
    assert hasattr(Path, "Meta")
    assert Path.Meta.name == "path"


