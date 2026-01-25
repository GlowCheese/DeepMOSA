####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_path_default_initialization():
    path = Path()
    assert path.platform == sys.platform
    assert isinstance(path._pathlib_home, (PureWindowsPath, PurePosixPath))
    assert str(path._pathlib_home) == PLATFORMS[sys.platform]["home"]

def test_path_custom_platform():
    path = Path(platform="linux")
    assert path.platform == "linux"
    assert isinstance(path._pathlib_home, PurePosixPath)
    assert str(path._pathlib_home) == PLATFORMS["linux"]["home"]

def test_path_freebsd_platform():
    path = Path(platform="freebsd12")
    assert path.platform == "freebsd"
    assert isinstance(path._pathlib_home, PurePosixPath)
    assert str(path._pathlib_home) == PLATFORMS["freebsd"]["home"]

def test_path_with_seed():
    path = Path(seed=42)
    assert path.seed == 42
    assert path._has_seed() is True

def test_path_with_custom_random():
    custom_random = _random.Random()
    path = Path(random=custom_random)
    assert path.random is custom_random

def test_path_invalid_random_type():
    with pytest.raises(TypeError):
        Path(random="not_a_random_instance")


# LLM-generated content at query #2
#--------------------------

```python
def test_path_init_windows_platform():
    path_provider = Path(platform="win32")
    assert isinstance(path_provider._pathlib_home, PureWindowsPath)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_path_default_initialization():
    path = Path()
    assert path.platform == sys.platform
    assert isinstance(path._pathlib_home, (PureWindowsPath, PurePosixPath))
    assert str(path._pathlib_home) == PLATFORMS[sys.platform]["home"]

def test_path_custom_platform():
    path = Path(platform="linux")
    assert path.platform == "linux"
    assert isinstance(path._pathlib_home, PurePosixPath)
    assert str(path._pathlib_home) == PLATFORMS["linux"]["home"]

def test_path_freebsd_platform():
    path = Path(platform="freebsd12")
    assert path.platform == "freebsd"
    assert isinstance(path._pathlib_home, PurePosixPath)
    assert str(path._pathlib_home) == PLATFORMS["freebsd"]["home"]

def test_path_with_seed():
    seed = 42
    path = Path(seed=seed)
    assert path.seed == seed
    assert path._has_seed() is True

def test_path_with_custom_random():
    custom_random = _random.Random()
    path = Path(random=custom_random)
    assert path.random is custom_random

def test_path_with_invalid_random():
    with pytest.raises(TypeError):
        Path(random="not_a_random_object")


# LLM-generated content at query #2
#--------------------------

```python
def test_platform_starts_with_win():
    path = Path(platform="win32")
    assert isinstance(path._pathlib_home, PureWindowsPath)


