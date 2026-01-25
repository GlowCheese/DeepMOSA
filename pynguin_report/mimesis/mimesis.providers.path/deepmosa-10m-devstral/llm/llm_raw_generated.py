####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_path_init_default_platform():
    path = Path()
    assert path.platform == sys.platform
    assert isinstance(path._pathlib_home, (PureWindowsPath, PurePosixPath))
    assert str(path._pathlib_home) == PLATFORMS[sys.platform]["home"]

def test_path_init_custom_platform():
    path = Path(platform="linux")
    assert path.platform == "linux"
    assert isinstance(path._pathlib_home, PurePosixPath)
    assert str(path._pathlib_home) == PLATFORMS["linux"]["home"]

def test_path_init_freebsd_platform():
    path = Path(platform="freebsd12")
    assert path.platform == "freebsd"
    assert isinstance(path._pathlib_home, PurePosixPath)
    assert str(path._pathlib_home) == PLATFORMS["freebsd"]["home"]

def test_path_init_with_seed():
    path = Path(seed=42)
    assert path.seed == 42
    assert path._has_seed() is True

def test_path_init_with_custom_random():
    custom_random = _random.Random()
    path = Path(random=custom_random)
    assert path.random is custom_random

def test_path_init_invalid_random_type():
    with pytest.raises(TypeError):
        Path(random="not a random instance")


# LLM-generated content at query #2
#--------------------------

```python
def test_user_linux_platform():
    path = Path(platform="linux")
    user_path = path.user()
    assert user_path.startswith("/home/")
    assert len(user_path.split("/")) == 3
    assert user_path.split("/")[2].islower()

def test_user_windows_platform():
    path = Path(platform="win32")
    user_path = path.user()
    assert user_path.startswith("\\Users\\")
    assert len(user_path.split("\\")) == 3
    assert user_path.split("\\")[2].istitle()

def test_user_darwin_platform():
    path = Path(platform="darwin")
    user_path = path.user()
    assert user_path.startswith("/Users/")
    assert len(user_path.split("/")) == 3
    assert user_path.split("/")[2].islower()

def test_user_freebsd_platform():
    path = Path(platform="freebsd")
    user_path = path.user()
    assert user_path.startswith("/home/")
    assert len(user_path.split("/")) == 3
    assert user_path.split("/")[2].islower()


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_path_initialization_with_default_platform():
    path = Path()
    assert path.platform == sys.platform
    assert isinstance(path._pathlib_home, (PureWindowsPath, PurePosixPath))
    assert str(path._pathlib_home) == PLATFORMS[sys.platform]["home"]

def test_path_initialization_with_linux_platform():
    path = Path(platform="linux")
    assert path.platform == "linux"
    assert isinstance(path._pathlib_home, PurePosixPath)
    assert str(path._pathlib_home) == PLATFORMS["linux"]["home"]

def test_path_initialization_with_darwin_platform():
    path = Path(platform="darwin")
    assert path.platform == "darwin"
    assert isinstance(path._pathlib_home, PurePosixPath)
    assert str(path._pathlib_home) == PLATFORMS["darwin"]["home"]

def test_path_initialization_with_win32_platform():
    path = Path(platform="win32")
    assert path.platform == "win32"
    assert isinstance(path._pathlib_home, PureWindowsPath)
    assert str(path._pathlib_home) == PLATFORMS["win32"]["home"]

def test_path_initialization_with_win64_platform():
    path = Path(platform="win64")
    assert path.platform == "win64"
    assert isinstance(path._pathlib_home, PureWindowsPath)
    assert str(path._pathlib_home) == PLATFORMS["win64"]["home"]

def test_path_initialization_with_freebsd_platform():
    path = Path(platform="freebsd")
    assert path.platform == "freebsd"
    assert isinstance(path._pathlib_home, PurePosixPath)
    assert str(path._pathlib_home) == PLATFORMS["freebsd"]["home"]

def test_path_initialization_with_freebsd_variant_platform():
    path = Path(platform="freebsd12")
    assert path.platform == "freebsd"
    assert isinstance(path._pathlib_home, PurePosixPath)
    assert str(path._pathlib_home) == PLATFORMS["freebsd"]["home"]


