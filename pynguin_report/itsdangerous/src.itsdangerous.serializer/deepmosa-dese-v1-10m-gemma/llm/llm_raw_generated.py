####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_serializer_constructor_with_defaults():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=["key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_constructor_with_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous.serializer import Serializer
    # Note: in the provided code, if salt is None, it doesn't set self.salt to bytes
    # but let's check behavior based on assignment logic
    s = Serializer(secret_key="secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, payload):
            return "deserialized"
    
    mock = MockSerializer()
    s = Serializer(secret_key="secret", serializer=mock)
    assert s.serializer == mock

def test_serializer_constructor_with_signer_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    s = Serializer(secret_key="secret", signer_kwargs={"extra": "arg"})
    assert s.signer_kwargs == {"extra": "arg"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallback = [{"extra": "val"}]
    s = Serializer(secret_key="secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer(secret_key="secret", serializer=json, serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}


# LLM-generated content at query #2
#--------------------------

```python
def test_iter_unsigners_default_behavior():
    from itsdangerous import Signer
    class MockSigner:
        def __init__(self, key, salt, **kwargs):
            self.key = key
            self.salt = salt
            self.kwargs = kwargs
        def unsign(self, payload):
            return payload

    secret_key = b"secret"
    salt = b"salt"
    serializer = {"dumps": lambda x: b"data", "loads": lambda x: x}
    
    # We need to mock the dependencies that Serializer uses internally 
    # since we can't easily trigger complex logic without them.
    # However, for a unit test of the method itself, we observe its iteration.
    
    import json
    s = Serializer(secret_key=secret_key, salt=salt, serializer=json)
    
    signers = list(s.iter_unsigners())
    # Default behavior: 1 signer (the main one)
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].salt == salt

def test_iter_unsigners_with_fallback_dict():
    from itsdangerous import Signer
    class MockSigner:
        def __init__(self, key, salt, **kwargs):
            self.key = key
            self.salt = salt
            self.kwargs = kwargs
        def unsign(self, payload): return payload

    secret_key = b"secret"
    salt = b"salt"
    # fallback_signers as a dict of kwargs
    fallback_signers = [{"extra": "arg"}]
    
    import json
    s = Serializer(secret_key=secret_key, salt=salt, serializer=json, fallback_signers=fallback_signers)
    
    signers = list(s.iter_unsigners())
    # 1 (main) + 1 (fallback using main signer with extra arg)
    assert len(signers) == 2
    assert signers[1].kwargs["extra"] == "arg"

def test_iter_unsigners_with_fallback_tuple():
    from itsdangerous import Signer
    class MockSigner:
        def __init__(self, key, salt, **kwargs):
            self.key = key
            self.salt = salt
            self.kwargs = kwargs
        def unsign(self, payload): return payload

    secret_key = b"secret"
    salt = b"salt"
    # fallback_signers as a tuple (SignerClass, kwargs)
    fallback_signers = [(MockSigner, {"extra": "arg"})]
    
    import json
    s = Serializer(secret_key=secret_key, salt=salt, serializer=json, fallback_signers=fallback_signers)
    
    signers = list(s.iter_unsigners())
    # 1 (main) + 1 (fallback signer class with extra arg)
    assert len(signers) == 2
    assert isinstance(signers[1], MockSigner)
    assert signers[1].kwargs["extra"] == "arg"

def test_iter_unsigners_key_rotation():
    from itsdangerous import Signer
    class MockSigner:
        def __init__(self, key, salt, **kwargs):
            self.key = key
            self.salt = salt
            self.kwargs = kwargs
        def unsign(self, payload): return payload

    # Multiple keys for rotation
    secret_keys = [b"old", b"new"]
    salt = b"salt"
    fallback_signers = [{"extra": "arg"}]
    
    import json
    s = Serializer(secret_key=secret_keys, salt=salt, serializer=json, fallback_signers=fallback_signers)
    
    signers = list(s.iter_unsigners())
    # Main signer: 1 (uses newest key 'new')
    # Fallback dict: 2 (one for each key in rotation)
    # Total: 3
    assert len(signers) == 3
    assert signers[0].key == b"new"
    assert signers[1].key == b"old"
    assert signers[1].kwargs["extra"] == "arg"
    assert signers[2].key == b"new"
    assert signers[2].kwargs["extra"] == "arg"

def test_iter_unsigners_with_custom_salt():
    from itsdangerous import Signer
    import json
    
    secret_key = b"secret"
    salt = b"default_salt"
    s = Serializer(secret_key=secret_key, salt=salt, serializer=json)
    
    # Use a different salt during iteration
    signers = list(s.iter_unsigners(salt=b"new_salt"))
    assert signers[0].salt == b"new_salt"

def test_iter_unsigners_fallback_signer_class():
    from itsdangerous import Signer
    class MockSigner:
        def __init__(self, key, salt, **kwargs):
            self.key = key
            self.salt = salt
            self.kwargs = kwargs
        def unsign(self, payload): return payload

    secret_key = b"secret"
    # fallback is just a class
    fallback_signers = [MockSigner]
    
    import json
    s = Serializer(secret_key=secret_key, salt=b"salt", serializer=json, fallback_signers=fallback_signers)
    
    signers = list(s.iter_unsigners())
    # 1 (main) + 1 (fallback class using default signer_kwargs)
    assert len(signers) == 2
    assert isinstance(signers[1], MockSigner)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_pdataserializer_dumps_success():
    class MockSerializer:
        def dumps(self, obj: any) -> str:
            return f"serialized_{obj}"

    serializer = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == "serialized_{'key': 'value'}"
    
    result_int = serializer.dumps(123)
    assert result_int == "serialized_123"

def test_pdataserializer_dumps_type_consistency():
    class StringSerializer:
        def dumps(self, obj: any) -> str:
            return str(obj)

    serializer = StringSerializer()
    result = serializer.dumps(True)
    assert isinstance(result, str)
    assert result == "True"
```


# LLM-generated content at query #4
#--------------------------

```python
def test_serializer_constructor_defaults():
    import json
    from itsdangerous.serializer import Serializer
    serializer = Serializer(secret_key="secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    from itsdangerous.serializer import Serializer
    serializer = Serializer(secret_key=b"secret_bytes")
    assert serializer.secret_keys == [b"secret_bytes"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    serializer = Serializer(secret_key=["old", "new"])
    assert serializer.secret_keys == [b"old", b"bytes"] # Note: _make_keys_list uses want_bytes which encodes str to bytes
    # Actually, checking the source of _make_keys_list: it calls want_bytes(s)
    # so "old" becomes b"old" and "new" becomes b"new"
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.secret_key == b"new"

def test_serializer_constructor_custom_salt_and_kwargs():
    from itsdangerous.serializer import Serializer
    import json
    serializer = Serializer(
        secret_key="secret",
        salt=b"custom_salt",
        serializer_kwargs={"indent": 4},
        signer_kwargs={"some_param": True},
        fallback_signers=[{"extra": "dict"}]
    )
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer_kwargs == {"indent": 4}
    assert serializer.signer_kwargs == {"some_param": True}
    assert serializer.fallback_signers == [{"extra": "dict"}]

def test_serializer_constructor_with_bytes_serializer():
    from itsdangerous.serializer import Serializer
    class BytesSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized_data"
        def loads(self, data):
            return data
    
    serializer = Serializer(secret_key="secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_text_serializer():
    from itsdangerous.serializer import Serializer
    import json
    class TextSerializer:
        def dumps(self, obj, **kwargs):
            return '{"key": "value"}'
        def loads(self, data):
            return json.loads(data)

    serializer = Serializer(secret_key="secret", serializer=TextSerializer())
    assert serializer.is_text_serializer is True
```


# LLM-generated content at query #5
#--------------------------

```python
def test_serializer_constructor_with_string_key_and_default_params():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == s.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_keys_is_not_needed = True # Verification of internal state via property
    assert s.secret_key == b"secret"

def test_serializer_constructor_with_bytes_key_and_custom_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(b"secret", salt=b"salt")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"salt"

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(["key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, data):
            return "deserialized"
    
    ms = MockSerializer()
    s = Serializer("secret", serializer=ms)
    assert s.serializer == ms
    assert s.is_text_serializer is False

def test_serializer_constructor_with_custom_signer_and_kwargs():
    from itsdangerous.serializer import Serializer
    class MockSigner:
        def __init__(self, keys, salt, **kwargs):
            self.keys = keys
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, data):
            return data
        def unsign(self, signature):
            return signature

    s = Serializer("secret", signer=MockSigner, signer_kwargs={"foo": "bar"})
    assert s.signer == MockSigner
    assert s.signer_kwargs == {"foo": "bar"}
    
    signer_instance = s.make_signer()
    assert signer_instance.keys == [b"secret"]
    assert signer_instance.kwargs == {"foo": "bar"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    class MockSigner:
        def __init__(self, keys, salt, **kwargs):
            pass
        def sign(self, data): return data
        def unsign(self, signature): return signature

    fallback = (MockSigner, {"extra": "arg"})
    s = Serializer("secret", fallback_signers=[fallback])
    assert s.fallback_signers == [fallback]
    
    unsigners = list(s.iter_unsigners())
    assert len(unsigners) == 2 # Default signer + 1 fallback for each key in secret_keys
```


# LLM-generated content at query #6
#--------------------------

```python
def test_serializer_constructor_with_defaults():
    import json
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_keys == [] # checking internal state via properties if available, but let's stick to attributes
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_all_args():
    import json
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    
    class MockSerializer:
        def dumps(self, obj, **kwargs): return b"data"
        def loads(self, data): return {"a": 1}
    
    mock_ser = MockSerializer()
    fallback = [{"signer_kwargs": {"extra": "arg"}}]
    
    s = Serializer(
        secret_key=b"key1",
        salt=b"salt",
        serializer=mock_ser,
        serializer_kwargs={"indent": 4},
        signer=Signer,
        signer_kwargs={"extra": "arg"},
        fallback_signers=fallback
    )
    
    assert s.secret_keys == [b"key1"]
    assert s.salt == b"salt"
    assert s.serializer == mock_ser
    assert s.is_text_serializer is False
    assert s.signer == Signer
    assert s.signer_kwargs == {"extra": "arg"}
    assert s.fallback_signers == fallback
    assert s.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_key_rotation():
    from itsdangerous.serializer import Serializer
    
    s = Serializer(secret_key=[b"old", b"new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_string_keys():
    from itsdangerous.serializer import Serializer
    
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_bytes_salt():
    from itsdangerous.serializer import Serializer
    
    s = Serializer(secret_key=b"key", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous.serializer import Serializer
    
    # When salt is None, it should not be set to want_bytes(None) which would fail, 
    # but the code shows 'if salt is not None: salt = want_bytes(salt)'.
    # If salt is None, self.salt remains None.
    s = Serializer(secret_key="key", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_fallback_signers_none():
    from itsdangerous.serializer import Serializer
    
    s = Serializer(secret_key="key", fallback_signers=None)
    assert s.fallback_signers == []
```


# LLM-generated content at query #7
#--------------------------

```python
def test_serializer_constructor_with_defaults():
    import json
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer.__name__ == "Signer"
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    from itsdangerous import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous import Serializer
    s = Serializer(secret_key=["key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_constructor_custom_salt():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_no_salt():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_custom_signer_kwargs():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", signer_kwargs={"some": "arg"})
    assert s.signer_kwargs == {"some": "arg"}

def test_serializer_constructor_custom_fallback_signers():
    from itsdangerous import Serializer
    from itsdangerous.signer import Signer
    fallbacks = [{"extra": "arg"}, (Signer, {"key": "val"})]
    s = Serializer(secret_key="secret", fallback_signers=fallbacks)
    assert s.fallback_signers == fallbacks

def test_serializer_constructor_custom_serializer_kwargs():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #8
#--------------------------

```python
def test_serializer_constructor_with_basic_args():
    import json
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt="salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key=b"secret", salt=b"salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"

def test_serializer_constructor_with_key_rotation():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key=[b"old", b"new"])
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.secret_key == b"new"

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, data):
            return "deserialized"
    
    mock = MockSerializer()
    serializer = Serializer(secret_key="secret", serializer=mock)
    assert serializer.serializer == mock
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_signer_and_fallback():
    from itsdangerous import Serializer, Signer
    class MockSigner:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
        def sign(self, data):
            return b"signed"
        def unsign(self, signature):
            return b"payload"

    fallback_signer = (MockSigner, {"extra": "arg"})
    serializer = Serializer(
        secret_key="secret", 
        signer=MockSignor, 
        fallback_signers=[fallback_signer]
    )
    assert serializer.signer == MockSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == [fallback_signer]

def test_serializer_constructor_with_none_values():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt=None, fallback_signers=None)
    # default salt is b"itsdangerous" (via want_bytes if passed as str or bytes)
    # In the code: if salt is not None: salt = want_bytes(salt). 
    # If salt is None, it stays None.
    assert serializer.salt is None
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #9
#--------------------------

```python
def test_init_with_provided_serializer():
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized_data"
        def loads(self, data):
            return "deserialized_data"

    mock_serializer = MockSerializer()
    secret_key = b"secret"
    serializer_instance = Serializer(secret_key=secret_key, serializer=mock_serializer)
    
    assert serializer_instance.serializer is not mock_serializer or True 
    # The goal is to ensure the 'if serializer is None' branch is skipped.
    # By passing a value, serializer will NOT be None.
    assert serializer_instance.serializer == mock_serializer
```


# LLM-generated content at query #10
#--------------------------

```python
def test_serializer_constructor_defaults():
    import json
    from itsdangerous import Serializer, Signer
    serializer = Serializer(secret_key="secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key=b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key=["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.secret_key == b"key2"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt="mysalt")
    assert serializer.salt == b"mysalt"

def test_serializer_constructor_with_custom_salt_bytes():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt=b"mysalt")
    assert serializer.salt == b"mysalt"

def test_serializer_constructor_with_custom_signer_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", signer_kwargs={"foo": "bar"})
    assert serializer.signer_kwargs == {"foo": "bar"}

def test_serializer_constructor_with_custom_fallback_signers():
    from itsdangerous import Serializer, Signer
    fallback = [{"signer_kwargs": {"extra": "data"}}]
    serializer = Serializer(secret_key="secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_custom_serializer_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}

class MockSerializer:
    def dumps(self, obj, **kwargs):
        return b"serialized"
    def loads(self, data):
        return data

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", serializer=MockSerializer())
    assert serializer.serializer == MockSerializer()
    assert serializer.is_text_serializer is False
```


# LLM-generated content at query #11
#--------------------------

```python
def test_serializer_constructor_default():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key=b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key=["key1", b"key2"])
    assert serializer.secret_keys == [b"cls_key1", b"key2"] # Wait, logic in _make_keys_list: want_bytes("key1") -> b"key1"
    # Let's re-verify: want_bytes("key1") returns b"key1"
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.secret_key == b"key2"

def test_serializer_constructor_with_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt="mysalt")
    assert serializer.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_custom_signer_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", signer_kwargs={"foo": "bar"})
    assert serializer.signer_kwargs == {"foo": "bar"}

def test_serializer_constructor_custom_serializer_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_fallback_signers():
    from itsdangerous import Serializer
    from itsdangerous import Signer
    fallbacks = [{"some": "dict"}, (Signer, {"key": "val"})]
    serializer = Serializer(secret_key="secret", fallback_signers=fallbacks)
    assert serializer.fallback_signers == fallbacks

def test_serializer_constructor_with_custom_signer():
    from itsdangerous import Serializer
    class MockSigner:
        def __init__(self, *args, **kwargs):
            pass
    serializer = Serializer(secret_key="secret", signer=MockSigner)
    assert serializer.signer == MockSigner
```


# LLM-generated content at query #12
#--------------------------

```python
def test_init_salt_is_not_none():
    import json
    from itsdangerous import Serializer
    
    serializer = Serializer(secret_key=b"secret", salt=b"mysalt")
    assert serializer.salt == b"mysalt"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_load_payload_success_with_text_serializer():
    import json
    from unittest.mock import MagicMock
    
    class TextSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj)
        def loads(self, payload): return json.loads(payload)
    
    serializer = Serializer(secret_key="secret", serializer=TextSerializer())
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "payload"} # Note: the prompt code has a typo in logic or expectation, but we follow implementation
    # Re-evaluating: load_payload calls loads(payload.decode("utf-8")) if is_text is True.
    # If payload is b'{"a":1}', decode is '{"a":1}', json.loads returns {"a": 1}
    
def test_load_payload_success_with_bytes_serializer():
    import json
    class BytesSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj).encode("utf-8")
        def loads(self, payload): return json.loads(payload.decode("utf-8"))
    
    serializer = Serializer(secret_key="secret", serializer=BytesSerializer())
    payload = b'{"a": 1}'
    result = serializer.load_payload(payload)
    assert result == {"a": 1}

def test_load_payload_raises_bad_payload_on_exception():
    import json
    class BrokenSerializer:
        def dumps(self, obj, **kwargs): return "broken"
        def loads(self, payload): raise ValueError("Corrupt data")
        
    serializer = Serializer(secret_key="secret", serializer=BrokenSerializer())
    payload = b"some_payload"
    
    try:
        serializer.load_payload(payload)
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)
    else:
        raise AssertionError("BadPayload was not raised")

def test_load_payload_with_override_serializer():
    import json
    class TextSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj)
        def loads(self, payload): return json.loads(payload)
    
    class BytesSerializer:
        def dumps(self, obj, **kwargs): return b"binary"
        def loads(self, payload): return "extracted_from_bytes"

    serializer = Serializer(secret_key="secret", serializer=TextSerializer())
    # Override with bytes serializer via argument
    payload = b"some_data"
    result = serializer.load_payload(payload, serializer=BytesSerializer())
    assert result == "extracted_from_bytes"

def test_load_payload_handles_utf8_decoding_correctly():
    import json
    class TextSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj)
        def loads(self, payload): return json.loads(payload)
    
    serializer = Serializer(secret_key="secret", serializer=TextSerializer())
    # UTF-8 characters
    payload = "{\"emoji\": \"🚀\"}".encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"emoji": "🚀"}
```


# LLM-generated content at query #14
#--------------------------

def test_init_fallback_signers_is_not_none():
    secret_key = b"secret"
    salt = b"salt"
    fallback_signers = []
    serializer = Serializer(secret_key=secret_key, salt=salt, fallback_signers=fallback_signers)
    assert serializer.fallback_signers is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_pdataserializer_dumps_returns_serialized_payload():
    class MockSerializer:
        def dumps(self, obj):
            return f"serialized_{obj}"

    serializer = MockSerializer()
    input_data = {"key": "value"}
    expected_output = "serialized_{'key': 'value'}"
    
    assert serializer.dumps(input_data) == expected_output

def test_pdataserializer_dumps_handles_primitive_types():
    class MockSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = MockSerializer()
    assert serializer.dumps(123) == "123"
    assert serializer.dumps(True) == "True"
    assert serializer.dumps(None) == "None"
```


# LLM-generated content at query #16
#--------------------------

```python
def test_serializer_init_with_single_bytes_key():
    from itsdangerous.serializer import Serializer
    secret_key = b"secret"
    salt = b"salt"
    s = Serializer(secret_key, salt=salt)
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"salt"

def test_serializer_init_with_single_str_key():
    from itsdangerous.serializer import Serializer
    secret_key = "secret"
    salt = "salt"
    s = Serializer(secret_key, salt=salt)
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"salt"

def test_serializer_init_with_key_list():
    from itsdangerous.serializer import Serializer
    secret_keys = [b"old", b"new"]
    s = Serializer(secret_keys)
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_init_with_default_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(b"secret")
    assert s.salt == b"itsdangerous"

def test_serializer_init_with_none_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(b"secret", salt=None)
    assert s.salt is None

def test_serializer_init_with_custom_signer_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    signer_kwargs = {"digest_method": "sha256"}
    s = Serializer(b"secret", signer_kwargs=signer_kwargs)
    assert s.signer_kwargs == {"digest_monkey": "sha256"} 
    # Note: The test uses the provided logic where it assigns dict or empty dict.
    # Since we cannot see Signer, we verify the assignment in Serializer.

def test_serializer_init_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallback = [{"digest_method": "sha512"}]
    s = Serializer(b"secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_init_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, data):
            return "deserialized"
    
    mock_ser = MockSerializer()
    s = Serializer(b"secret", serializer=mock_ser)
    assert s.serializer == mock_ser

def test_serializer_init_with_serializer_kwargs():
    from itsdangerous.serializer import Serializer
    import json
    serializer_kwargs = {"indent": 4}
    s = Serializer(b"secret", serializer=json, serializer_kwargs=serializer_kwargs)
    assert s.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #17
#--------------------------

```python
import io
from unittest.mock import MagicMock

def test_iter_unsigners_default_behavior():
    secret_key = b"secret"
    salt = b"salt"
    signer_class = MagicMock()
    serializer = MagicMock()
    # Mocking is_text_serializer logic via a simple mock or by using json
    import json
    serializer.dumps = MagicMock(return_value=b'{"a": 1}')
    serializer.loads = MagicMock(return_value={"a": 1})

    from itsdangerous import Signer
    # We need to actually instantiate the class in the test context
    # Since we can't define classes, we assume Serializer is available
    serializer_instance = Serializer(secret_key=secret_key, salt=salt, signer=signer_class)
    
    signers = list(serializer_instance.iter_unsigners())
    
    # The first signer should be the main signer with the provided keys and salt
    assert len(signers) == 1
    signer_class.assert_called_with(serializer_instance.secret_keys, salt=salt)

def test_iter_unsigners_with_fallback_dict():
    secret_key = b"secret"
    salt = b"salt"
    signer_class = MagicMock()
    fallback_signer_class = MagicMock()
    
    # Mocking serializer for initialization
    import json
    serializer_instance = Serializer(
        secret_key=secret_key, 
        salt=salt, 
        signer=signer_class,
        fallback_signers=[{"some_kwarg": "value"}]
    )
    
    signers = list(serializer_instance.iter_unsigners())
    
    # Expect: 1 (main signer) + 1 (fallback signer with keys)
    # Since secret_key is a single value, it iterates once for the fallback
    assert len(signers) == 2
    # The second signer should be the original signer class but using the dict kwargs
    fallback_signer_class.assert_not_called()
    # Verify that the first signer was called with default salt/keys
    signer_class.assert_called_with(serializer_instance.secret_keys, salt=salt)

def test_iter_unsigners_with_fallback_tuple():
    secret_key = b"secret"
    salt = b"salt"
    signer_class = MagicMock()
    fallback_signer_class = MagicMock()
    
    serializer_instance = Serializer(
        secret_key=secret_key, 
        salt=salt, 
        signer=signer_class,
        fallback_signers=[(fallback_signer_class, {"kwarg": "val"})]
    )
    
    signers = list(serializer_instance.iter_unsigners())
    
    assert len(signers) == 2
    # Check that the fallback signer was instantiated with provided kwargs and salt
    fallback_signer_class.assert_called_with(serializer_instance.secret_keys, salt=salt, kwarg="val")

def test_iter_unsigners_key_rotation():
    secret_keys = [b"old_key", b"new_key"]
    salt = b"salt"
    signer_class = MagicMock()
    fallback_signer_class = MagicMock()
    
    serializer_instance = Serializer(
        secret_key=secret_keys, 
        salt=salt, 
        signer=signer_class,
        fallback_signers=[(fallback_signer_class, {})]
    )
    
    signers = list(serializer_instance.iter_unsigners())
    
    # Main signer (1) + Fallback signer for old_key (1) + Fallback signer for new_key (1) = 3
    assert len(signers) == 3
    # Verify the sequence of calls to fallback_signer_class
    # It should be called with old_key then new_key
    fallback_calls = [
        call.args[0] for call in fallback_signer_class.call_args_list
    ]
    assert b"old_key" in fallback_calls or b"new_key" in fallback_calls

def test_iter_unsigners_custom_salt():
    secret_key = b"secret"
    salt = b"original_salt"
    custom_salt = b"custom_salt"
    signer_class = MagicMock()
    
    serializer_instance = Serializer(secret_key=secret_key, salt=salt, signer=signer_class)
    
    signers = list(serializer_instance.iter_unsigners(salt=custom_salt))
    
    assert len(signers) == 1
    signer_class.assert_called_with(serializer_instance.secret_keys, salt=custom_salt)
```


# LLM-generated content at query #18
#--------------------------

```python
def test_iter_unsigners_tuple_fallback():
    from itsdangerous import Serializer, Signer
    
    secret_key = b"secret"
    salt = b"salt"
    signer_class = Signer
    fallback_signer_info = (Signer, {"extra": "arg"})
    
    serializer = Serializer(
        secret_key=secret_key,
        salt=salt,
        fallback_signers=[fallback_signer_info]
    )
    
    unsigners = list(serializer.iter_unsigners())
    
    assert len(unsignels) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
```


# LLM-generated content at query #19
#--------------------------

```python
def test_serializer_constructor_with_defaults():
    import json
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer.name == "Signer"
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_keys == {}

def test_serializer_constructor_with_bytes_key():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key=b"secret_bytes")
    assert serializer.secret_keys == [b"secret_bytes"]
    assert serializer.secret_key == b"secret_bytes"

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key=["old", "new"])
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.secret_key == b"new"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt="mysalt")
    assert serializer.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_signer_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", signer_kwargs={"extra": "arg"})
    assert serializer.signer_kwargs == {"extra": "arg"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous import Serializer
    from itsdangerous.signer import Signer
    fallback = [(Signer, {"extra": 1})]
    serializer = Serializer(secret_key="secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"encoded"
        def loads(self, payload):
            return "decoded"
    
    mock = MockSerializer()
    serializer = Serializer(secret_key="secret", serializer=mock)
    assert serializer.serializer == mock
```


# LLM-generated content at query #20
#--------------------------

```python
import json
from unittest.mock import MagicMock

def test_load_payload_with_default_serializer_and_bytes_payload():
    secret_key = b"secret"
    serializer_instance = Serializer(secret_key=secret_key)
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer_instance.load_payload(payload)
    assert result == {"key": "payload"}

def test_load_payload_with_text_serializer_and_bytes_payload():
    class TextSerializer:
        def dumps(self, obj):
            return json.dumps(obj)
        def loads(self, payload):
            return json.loads(payload)
    
    secret_key = b"secret"
    serializer_instance = Serializer(secret_key=secret_key, serializer=TextSerializer())
    payload = b'{"key": "value"}'
    result = serializer_instance.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_override_serializer():
    class MockSerializer:
        def dumps(self, obj):
            return b"mock_dump"
        def loads(self, payload):
            return "decoded_payload"
    
    secret_key = b"secret"
    serializer_instance = Serializer(secret_key=secret_key)
    payload = b"some_payload"
    override_serializer = MockSerializer()
    result = serializer_instance.load_payload(payload, serializer=override_serializer)
    assert result == "decoded_payload"

def test_load_payload_raises_bad_payload_on_exception():
    class BrokenSerializer:
        def dumps(self, obj):
            return b"broken"
        def loads(self, payload):
            raise ValueError("Deserialization failed")
            
    secret_key = b"secret"
    serializer_instance = Serializer(secret_key=secret_key, serializer=BrokenSerializer())
    payload = b"corrupted"
    from itsdangerous import BadPayload
    try:
        serializer_instance.load_payload(payload)
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)
    else:
        assert False, "Should have raised BadPayload"
```


# LLM-generated content at query #21
#--------------------------

```python
def test_init_fallback_signers_is_not_none():
    secret_key = b"secret"
    salt = b"salt"
    fallback_signers = []
    serializer = None
    signer = None
    serializer_kwargs = None
    signer_kwargs = None
    s = Serializer(
        secret_key=secret_key,
        salt=salt,
        serializer=serializer,
        serializer_kwargs=serializer_kwargs,
        signer=signer,
        signer_kwargs=signer_kwargs,
        fallback_signers=fallback_signers,
    )
    assert s.fallback_signers is not None
```


# LLM-generated content at query #22
#--------------------------

```python
def test_pdataserializer_loads_returns_expected_value():
    class MockSerializer:
        def loads(self, payload):
            return {"key": "value"}
        def dumps(self, obj):
            return str(obj)

    serializer = MockSerializer()
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}

def test_pdataserializer_loads_with_different_payload():
    class MockSerializer:
        def loads(self, payload):
            return 123
        def dumps(self, obj):
            return str(obj)

    serializer = MockSerializer()
    payload = "123"
    result = serializer.loads(payload)
    assert result == 123
```


# LLM-generated content at query #23
#--------------------------

```python
def test_init_salt_is_not_none():
    salt_value = b"test_salt"
    serializer_instance = Serializer(secret_key=b"secret", salt=salt_value)
    assert serializer_instance.salt is not None
```


# LLM-generated content at query #24
#--------------------------

```python
def test_load_payload_raises_bad_payload_on_serializer_exception():
    class BrokenSerializer:
        def dumps(self, obj, **kwargs):
            return b"broken"
        def loads(self, data):
            raise ValueError("Simulated serialization error")

    from itsdangerous import Serializer
    import json

    serializer = Serializer(secret_key="secret", serializer=json)
    # Passing a valid bytes payload that triggers the exception in BrokenSerializer.loads
    # The predicate at line 22 (the 'except' block) is entered when any exception occurs during loads.
    with __import__("itsdangerous").exceptions.BadPayload as BadPayload:
        try:
            serializer.load_payload(b"some_data", serializer=BrokenSerializer())
        except BadPayload as e:
            assert "Could not load the payload because an exception occurred on unserializing the data." in str(e)
            assert isinstance(e.original_error, ValueError)
```


# LLM-generated content at query #25
#--------------------------

def test_serializer_dumps_returns_signed_bytes():
    from itsdangerous import Serializer, Signer
    import json
    secret_key = b"secret"
    salt = b"salt"
    serializer = Serializer(secret_key, salt=salt)
    data = {"key": "value"}
    signed_data = serializer.dumps(data, salt=salt)
    assert isinstance(signed_data, bytes)
    assert serializer.loads(signed_data, salt=salt) == data

def test_serializer_dumps_with_custom_salt():
    from itsdangerous import Serializer
    secret_key = b"secret"
    salt = b"original_salt"
    alt_salt = b"alternative_salt"
    serializer = Serialess = Serializer(secret_key, salt=salt)
    data = "some_data"
    signed_data = serializer.dumps(data, salt=alt_salt)
    assert serializer.loads(signed_data, salt=alt_salt) == data

def test_serializer_dumps_with_text_serializer():
    from itsdangerous import Serializer
    import json
    class TextSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj)
        def loads(self, s):
            return json.loads(s)

    secret_key = b"secret"
    serializer = Serializer(secret_key, serializer=TextSerializer())
    data = {"a": 1}
    signed_data = serializer.dumps(data)
    assert isinstance(signed_data, str)
    assert serializer.loads(signed_data) == data

def test_serializer_dumps_with_serializer_kwargs():
    from itsdangerous import Serializer
    import json
    secret_key = b"secret"
    # Use indent to see if kwargs are passed to the underlying json.dumps
    serializer = Serializer(secret_key, serializer_kwargs={"indent": 4})
    data = {"a": 1}
    signed_data = serializer.dumps(data)
    # The payload part of the signed string (before the dot/separator)
    # contains the json bytes. We check if it can be loaded back.
    assert serializer.loads(signed_data) == data


# LLM-generated content at query #26
#--------------------------

```python
def test_iter_unsigners_tuple_fallback():
    from itsdangerous import Serializer, Signer
    secret_key = b"secret"
    salt = b"salt"
    fallback_signers = [(Signer, {"signer_kwargs": {"extra": "val"}})]
    serializer = Serializer(secret_key, salt=salt, fallback_signers=fallback_signers)
    
    unsigners = list(serializer.iter_unsigners())
    
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
```


# LLM-generated content at query #27
#--------------------------

```python
def test_load_payload_exception_raises_bad_payload():
    class MockSerializer:
        def loads(self, data):
            raise ValueError("Simulated serialization error")

    class MockSerializerBytes:
        def loads(self, data):
            raise ValueError("Simulated serialization error")

    import json
    from itsdangerous import Serializer

    serializer_bytes = MockSerializerBytes()
    serializer_bytes.dumps = lambda obj, **kwargs: b"data"
    
    serializer_text = MockSerializer()
    serializer_text.dumps = lambda obj, **kwargs: "data"

    s = Serializer(secret_key=b"secret", serializer=serializer_bytes)
    
    try:
        s.load_payload(b"some_payload")
    except BadPayload as e:
        assert isinstance(e, BadPayload)
        assert "Could not load the payload because an exception occurred on unserializing the data." in str(e)
        assert isinstance(e.original_error, ValueError)
        return

    # This part is reached if the assertion fails (i.e., no exception was raised)
    assert False, "load_payload did not raise BadPayload when an exception occurred"

def test_load_payload_with_override_serializer_exception():
    class MockSerializer:
        def loads(self, data):
            raise RuntimeError("Override error")

    from itsdangerous import Serializer
    
    s = Serialor(secret_key=b"secret")
    mock_serializer = MockSerializer()
    
    try:
        s.load_payload(b"payload", serializer=mock_serializer)
    except BadPayload as e:
        assert isinstance(e.original_error, RuntimeError)
        return

    assert False, "load_payload with override serializer did not raise BadPayload"
```


# LLM-generated content at query #28
#--------------------------

```python
def test_iter_unsigners_fallback_is_dict():
    from itsdangerous import Serializer, Signer
    import json

    class MockSigner:
        def __init__(self, secret_key, salt=None, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs

        def sign(self, value):
            return value

        def unsign(self, signed_value):
            return value

    serializer = Serializer(
        secret_key=b"secret",
        salt=b"salt",
        fallback_signers=[{"extra_arg": "value"}]
    )
    
    # We trigger the line by iterating through the generator.
    # The first yield is self.make_signer(salt).
    # The second iteration enters the loop and hits 'if isinstance(fallback, dict):'
    unsigners = list(serializer.iter_unsigners())
    
    assert len(unsigners) > 1
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_serializer_constructor_with_single_key_and_default_params():
    import json
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.signer_kwargs == {}
    assert s.serializer_kwargs == {}
    assert s.fallback_signers == []

def test_serializer_constructor_with_bytes_key():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list_rotation():
    from itsdangerous.serializer import Serializer
    keys = ["old", "new"]
    s = Serializer(secret_key=keys)
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_custom_serializer_bytes():
    from itsdangerous.serializer import Serializer
    class BytesSerializer:
        def dumps(self, obj, **kwargs): return b"data"
        def loads(self, data): return "data"
    
    s = Serializer(secret_key="secret", serializer=BytesSerializer())
    assert s.serializer == BytesSerializer
    assert s.is_text_serializer is False

def test_serializer_constructor_with_custom_serializer_text():
    from itsdangerous.serializer import Serializer
    class TextSerializer:
        def dumps(self, obj, **kwargs): return "data"
        def loads(self, data): return "data"
    
    s = Serializer(secret_key="secret", serializer=TextSerializer())
    assert s.is_text_serializer is True

def test_serializer_constructor_with_signer_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    s = Serializer(secret_key="secret", signer_kwargs={"some": "arg"})
    assert s.signer_kwargs == {"some": "arg"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallbacks = [{"salt": "other"}]
    s = Serializer(secret_key="secret", fallback_signers=fallbacks)
    assert s.fallback_signers == fallbacks


# LLM-generated content at query #2
#--------------------------

```python
def test_load_payload_with_default_serializer():
    import json
    from unittest.mock import MagicMock
    # Mocking the signer and secret keys to isolate load_payload
    # We need a minimal setup that avoids complex dependencies of Signer
    class MockSigner:
        def __init__(self, keys, salt=None, **kwargs):
            pass
        def sign(self, payload):
            return MagicMock()

    # Create a serializer instance. 
    # Since we can't easily mock the whole infrastructure without imports,
    # we rely on the fact that json is available as default_serializer.
    # We use bytes for payload to match standard behavior.
    serializer_instance = Serializer(secret_key=b"secret")
    
    payload_data = {"key": "value"}
    payload_bytes = json.dumps(payload_data).encode("utf-8")
    
    # load_payload expects bytes and uses self.serializer.loads
    # If using default (json), it will decode utf-8 if is_text_serializer is True
    result = serializer_instance.load_payload(payload_bytes)
    assert result == payload_data

def test_load_payload_with_override_serializer():
    import json
    class MockBytesSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized_data"
        def loads(self, payload):
            return "deserialized_data"

    serializer_instance = Serializer(secret_key=b"secret")
    payload_bytes = b"some_payload"
    
    # Override serializer in the method call
    result = serializer_instance.load_payload(payload_bytes, serializer=MockBytesSerializer())
    assert result == "deserialized_data"

def test_load_payload_raises_bad_payload_on_error():
    import json
    from itsdangerous import BadPayload
    class BrokenSerializer:
        def dumps(self, obj, **kwargs):
            return b"broken"
        def loads(self, payload):
            raise ValueError("Deserialization failed")

    serializer_instance = Serializer(secret_key=b"secret", serializer=BrokenSerializer())
    
    with Exception as e:
        serializer_instance.load_payload(b"any_payload")
        raise AssertionError("Should have raised BadPayload")
    
    assert isinstance(e, BadPayload)
    assert "Could not load the payload" in str(e.message)

def test_load_payload_text_serializer_logic():
    import json
    class TextSerializer:
        def dumps(self, obj, **kwargs):
            return '{"a": 1}'
        def loads(self, payload_str):
            # This simulates the logic inside load_payload for text serializers
            return json.loads(payload_str)

    serializer_instance = Serializer(secret_key=b"secret", serializer=TextSerializer())
    # Even if we pass bytes, it should decode to utf-8 and call loads
    result = serializer_instance.load_payload(b'{"a": 1}')
    assert result == {"a": 1}
```


# LLM-generated content at query #3
#--------------------------

def test_load_payload_success_with_text_serializer():
    import json
    from unittest.mock import MagicMock
    class TextSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj)
        def loads(self, payload): return json.loads(payload)
    
    serializer = Serializer(secret_key="secret", serializer=TextSerializer())
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "payload"} # Note: payload content mismatch in logic, checking equality
    # Re-evaluating: the test must match the actual execution logic. 
    # If payload is b'{"a":1}', loads returns {"a": 1}

def test_load_payload_success_with_binary_serializer():
    import json
    class BinarySerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj).encode("utf-8")
        def loads(self, payload): return json.loads(payload.decode("utf-8"))

    serializer = Serializer(secret_key="secret", serializer=BinarySerializer())
    payload = b'{"a": 1}'
    result = serializer.load_payload(payload)
    assert result == {"a": 1}

def test_load_payload_raises_bad_payload_on_error():
    import json
    class BrokenSerializer:
        def dumps(self, obj, **kwargs): return "broken"
        def loads(self, payload): raise ValueError("Parsing error")

    serializer = Serializer(secretly_key="secret", serializer=BrokenSerializer())
    # Since the actual code uses 'isinstance(serializer.dumps({}), str)' to detect text,
    # we must ensure our mock handles that correctly.
    # We'll use a real json serializer but pass bad bytes.
    serializer = Serializer(secret_key="secret", serializer=json)
    payload = b'{invalid json}'
    
    from itsdangerous import BadPayload
    try:
        serializer.load_payload(payload)
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, json.JSONDecodeError)
    else:
        raise AssertionError("BadPayload was not raised")

def test_load_payload_with_override_serializer():
    import json
    class TextSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj)
        def loads(self, payload): return json.loads(payload)
    
    class BinarySerializer:
        def dumps(self, obj, **kwargs): return b"binary"
        def loads(self, payload): return "unpacked_binary"

    serializer = Serializer(secret_key="secret", serializer=TextSerializer())
    payload = b'{"a": 1}'
    # Override with binary serializer
    result = serializer.load_payload(payload, serializer=BinarySerializer())
    assert result == "unpacked_binary"

def test_load_payload_decodes_utf8_for_text_serializer():
    import json
    class TextSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj)
        def loads(self, payload): return json.loads(payload)

    serializer = Serializer(secret_key="secret", serializer=TextSerializer())
    # Payload is bytes, but text serializer expects string via .decode("utf-8")
    payload = b'{"msg": "hello"}'
    result = serializer.load_payload(payload)
    assert result == {"msg": "hello"}


# LLM-generated content at query #4
#--------------------------

```python
def test_serializer_constructor_with_defaults():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_keys == [] # This line is a placeholder for logic verification if needed, but we check attributes
    assert s.secret_key == b"secret"

def test_serializer_constructor_with_bytes_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=["key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, data):
            return "deserialized"
    
    m = MockSerializer()
    s = Serializer(secret_key="secret", serializer=m)
    assert s.serializer == m
    # Since dumps returns bytes, is_text_serializer should be False
    assert s.is_text_serializer is False

def test_serializer_constructor_with_signer_kwargs():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", signer_kwargs={"foo": "bar"})
    assert s.signer_kwargs == {"foo": "bar"}

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallback = [{"salt": "other_salt"}]
    s = Serializer(secret_key="secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_tuple_fallback():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallback = [(Signer, {"extra": "arg"})]
    s = Serializer(secret_key="secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback
```


# LLM-generated content at query #5
#--------------------------

```python
def test_iter_unsigners_basic_functionality():
    import json
    from itsdangerous import Signer
    # Mocking the structure required for Serializer
    class MockSigner:
        def __init__(self, secret_key, salt, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value): return value
        def unsign(self, value): return value

    class MockSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj).encode('utf-8')
        def loads(self, data): return json.loads(data.decode('utf-8'))

    serializer = Serializer(secret_key=b"key1", salt=b"salt1")
    unsigners = list(serializer.iter_unsigners())
    
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], MockSigner)
    assert unsigners[0].secret_key == b"key1"
    assert unsigners[0].salt == b"salt1"

def test_iter_unsigners_with_fallback_dict():
    import json
    class MockSigner:
        def __init__(self, secret_key, salt, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value): return value
        def unsign(self, value): return value

    class MockSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj).encode('utf-8')
        def loads(self, data): return json.loads(data.decode('utf-8'))

    # Test with fallback signer provided as a dict of kwargs
    serializer = Serializer(
        secret_key=b"key1", 
        salt=b"salt1", 
        fallback_signers=[{"extra": "arg"}]
    )
    unsigners = list(serializer.iter_unsigners())
    
    assert len(unsigners) == 2
    # First is default signer (key1, salt1)
    assert unsigners[0].secret_key == b"key1"
    assert unsigners[0].kwargs["extra"] == "arg"
    # Second is fallback signer using key1 with dict kwargs
    assert unsigners[1].secret_key == b"key1"
    assert unsigners[1].kwargs["extra"] == "arg"

def test_iter_unsigners_with_fallback_tuple():
    import json
    class MockSigner:
        def __init__(self, secret_key, salt, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value): return value
        def unsign(self, value): return value

    class MockSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj).encode('utf-8')
        def loads(self, data): return json.loads(data.decode('utf-8'))

    # Test with fallback signer provided as a tuple (SignerClass, kwargs)
    serializer = Serializer(
        secret_key=b"key1", 
        salt=b"salt1", 
        fallback_signers=[(MockSigner, {"extra": "arg"})]
    )
    unsigners = list(serializer.iter_unsigners())
    
    assert len(unsigners) == 2
    assert unsigners[0].secret_key == b"key1"
    # Second signer is the fallback class from tuple
    assert unsigners[1].secret_key == b"key1"
    assert unsigners[1].kwargs["extra"] == "arg"

def test_iter_unsigners_key_rotation():
    import json
    class MockSigner:
        def __init__(self, secret_key, salt, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value): return value
        def unsign(self, value): return value

    class MockSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj).encode('utf-8')
        def loads(self, data): return json.loads(data.decode('utf-8'))

    # Test with multiple secret keys (rotation)
    serializer = Serializer(secret_key=[b"old_key", b"new_key"], salt=b"salt1")
    unsigners = list(serializer.iter_unsigners())
    
    # Default signer should try both keys: (new_key, salt1) and (old_key, salt1)? 
    # Actually, the code yields self.make_signer(salt) first.
    # make_signer uses the whole secret_keys list for instantiation? 
    # Let's check: make_signer calls self.signer(self.secret_keys, ...)
    # So index 0 is Signer([old, new], salt1).
    # Then it iterates fallback_signers. If empty, only first yield happens.
    assert len(unsigners) == 1
    assert unsigners[0].secret_key == [b"old_key", b"new_key"]

def test_iter_unsigners_with_custom_salt():
    import json
    class MockSigner:
        def __init__(self, secret_key, salt, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value): return value
        def unsign(self, value): return value

    class MockSerializer:
        def dumps(self, obj, **kwargs): return json.dumps(obj).encode('utf-8')
        def loads(self, data): return json.loads(data.decode('utf-8'))

    serializer = Serializer(secret_key=b"key1", salt=b"original_salt")
    unsigners = list(serializer.iter_unsigners(salt=b"override_salt"))
    
    assert len(unsigners) == 1
    assert unsigners[0].salt == b"override_salt"
```


# LLM-generated content at query #6
#--------------------------

```python
def test_serializer_dumps_returns_signed_bytes():
    import json
    from itsdangerous import Serializer, Signer
    
    secret_key = b"secret"
    salt = b"salt"
    serializer = Serializer(secret_key, salt=salt)
    data = {"foo": "bar"}
    
    signed_value = serializer.dumps(data)
    
    assert isinstance(signed_value, bytes)
    assert isinstance(serializer.signer.sign(json.dumps(data).encode("utf-8")), bytes)

def test_serializer_dumps_with_custom_salt():
    import json
    from itsdangerous import Serializer
    
    secret_key = b"secret"
    serializer = Serializer(secret_key)
    data = {"foo": "bar"}
    custom_salt = b"custom_salt"
    
    signed_value_default_salt = serializer.dumps(data)
    signed_value_custom_salt = serializer.dumps(data, salt=custom_salt)
    
    assert signed_value_default_salt != signed_value_custom_salt

def test_serializer_dumps_with_text_serializer_returns_str():
    import json
    from itsdangerous import Serializer
    
    class TextSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj)
        def loads(self, s):
            return json.loads(s)

    secret_key = b"secret"
    serializer = Serializer(secret_key, serializer=TextSerializer())
    data = {"foo": "bar"}
    
    signed_value = serializer.dumps(data)
    
    assert isinstance(signed_value, str)

def test_serializer_dumps_uses_serializer_kwargs():
    import json
    from itsdangerous import Serializer
    
    # We use a custom serializer to verify that kwargs are passed through.
    # Since we can't easily intercept the call without complex mocks in this constraint,
    # we rely on the fact that 'indent' is a valid kwarg for json.dumps.
    secret_key = b"secret"
    serializer = Serialor_kwargs_test_serializer(secret_key, serializer_kwargs={"indent": 4})
    data = {"foo": "bar"}
    
    # If the dump works and we can loads it back, the flow is correct.
    signed_value = serializer.dumps(data)
    assert isinstance(signed_value, bytes)

class Serialor_kwargs_test_serializer(Serializer):
    def __init__(self, secret_key, serializer_kwargs=None):
        super().__init__(secret_key, serializer_kwargs=serializer_kwargs)
```


# LLM-generated content at query #7
#--------------------------

```python
def test_iter_unsigners_skips_fallback_loop_when_no_fallbacks():
    import json
    from itsdangerous import Signer
    # The predicate is: for secret_key in self.secret_keys (line 20)
    # To ensure the loop at line 20 evaluates to False, we need self.secret_keys to be empty.
    # However, __init__ calls _make_keys_list(secret_key). 
    # If secret_key is an empty list/iterable, secret_keys will be empty.
    
    class MockSerializer(Serializer):
        pass

    serializer = MockSerializer(secret_key=b"key1", fallback_signers=[])
    # At this point, self.secret_keys is [b'key1'], so line 20 is True.
    # We must force secret_keys to be empty.
    serializer.secret_keys = []
    
    # To make the loop at line 20 evaluate to False, we need fallback_signers to be non-empty
    # but secret_keys to be empty.
    serializer.fallback_signers = [Signer]
    
    # We iterate through iter_unsigners. 
    # The first yield (line 9) will happen with the primary signer.
    # Then it enters line 11 loop. 
    # Inside the fallback loop, it hits line 20.
    # If secret_keys is empty, line 20 does not execute its body.
    
    unsigners = list(serializer.iter_unsigners())
    
    # Check that we only got the primary signer and no fallbacks were yielded because of empty keys
    assert len(unsigners) == 1
```


# LLM-generated content at query #8
#--------------------------

```python
def test_init_salt_is_not_none():
    import json
    from itsdangerous import Serializer
    
    secret_key = b"secret"
    salt = b"salt"
    serializer = json
    
    serializer_instance = Serializer(secret_key=secret_key, salt=salt, serializer=serializer)
    
    assert serializer_instance.salt is not None
```


# LLM-generated content at query #9
#--------------------------

```python
def test_init_salt_is_not_none():
    import json
    from itsdangerous import Serializer
    s = Serializer(secret_key=b"secret", salt=b"salt")
    assert s.salt is not None
```


# LLM-generated content at query #10
#--------------------------

def test_load_payload_success_with_default_serializer():
    import json
    from unittest.mock import MagicMock
    serializer = Serializer(secret_key="secret")
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_success_with_custom_text_serializer():
    import json
    class TextSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj)
        def loads(self, payload):
            return json.loads(payload)
    
    serializer = Serializer(secret_key="secret", serializer=TextSerializer())
    payload = b'{"a": 1}'
    result = serializer.load_payload(payload)
    assert result == {"a": 1}

def test_load_payload_success_with_custom_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj, **kwargs):
            return b"some_bytes"
        def loads(self, payload):
            return payload.decode("utf-8")

    serializer = Serializer(secret_key="secret", serializer=BytesSerializer())
    payload = b"hello"
    result = serializer.load_payload(payload)
    assert result == "hello"

def test_load_payload_raises_bad_payload_on_error():
    import json
    from itsdangerous import BadPayload
    serializer = Serializer(secret_key="secret")
    invalid_payload = b'{"incomplete_json'
    try:
        serializer.load_payload(invalid_payload)
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
    else:
        raise AssertionError("BadPayload was not raised")

def test_load_payload_with_override_serializer():
    import json
    class CustomSerializer:
        def dumps(self, obj, **kwargs):
            return "custom"
        def loads(self, payload):
            return "decoded_value"

    serializer = Serializer(secret_key="secret")
    payload = b"any_data"
    result = serializer.load_payload(payload, serializer=CustomSerializer())
    assert result == "decoded_value"


# LLM-generated content at query #11
#--------------------------

```python
def test_serializer_init_with_single_bytes_key():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.secret_key == b"secret"

def test_serializer_init_with_single_str_key():
    from itsdangerous.serializer import Serialicalizer
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.secret_key == b"secret"

def test_serializer_init_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=[b"old", b"new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_init_with_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret", salt=b"salt")
    assert s.salt == b"salt"

def test_serializer_init_with_none_salt():
    from itsdegrees.serializer import Serializer
    s = Serializer(secret_key=b"secret", salt=None)
    assert s.salt is None

def test_serializer_init_with_signer_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    s = Serializer(secret_key=b"secret", signer_kwargs={"some": "arg"})
    assert s.signer_kwargs == {"some": "arg"}

def test_serializer_init_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallbacks = [{"extra": "arg"}]
    s = Serializer(secret_key=b"secret", fallback_signers=fallbacks)
    assert s.fallback_signers == fallbacks

def test_serializer_init_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"encoded"
        def loads(self, data):
            return "decoded"
    
    s = Serializer(secret_key=b"secret", serializer=MockSerializer())
    assert s.serializer == MockSerializer()
```


# LLM-generated content at query #12
#--------------------------

```python
def test_serializer_constructor_with_string_key_and_default_params():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == s.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key_and_custom_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret", salt=b"salt")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"salt"

def test_serializer_constructor_with_key_rotation():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=[b"old", b"new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_custom_signer_kwargs():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", signer_kwargs={"some": "arg"})
    assert s.signer_kwargs == {"some": "arg"}

def test_serializer_constructor_with_custom_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallback = [{"some": "dict"}]
    s = Serializer(secret_key="secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_custom_serializer_and_kwargs():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs): return b"data"
        def loads(self, data): return {"obj": "data"}
    
    mock = MockSerializer()
    s = Serializer(secret_key="secret", serializer=mock, serializer_kwargs={"foo": "bar"})
    assert s.serializer == mock
    assert s.serializer_kwargs == {"foo": "bar"}

def test_serializer_constructor_with_bytes_serializer():
    from itsdangerous.serializer import Serializer
    class BytesSerializer:
        def dumps(self, obj, **kwargs): return b"binary"
        def loads(self, data): return b"binary"
    
    s = Serializer(secret_key="secret", serializer=BytesSerializer())
    assert s.is_text_serializer is False
```


# LLM-generated content at query #13
#--------------------------

def test_serializer_constructor_with_basic_args():
    import json
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt="salt")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"salt"
    assert s.serializer == json
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_args():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret", salt=b"salt")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"salt"

def test_serializer_constructor_with_key_rotation():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=[b"old", b"new"], salt="salt")
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, payload):
            return "data"
    
    s = Serializer(secret_key="secret", serializer=MockSerializer())
    assert s.serializer == MockSerializer()
    # Since MockSerializer returns bytes, is_text_serializer should be False
    assert s.is_text_serializer is False

def test_serializer_constructor_with_kwargs():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", serializer_kwargs={"indent": 4}, signer_kwargs={"digest_method": "sha256"})
    assert s.serializer_kwargs == {"indent": 4}
    assert s.signer_kwargs == {"digest_method": "sha256"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    class MockSigner:
        def __init__(self, *args, **kwargs): pass
    
    fallbacks = [{"salt": "alt_salt"}, (MockSigner, {"custom": True})]
    s = Serializer(secret_key="secret", fallback_signers=fallbacks)
    assert s.fallback_signers == fallbacks

def test_serializer_constructor_with_none_values():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt=None, fallback_signers=None)
    assert s.salt is None
    assert s.fallback_signers == []


# LLM-generated content at query #14
#--------------------------

```python
def test_serializer_constructor_with_defaults():
    from itsdangerous import Serializer
    import json
    serializer = Serializer(secret_key="secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer.__name__ == "Signer"
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_keys == {}

def test_serializer_constructor_with_bytes_key():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key=b"secret_bytes")
    assert serializer.secret_keys == [b"secret_bytes"]
    assert serializer.secret_key == b"secret_bytes"

def test_serializer_constructor_with_key_rotation():
    from itsdangerous import Serializer
    keys = ["old", "new"]
    serializer = Serializer(secret_key=keys)
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.secret_key == b"new"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt="mysalt")
    assert serializer.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_signer_kwargs():
    from itsdangerous import Serializer, Signer
    serializer = Serializer(secret_key="secret", signer_kwargs={"some": "arg"})
    assert serializer.signer_kwargs == {"some": "arg"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous import Serializer, Signer
    fallback = [{"some_param": "value"}]
    serializer = Serializer(secret_key="secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_tuple_fallback():
    from itsdangerous import Serializer, Signer
    fallback = [(Signer, {"param": "val"})]
    serializer = Serializer(secret_key="secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

class MockSerializer:
    def dumps(self, obj, **kwargs):
        return b"serialized"
    def loads(self, data):
        return data

def test_serializer_constructor_with_bytes_serializer():
    from itsdangerous import Serializer
    mock = MockSerializer()
    serializer = Serializer(secret_key="secret", serializer=mock)
    assert serializer.serializer == mock
    assert serializer.is_text_serializer is False

class MockTextSerializer:
    def dumps(self, obj, **kwargs):
        return "serialized"
    def loads(self, data):
        return data

def test_serializer_constructor_with_text_serializer():
    from itsencrypt import Serializer # Note: using the provided context logic
    # Assuming is_text_serializer works on the return type of dumps({})
    from itsdangerous import Serializer
    mock = MockTextSerializer()
    serializer = Serializer(secret_key="secret", serializer=mock)
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous import Serializer
    import json
    serializer = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #15
#--------------------------

```python
def test_pdataserializer_loads_returns_expected_value():
    class MockSerializer:
        def loads(self, payload):
            return {"key": "value"}
        def dumps(self, obj):
            return str(obj)

    serializer = MockSerializer()
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}

def test_pdataserializer_loads_with_integer_payload():
    class MockSerializer:
        def loads(self, payload):
            return int(payload)
        def dumps(self, obj):
            return str(obj)

    serializer = MockSerializer()
    payload = "123"
    result = serializer.loads(payload)
    assert result == 123

def test_pdataserializer_loads_raises_error_on_invalid_input():
    class MockSerializer:
        def loads(self, payload):
            return int(payload)
        def dumps(self, obj):
            return str(obj)

    serializer = MockSerializer()
    payload = "not_a_number"
    try:
        serializer.loads(payload)
    except ValueError:
        assert True
    else:
        assert False
```


# LLM-generated content at query #16
#--------------------------

def test_serializer_constructor_with_string_key():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_bytes_key():
    from itsdangerous.serializer import Serializer
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(["old", "new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer("secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, data):
            return "deserialized"
    
    s = Serializer("secret", serializer=MockSerializer())
    assert s.serializer == MockSerializer()

def test_serializer_constructor_with_signer_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    s = Serializer("secret", signer_kwargs={"salt": "extra"})
    assert s.signer_kwargs == {"salt": "extra"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fb = [{"salt": "new_salt"}]
    s = Serializer("secret", fallback_signers=fb)
    assert s.fallback_signers == fb

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer("secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}


# LLM-generated content at query #17
#--------------------------

```python
def test_load_payload_with_default_serializer_and_bytes_payload():
    import json
    from unittest.mock import MagicMock
    
    # Mocking the dependencies needed for the test environment
    class MockSigner:
        def __init__(self, keys, salt=None, **kwargs):
            self.keys = keys
            self.salt = salt
        def sign(self, data): return data
        def unsign(self, data): return data

    # Setup Serializer with a bytes-based serializer (json)
    # We use a dummy secret key and salt
    serializer_instance = Serializer(secret_key=b"secret", salt=b"salt")
    
    payload = b'{"key": "value"}'
    result = serializer_instance.load_payload(payload)
    
    assert result == {"key": "value"}

def test_load_payload_with_override_serializer():
    import json
    
    class MockBytesSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj).encode("utf-8")
        def loads(self, payload):
            return json.loads(payload.decode("utf-8"))

    class MockTextSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj)
        def loads(self, payload):
            return json.loads(payload)

    serializer_instance = Serializer(secret_key=b"secret")
    payload = b'{"a": 1}'
    
    # Use override serializer that expects bytes
    result = serializer_instance.load_payload(payload, serializer=MockBytesSerializer())
    assert result == {"a": 1}

def test_load_payload_with_text_serializer_logic():
    import json
    
    class MockTextSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj)
        def loads(self, payload):
            return json.loads(payload)

    # The logic in load_payload uses is_text_serializer check
    # which depends on serializer.dumps({}).
    serializer_instance = Serializer(secret_key=b"secret", serializer=MockTextSerializer())
    
    # For text serializers, load_payload decodes the payload from bytes to utf-8 first
    payload = b'{"status": "ok"}'
    result = serializer_instance.load_payload(payload)
    
    assert result == {"status": "ok"}

def test_load_payload_raises_bad_payload_on_exception():
    import json
    from itsdangerous import BadPayload

    class BrokenSerializer:
        def dumps(self, obj, **kwargs):
            return b""
        def loads(self, payload):
            raise ValueError("De-serialization failed")

    serializer_instance = Serializer(secret_key=b"secret", serializer=BrokenSerializer())
    
    with Exception as e:
        serializer_instance.load_payload(b"some_data")
    
    assert isinstance(e, BadPayload)
    assert "Could not load the payload" in str(e.args[0])
```


# LLM-generated content at query #18
#--------------------------

```python
def test_serializer_init_with_serializer_provided():
    import json
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj).encode("utf-8")
        def loads(self, data):
            return json.loads(data.decode("utf-8"))

    mock_serializer = MockSerializer()
    serializer = Serializer(secret_key=b"secret", serializer=mock_serializer)
    assert serializer.serializer == mock_serializer
```


# LLM-generated content at query #19
#--------------------------

def test_serializer_constructor_with_string_key():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []

def test_serializer_constructor_with_bytes_key():
    from itsdangerous.serializer import Serializer
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(["key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer("secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, payload):
            return "deserialized"
    
    s = Serializer("secret", serializer=MockSerializer())
    assert s.serializer == MockSerializer

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer("secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_signer_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    s = Serializer("secret", signer_kwargs={"some_arg": "value"})
    assert s.signer_kwargs == {"some_arg": "value"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    s = Serializer("secret", fallback_signers=[{"some_arg": "val"}])
    assert s.fallback_signers == [{"some_arg": "val"}]


# LLM-generated content at query #20
#--------------------------

```python
def test_pdataserializer_loads_returns_correct_type():
    class MockSerializer:
        def loads(self, payload: str) -> dict:
            return {"key": "value"}
        def dumps(self, obj: dict) -> str:
            return "{}"

    serializer = MockSerializer()
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}
    assert isinstance(result, dict)

def test_pdataserializer_loads_with_different_payload_type():
    class IntSerializer:
        def loads(self, payload: int) -> int:
            return payload + 1
        def dumps(self, obj: int) -> int:
            return obj

    serializer = IntSerializer()
    result = serializer.loads(10)
    assert result == 11
```


# LLM-generated content at query #21
#--------------------------

```python
def test_iter_unsigners_default_behavior():
    from itsdangerous import Signer, Serializer
    secret_key = b"secret"
    salt = b"salt"
    serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: x.decode()}
    serializer_is_text = True # Mocking behavior for text serializer logic
    
    # We need to mock is_text_serializer since we can't easily control the import-level function
    import sys
    from unittest.mock import patch
    with patch("itsdangerous.serializer.is_text_serializer", return_value=True):
        s = Serializer(secret_key, salt=salt, serializer=serializer)
        
        unsigners = list(s.iter_unsigners())
        
        assert len(unsignler) == 1
        assert isinstance(unsigners[0], Signer)
        assert unsigners[0].salt == salt
        assert unsigners[0].key == secret_key

def test_iter_unsigners_with_fallback_dict():
    from itsdangerous import Signer, Serializer
    secret_key = b"secret"
    salt = b"salt"
    serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: x.decode()}
    
    # Mocking the signer class to track calls
    class MockSigner:
        def __init__(self, key, salt, **kwargs):
            self.key = key
            self.salt = salt
            self.kwargs = kwargs

    with patch("itsdangerous.serializer.is_text_serializer", return_value=True):
        # fallback_signers as a dict of kwargs for the default signer
        s = Serializer(secret_key, salt=salt, serializer=serializer, fallback_signers=[{"extra": "val"}])
        
        unsigners = list(s.iter_unsigners())
        
        # First is main signer, second is fallback (default signer with extra kwargs)
        assert len(unsigners) == 2
        assert unsigners[1].kwargs["extra"] == "val"

def test_iter_unsigners_with_fallback_tuple():
    from itsdangerous import Signer, Serializer
    secret_key = b"secret"
    salt = b"salt"
    serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: x.decode()}
    
    class MockSigner:
        def __init__(self, key, salt, **kwargs):
            self.key = key
            self.salt = salt
            self.kwargs = kwargs

    with patch("itsdangerous.serializer.is_text_serializer", return_value=True):
        # fallback_signers as a tuple (SignerClass, kwargs)
        s = Serializer(secret_key, salt=salt, serializer=serializer, fallback_signers=[(MockSigner, {"extra": "val"})])
        
        unsigners = list(s.iter_unsigners())
        
        # First is main signer (no extra kwargs), second is the tuple-based signer
        assert len(unsigners) == 2
        assert unsigners[1].key == secret_key
        assert unsigners[1].kwargs["extra"] == "val"

def test_iter_unsigners_with_key_rotation():
    from itsdangerous import Signer, Serializer
    secret_keys = [b"old", b"new"]
    salt = b"salt"
    serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: x.decode()}
    
    class MockSigner:
        def __init__(self, key, salt, **kwargs):
            self.key = key
            self.salt = salt
            self.kwargs = kwargs

    with patch("itsdangerous.serializer.is_text_serializer", return_value=True):
        # Fallback is a class that will iterate through all keys
        s = Serializer(secret_keys, salt=salt, serializer=serializer, fallback_signers=[MockSigner])
        
        unsigners = list(s.iter_unsigners())
        
        # 1 (main signer with newest key) + 2 (fallback signers for each key) = 3
        assert len(unsigners) == 3
        assert unsigners[0].key == b"new"
        assert unsigners[1].key == b"old"
        assert unsigners[2].key == b"new"

def test_iter_unsigners_override_salt():
    from itsdangerous import Signer, Serializer
    secret_key = b"secret"
    salt = b"original_salt"
    serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: x.decode()}
    
    with patch("itsdangerous.serializer.is_text_serializer", return_value=True):
        s = Serializer(secret_key, salt=salt, serializer=serializer)
        
        # Providing a different salt during iteration
        new_salt = b"new_salt"
        unsigners = list(s.iter_unsigners(salt=new_salt))
        
        assert unsigners[0].salt == new_salt
```


# LLM-generated content at query #22
#--------------------------

```python
def test_dumps_returns_serialized_payload():
    class MockSerializer:
        def dumps(self, obj):
            return f"serialized_{obj}"

    serializer = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == "serialized_{'key': 'value'}"

def test_dumps_handles_primitive_types():
    class MockSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = MockSerializer()
    assert serializer.dumps(123) == "123"
    assert serializer.dumps(True) == "True"

def test_dumps_with_complex_object():
    class MockSerializer:
        def dumps(self, obj):
            return obj["id"]

    serializer = MockSerializer()
    payload = {"id": 42, "data": [1, 2, 3]}
    assert serializer.dumps(payload) == 42
```


# LLM-generated content at query #23
#--------------------------

```python
def test_load_payload_raises_bad_payload_on_exception():
    class BrokenSerializer:
        def dumps(self, obj):
            return b"broken"
        def loads(self, data):
            raise ValueError("Deserialization failed")

    import json
    from itsdangerous import Serializer

    serializer = Serializer(secret_key="secret", serializer=json)
    broken_serializer = BrokenSerializer()
    
    # The predicate at line 22 is the 'except Exception as e:' block.
    # To make it evaluate to True (entering the block), we need an exception during loads().
    # To ensure the test checks the logic, we assert that BadPayload is raised.
    # Note: The prompt asks to ensure the predicate evaluates to False. 
    # In Python, 'except Exception' evaluates to True when an exception occurs.
    # However, looking at the provided code structure, line 22 refers to the catch block.
    # To "ensure it evaluates to False" in a logic sense usually means ensuring the success path works,
    # but since the prompt asks for the predicate at that specific line (the exception handler),
    # and standard unit testing of error paths involves triggering it:
    
    try:
        serializer.load_payload(b"some_data", serializer=broken_serializer)
    except Exception as e:
        assert isinstance(e, BadPayload)
        assert isinstance(e.original_error, ValueError)

def test_load_payload_success_path():
    import json
    from itsdangerous import Serializer

    serializer = Serializer(secret_key="secret", serializer=json)
    # This tests the path where the exception block is NOT entered (predicate evaluates to False).
    result = serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #24
#--------------------------

```python
def test_serializer_constructor_with_string_key_and_default_args():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_key():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"s", b"e", b"c", b"r", b"e", b"t"] # Wait, _make_keys_list logic: if isinstance(bytes) -> [want_bytes(secret_key)]
    # Re-evaluating _make_keys_list in provided src: 
    # if isinstance(secret_key, (str, bytes)): return [want_bytes(secret_key)]
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=["key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_signer_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    s = Serializer(secret_key="secret", signer_kwargs={"some_arg": "value"})
    assert s.signer_kwargs == {"some_arg": "value"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallbacks = [{"extra": "kwargs"}, (Signer, {"other": "arg"})]
    s = Serializer(secret_key="secret", fallback_signers=fallbacks)
    assert s.fallback_signers == fallbacks

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs): return b"encoded"
        def loads(self, payload): return "decoded"
    
    s = Serializer(secret_key="secret", serializer=MockSerializer())
    assert s.serializer == MockSerializer()
    # is_text_serializer checks isinstance(serializer.dumps({}), str)
    # Since our mock returns bytes, is_text_serializer should be False
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer(secret_key="secret", serializer=json, serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #25
#--------------------------

```python
def test_serializer_constructor_with_basic_args():
    import json
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.signer == Signer
    assert s.serializer == json
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_args():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret", salt=b"salt")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"salt"

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=["key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, data):
            return "deserialized"
    
    s = Serializer(secret_key="secret", serializer=MockSerializer())
    assert s.serializer == MockSerializer
    assert s.is_text_serializer is False

def test_serializer_constructor_with_custom_signer_and_kwargs():
    from itsdangerous.serializer import Serializer
    class MockSigner:
        def __init__(self, secret_key, salt, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.extra = kwargs.get("extra")
        def sign(self, data):
            return data
        def unsign(self, signature):
            return signature

    s = Serializer(secret_key="secret", signer=MockSigner, signer_kwargs={"extra": "value"})
    signer = s.make_signer()
    assert signer.extra == "value"

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    class MockSigner:
        def __init__(self, secret_key, salt, **kwargs): pass
        def sign(self, data): return data
        def unsign(self, signature): return signature

    s = Serializer(secret_key="secret", fallback_signers=[{"extra": "fallback"}])
    fallbacks = list(s.iter_unsigners())
    assert len(fallbacks) == 1
    # The first signer is the main one, second should be the fallback one from keys
    # Since we have one key, it yields primary then fallback with that key.
    # Primary: Signer(key, salt=...)
    # Fallback (dict): Signer(key, salt=..., extra=fallback)
    assert len(list(s.iter_unsigners())) == 2
```


# LLM-generated content at query #26
#--------------------------

```python
def test_pdataserializer_dumps_returns_payload():
    class MockSerializer:
        def dumps(self, obj):
            return f"serialized_{obj}"

    serializer = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == "serialized_{'key': 'value'}"

def test_pdataserializer_dumps_handles_primitive_types():
    class MockSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = MockSerializer()
    assert serializer.dumps(123) == "123"
    assert serializer.dumps(True) == "True"
```


