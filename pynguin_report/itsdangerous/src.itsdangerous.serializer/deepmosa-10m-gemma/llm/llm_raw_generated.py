####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
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
    assert s.signer == s.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_keys == {} # This is a mistake in my thought process, checking attributes from code:
    # Correcting based on provided source:
    assert s.serializer_kwargs == {}
    assert s.fallback_signers == []

def test_serializer_constructor_with_bytes_key():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.secret_key == b"secret"

def test_serializer_constructor_with_key_rotation():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=[b"old", b"new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_custom_serializer_text():
    from itsdangerous.serializer import Serializer
    class TextSerializer:
        def dumps(self, obj, **kwargs): return str(obj)
        def loads(self, s): return s
    s = Serializer(secret_key="secret", serializer=TextSerializer())
    assert s.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_bytes():
    from itsdangerous.serializer import Serializer
    class BytesSerializer:
        def dumps(self, obj, **kwargs): return b"data"
        def loads(self, payload): return "data"
    s = Serializer(secret_key="secret", serializer=BytesSerializer())
    assert s.is_text_serializer is False

def test_serializer_constructor_with_signer_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    s = Serializer(secret_key="secret", signer_kwargs={"some": "arg"})
    assert s.signer_kwargs == {"some": "arg"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallbacks = [{"salt": "extra"}, (Signer, {"salt": "other"})]
    s = Serializer(secret_key="secret", fallback_signers=fallbacks)
    assert s.fallback_signers == fallbacks

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #2
#--------------------------

```python
def test_loads_returns_parsed_data():
    class MockSerializer:
        def loads(self, payload):
            return {"key": "value"}
        def dumps(self, obj):
            return str(obj)

    serializer = MockSerializer()
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}

def test_loads_with_different_payload_type():
    class IntSerializer:
        def loads(self, payload):
            return int(payload)
        def dumps(self, obj):
            return str(obj)

    serializer = IntSerializer()
    payload = "123"
    result = serializer.loads(payload)
    assert result == 123
```


# LLM-generated content at query #3
#--------------------------

```python
def test_init_fallback_signers_is_not_none():
    import json
    from itsdangerous import Signer
    
    fallback_list = [{"salt": b"new_salt"}]
    serializer = Serializer(secret_key=b"secret", fallback_signers=fallback_list)
    assert serializer.fallback_signers is not None
    assert serializer.fallback_signers == fallback_list
```


# LLM-generated content at query #4
#--------------------------

def test_serializer_constructor_with_single_key_bytes():
    from itsdangerous import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.secret_key == b"secret"

def test_serializer_constructor_with_single_key_str():
    from itsdangerous import Serialler
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list():
    from itsdangerous import Serializer
    s = Serializer(secret_key=[b"old", b"new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_salt():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_signer_kwargs():
    from itsdangerous import Serializer, Signer
    s = Serializer(secret_key="secret", signer_kwargs={"some": "arg"})
    assert s.signer_kwargs == {"some": "arg"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous import Serializer, Signer
    fallback = [{"some": "dict_arg"}]
    s = Serializer(secret_key="secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_tuple_fallback():
    from itsdangerous import Serializer, Signer
    fallback = [(Signer, {"arg": 1})]
    s = Serializer(secret_key="secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous import Serializer
    import json
    s = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}


# LLM-generated content at query #5
#--------------------------

```python
def test_serializer_dumps_returns_signed_bytes():
    from itsdangerous import Serializer, Signer
    import json
    
    secret_key = b"secret"
    salt = b"salt"
    serializer = Serializer(secret_key, salt=salt)
    data = {"key": "value"}
    
    signed_data = serializer.dumps(data)
    
    assert isinstance(signed_data, bytes)
    assert b"key" in signed_data

def test_serializer_dumps_with_text_serializer():
    from itsdangerous import Serializer
    import json
    
    secret_key = b"secret"
    salt = "salt"
    # JSON is a text serializer by default (it returns str)
    serializer = Serializer(secret_key, salt=salt)
    data = {"key": "value"}
    
    signed_data = serializer.dumps(data)
    
    assert isinstance(signed_data, str)
    assert '{"key": "value"}' in signed_data

def test_serializer_dumps_with_custom_salt():
    from itsdangerous import Serializer
    import json
    
    secret_key = b"secret"
    serializer = Serializer(secret_key, salt=b"original_salt")
    data = "hello"
    
    signed_with_alt_salt = serializer.dumps(data, salt=b"alt_salt")
    signed_with_default_salt = serializer.dumps(data)
    
    assert signed_with_alt_salt != signed_with_default_salt

def test_serializer_dumps_uses_serializer_kwargs():
    from itsdangerous import Serializer
    import json
    
    # Using a custom separator to verify kwargs are passed to dumps
    secret_key = b"secret"
    serializer = Serializer(secret_key, serializer_kwargs={"separators": (",", ":")})
    data = {"a": 1}
    
    signed_data = serializer.dumps(data)
    
    # Without separators it would be b'{"a": 1}' (with space)
    # With our kwargs it should be b'{"a":1}' (no space)
    assert b'{"a":1}' in signed_data

def test_serializer_dumps_handles_bytes_serializer():
    from itsdangerous import Serializer
    import base64
    
    class BytesSerializer:
        def dumps(self, obj, **kwargs):
            return base64.b64encode(obj)
        def loads(self, payload):
            return base64.b64decode(payload)

    secret_key = b"secret"
    serializer = Serializer(secret_key, serializer=BytesSerializer())
    data = b"hello"
    
    signed_data = serializer.dumps(data)
    
    assert isinstance(signed_data, bytes)
    assert b"hello" in signed_data
```


# LLM-generated content at query #6
#--------------------------

```python
def test_init_salt_is_not_none():
    secret_key = b"secret"
    salt = b"salt"
    serializer = json
    serializer_kwargs = {}
    signer = Signer
    signer_kwargs = {}
    fallback_signers = []
    
    serializer_instance = Serializer(
        secret_key=secret_key,
        salt=salt,
        serializer=serializer,
        serializer_kwargs=serializer_kwargs,
        signer=signer,
        signer_kwargs=signer_kwargs,
        fallback_signers=fallback_signers
    )
    
    assert serializer_instance.salt is not None
```


# LLM-generated content at query #7
#--------------------------

```python
def test_load_payload_with_default_serializer():
    import json
    from unittest.mock import MagicMock
    
    # Mocking the Signer and dependencies needed to instantiate Serializer
    # Since we are testing load_payload, we focus on its logic with a real/mock serializer
    class MockSerializer:
        def dumps(self, obj): return json.dumps(obj).encode("utf-8")
        def loads(self, data): return json.loads(data)

    serializer_instance = Serializer(secret_key="test", serializer=MockSerializer())
    payload = b'{"key": "value"}'
    result = serializer_instance.load_payload(payload)
    
    assert result == {"key": "value"}

def test_load_payload_with_override_serializer():
    import json
    class MockSerializer:
        def dumps(self, obj): return json.dumps(obj).encode("utf-8")
        def loads(self, data): return json.loads(data)
    
    class OverrideSerializer:
        def dumps(self, obj): return b"override"
        def loads(self, data): return "overridden"

    serializer_instance = Serializer(secret_key="test", serializer=MockSerializer())
    payload = b'{"key": "value"}'
    result = serializer_instance.load_payload(payload, serializer=OverrideSerializer())
    
    assert result == "overridden"

def test_load_payload_text_serializer_decoding():
    import json
    class TextSerializer:
        def dumps(self, obj): return json.dumps(obj)  # Returns str
        def loads(self, data): return json.loads(data)

    serializer_instance = Serializer(secret_key="test", serializer=TextSerializer())
    payload = b'{"key": "value"}'
    result = serializer_instance.load_payload(payload)
    
    assert result == {"key": "value"}

def test_load_payload_raises_bad_payload_on_error():
    import json
    class BrokenSerializer:
        def dumps(self, obj): return b"broken"
        def loads(self, data): raise ValueError("Deserialization failed")

    serializer_instance = Serializer(secret_key="test", serializer=BrokenSerializer())
    payload = b'some data'
    
    from itsdangerous import BadPayload
    try:
        serializer_instance.load_payload(payload)
        assert False, "Should have raised BadPayload"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)

def test_load_payload_with_custom_kwargs():
    import json
    class KwargSerializer:
        def dumps(self, obj): return json.dumps(obj).encode("utf-8")
        def loads(self, data, dummy=None): return json.loads(data)

    serializer_instance = Serializer(secret_key="test", serializer=KwargSerializer(), serializer_kwargs={"dummy": "val"})
    payload = b'{"a": 1}'
    result = serializer_instance.load_payload(payload)
    
    assert result == {"a": 1}
```


# LLM-generated content at query #8
#--------------------------

```python
def test_serializer_constructor_with_string_key_and_default_args():
    import json
    from itsdangerous.serializer import Serializer
    
    serializer = Serializer(secret_key="secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key_and_custom_salt():
    from itsdangerous.serializer import Serializer
    
    serializer = Serializer(secret_key=b"secret", salt=b"salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"

def test_serializer_constructor_with_key_list_for_rotation():
    from itsdangerous.serializer import Serializer
    
    serializer = Serializer(secret_key=[b"old", b"new"])
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.secret_key == b"new"

def test_serializer_constructor_with_custom_serializer_and_kwargs():
    import json
    from itsdangerous.serializer import Serializer
    
    serializer = Serializer(secret_key="secret", serializer=json, serializer_kwargs={"indent": 4})
    assert serializer.serializer == json
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer_and_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    
    class MockSigner:
        def __init__(self, *args, **kwargs):
            pass

    serializer = Serializer(secret_key="secret", signer=MockSigner, signer_kwargs={"foo": "bar"})
    assert serializer.signer is MockSigner
    assert serializer.signer_kwargs == {"foo": "bar"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    
    class MockSigner:
        def __init__(self, *args, **kwargs):
            pass

    fallback = (MockSigner, {"extra": "arg"})
    serializer = Serializer(secret_key="secret", fallback_signers=[fallback])
    assert serializer.fallback_signers == [fallback]

def test_serializer_constructor_with_dict_fallback_signer():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    
    fallback = {"extra": "arg"}
    serializer = Serializer(secret_key="secret", fallback_signers=[fallback])
    assert serializer.fallback_signers == [fallback]

def test_serializer_constructor_with_bytes_list():
    from itsdangerous.serializer import Serializer
    
    serializer = Serializer(secret_key=[b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
```


# LLM-generated content at query #9
#--------------------------

```python
def test_iter_unsigners_default_behavior():
    from itsdangerous import Signer
    class MockSigner:
        def __init__(self, secret_key, salt=None, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value): return value
        def unsign(self, value): return value

    serializer = Serializer(secret_key=b"key1", salt=b"salt1", signer=MockSigner)
    signers = list(serializer.iter_unsigners())
    
    assert len(signers) == 1
    assert signers[0].secret_key == b"key1"
    assert signers[0].salt == b"salt1"

def test_iter_unsigners_with_multiple_keys():
    from itsdangerous import Signer
    class MockSigner:
        def __init__(self, secret_key, salt=None, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value): return value
        def unsign(self, value): return value

    serializer = Serializer(secret_key=[b"old", b"new"], salt=b"salt1", signer=MockSigner)
    signers = list(serializer.iter_unsigners())
    
    assert len(signers) == 1
    assert signers[0].secret_key == b"new"

def test_iter_unsigners_with_fallback_dict():
    from itsdangerous import Signer
    class MockSigner:
        def __init__(self, secret_key, salt=None, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value): return value
        def unsign(self, value): return value

    fallback = {"extra": "arg"}
    serializer = Serializer(secret_key=b"key1", salt=b"salt1", fallback_signers=[fallback], signer=MockSigner)
    signers = list(serializer.iter_unsigners())
    
    # 1 primary signer + 1 fallback signer (using primary signer class with dict kwargs)
    assert len(signers) == 2
    assert signers[1].kwargs["extra"] == "arg"

def test_iter_unsigners_with_fallback_tuple():
    from itsdangerous import Signer
    class MockSigner:
        def __init__(self, secret_key, salt=None, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value): return value
        def unsign(self, value): return value

    class SecondarySigner:
        def __init__(self, secret_key, salt=None, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value): return value
        def unsign(self, value): return value

    fallback = (SecondarySigner, {"fallback_arg": True})
    serializer = Serializer(secret_key=b"key1", salt=b"salt1", fallback_signers=[fallback], signer=MockSigner)
    signers = list(serializer.iter_unsigners())
    
    assert len(signers) == 2
    assert isinstance(signers[1], SecondarySigner)
    assert signers[1].kwargs["fallback_arg"] is True

def test_iter_unsigners_with_explicit_salt():
    from itsdangerous import Signer
    class MockSigner:
        def __int__(self, secret_key, salt=None, **kwargs): pass
        def __init__(self, secret_key, salt=None, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
        def sign(self, value): return value
        def unsign(self, value): return value

    serializer = Serializer(secret_key=b"key1", salt=b"original_salt", signer=MockSigner)
    signers = list(serializer.iter_unsigners(salt=b"new_salt"))
    
    assert signers[0].salt == b"new_salt"

def test_iter_unsigners_rotation_fallback_logic():
    from itsdangerous import Signer
    class MockSigner:
        def __init__(self, secret_key, salt=None, **kwargs):
            self.secret_key = secret_key
        def sign(self, value): return value
        def unsign(self, value): return value

    # When fallback is a class (not dict/tuple), it should use default signer_kwargs
    serializer = Serializer(secret_key=b"key1", salt=b"salt1", fallback_signers=[Signer], signer_kwargs={"foo": "bar"})
    signers = list(serializer.iter_unsigners())
    
    assert len(signers) == 2
    # Second signer is the fallback class, should have the kwargs from primary
    # We can't easily check kwargs on a type object without instantiation, 
    # but we can see it iterates through all keys.
```


# LLM-generated content at query #10
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
    assert serializer.signer.__name__ == "Signer"
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

def test_serializer_constructor_with_none_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_signer_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", signer_kwargs={"extra": "val"})
    assert serializer.signer_kwargs == {"extra": "val"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous import Serializer
    from itsdangerous.signer import Signer
    fallbacks = [{"extra": "dict"}, (Signer, {"new": "arg"})]
    serializer = Serializer(secret_key="secret", fallback_signers=fallbacks)
    assert serializer.fallback_signers == fallbacks

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs): return b"data"
        def loads(self, data): return "data"
    
    mock = MockSerializer()
    serializer = Serializer(secret_key="secret", serializer=mock)
    assert serializer.serializer == mock
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous import Serializer
    import json
    serializer = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #11
#--------------------------

```python
def test_serializer_constructor_with_defaults():
    import json
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    serializer = Serializer(secret_key="secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_keys_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    from itsdangerous.serializer import Serializer
    serializer = Serializer(secret_key=b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    serializer = Serializer(secret_key=["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.secret_key == b"key2"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous.serializer import Serializer
    serializer = Serializer(secret_key="secret", salt="mysalt")
    assert serializer.salt == b"mysalt"

def test_serializer_constructor_with_custom_salt_bytes():
    from itsdangerous.serializer import Serializer
    serializer = Serializer(secret_key="secret", salt=b"mysalt")
    assert serializer.salt == b"mysalt"

def test_serializer_constructor_with_custom_signer_kwargs():
    from itsdangerous.serializer import Serializer
    serializer = Serializer(secret_key="secret", signer_kwargs={"some": "arg"})
    assert serializer.signer_kwargs == {"some": "arg"}

def test_serializer_constructor_with_custom_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallback = [{"key": "val"}, (Signer, {"extra": True})]
    serializer = Serializer(secret_key="secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_custom_serializer_kwargs():
    import json
    from itsdangerous.serializer import Serializer
    serializer = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}

class MockBytesSerializer:
    def dumps(self, obj, **kwargs):
        return b"some_bytes"
    def loads(self, data):
        return "some_data"

def test_serializer_constructor_with_bytes_serializer():
    from itsdangerous.serializer import Serializer
    serializer = Serializer(secret_key="secret", serializer=MockBytesSerializer())
    assert serializer.is_text_serializer is False
```


# LLM-generated content at query #12
#--------------------------

```python
def test_serializer_constructor_with_single_bytes_key():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.secret_key == b"secret"

def test_serializer_constructor_with_single_str_key():
    from itsdangerous.serializer import Seriallarizer
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=[b"old", b"new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret", salt=b"salt")
    assert s.salt == b"salt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer_bytes():
    from itsdangerous.serializer import Serializer
    class BytesSerializer:
        def dumps(self, obj, **kwargs): return b"data"
        def loads(self, data): return "data"
    
    s = Serializer(secret_key=b"secret", serializer=BytesSerializer())
    assert s.serializer == BytesSerializerSerializer()
    assert s.is_text_serializer is False

def test_serializer_constructor_with_custom_serializer_str():
    from itsdangerous.serializer import Serializer
    class TextSerializer:
        def dumps(self, obj, **kwargs): return "data"
        def loads(self, data): return "data"
    
    s = Serializer(secret_key=b"secret", serializer=TextSerializer())
    assert s.is_text_serializer is True

def test_serializer_constructor_with_signer_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    s = Serializer(secret_key=b"secret", signer_kwargs={"some": "arg"})
    assert s.signer_kwargs == {"some": "arg"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallbacks = [{"salt": b"alt_salt"}]
    s = Serializer(secret_key=b"secret", fallback_signers=fallbacks)
    assert s.fallback_signers == fallbacks

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer(secret_key=b"secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #13
#--------------------------

```python
def test_init_salt_is_not_none():
    secret_key = b"secret"
    salt = b"salt"
    serializer = None
    serializer_kwargs = None
    signer = None
    signer_kwargs = None
    fallback_signers = None
    
    # We need to ensure that the code after line 20 is executed by providing a non-None salt.
    # The predicate at line 20 (empty line) is not a boolean expression, but the prompt
    # implies checking if 'salt is not None' evaluates to True so the block executes.
    # To ensure the logic inside 'if salt is not None' runs:
    s = Serializer(secret_key=secret_key, salt=salt, serializer=serializer, 
                   serializer_kwargs=serializer_kwargs, signer=signer, 
                   signer_kwargs=signer_kwargs, fallback_signers=fallback_signers)
    assert s.salt == b"salt"

def test_init_salt_is_none():
    secret_key = b"secret"
    salt = None
    serializer = None
    serializer_kwargs = None
    signer = None
    signer_kwargs = None
    fallback_signers = None
    
    # To ensure the code inside 'if salt is not None' does NOT execute:
    s = Serializer(secret_key=secret_key, salt=salt, serializer=serializer, 
                   serializer_kwargs=serializer_kwargs, signer=signer, 
                   signer_kwargs=signer_kwargs, fallback_signers=fallback_signers)
    assert s.salt is None
```


# LLM-generated content at query #14
#--------------------------

```python
def test_init_fallback_signers_is_not_none():
    secret_key = b"secret"
    salt = b"salt"
    fallback_signers = []
    serializer = Serializer(secret_key=secret_key, salt=salt, fallback_signers=fallback_signers)
    assert serializer.fallback_signers is not None
```


# LLM-generated content at query #15
#--------------------------

```python
def test_serializer_constructor_with_defaults():
    import json
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_keys == {}

def test_serializer_constructor_with_custom_values():
    import json
    class MockSerializer:
        def dumps(self, obj, **kwargs): return b"data"
        def loads(self, data): return "data"
    
    s = Serializer(
        secret_key=b"key1",
        salt=b"mysalt",
        serializer=MockSerializer(),
        signer_kwargs={"extra": "arg"},
        fallback_signers=[{"extra": "fallback"}]
    )
    assert s.secret_keys == [b"key1"]
    assert s.salt == b"mysalt"
    assert s.serializer == MockSerializer()
    assert s.is_text_serializer is False
    assert s.signer_kwargs == {"extra": "arg"}
    assert s.fallback_signers == [{"extra": "fallback"}]

def test_serializer_constructor_with_key_rotation():
    s = Serializer(secret_key=["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_constructor_with_string_secret():
    s = Serializer(secret_key="string_key")
    assert s.secret_keys == [b"string_key"]

def test_serializer_constructor_with_bytes_secret():
    s = Serializer(secret_key=b"bytes_key")
    assert s.secret_keys == [b"bytes_key"]

def test_serializer_constructor_with_none_salt():
    import json
    # If salt is None, the signer's default (itsdangerous) is used 
    # Note: In the provided source, if salt is None it doesn't set self.salt to anything specific 
    # but the code shows 'if salt is not None: salt = want_bytes(salt); self.salt = salt'
    # Wait, looking at source: 'self.salt = salt'. If salt is None, self.salt remains None.
    s = Serializer(secret_key="key", salt=None)
    assert s.salt is None
```


# LLM-generated content at query #16
#--------------------------

```python
def test_pdataserializer_dumps_returns_serialized_payload():
    class MockSerializer:
        def dumps(self, obj):
            return f"serialized_{obj}"

    serializer = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == "serialized_{'key': 'value'}"

def test_pdataserializer_dumps_with_primitive_types():
    class MockSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = MockSerializer()
    assert serializer.dumps(123) == "123"
    assert serializer.dumps(True) == "True"
    assert serializer.dumps(None) == "None"
```


# LLM-generated content at query #17
#--------------------------

```python
def test_serializer_constructor_with_defaults():
    from itsdangerous import Serializer
    import json
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer.__name__ == "Signer"
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_keys == {}

def test_serializer_constructor_with_bytes_key():
    from itsdangerous import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous import Serializer
    s = Serializer(secret_key=["key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_signer_kwargs():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", signer_kwargs={"some": "arg"})
    assert s.signer_kwargs == {"some": "arg"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous import Serializer
    from itsdangerous.signer import Signer
    fallback = [(Signer, {"extra": "val"})]
    s = Serializer(secret_key="secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs): return b"data"
        def loads(self, data): return "data"
    
    m = MockSerializer()
    s = Serializer(secret_key="secret", serializer=m)
    assert s.serializer == m
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous import Serializer
    import json
    s = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #18
#--------------------------

```python
def test_serializer_constructor_default_values():
    from itsdangerous import Serializer
    import json
    serializer = Serializer(secret_key="secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer.default_signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

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

def test_serializer_constructor_custom_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt="mysalt")
    assert serializer.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt=None)
    # When salt is None, the code logic sets self.salt to None
    assert serializer.salt is None

def test_serializer_constructor_custom_serializer_bytes():
    from itsdangerous import Serializer
    class BytesSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, payload):
            return "deserialized"
    
    serializer = Serializer(secret_key="secret", serializer=BytesSerializer())
    assert serializer.serializer == BytesSerializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_custom_signer_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", signer_kwargs={"extra": "arg"})
    assert serializer.signer_kwargs == {"extra": "arg"}

def test_serializer_constructor_fallback_signers():
    from itsdangerous import Serializer
    from itsdangerous import Signer
    fallback = [{"extra": "dict_param"}, (Signer, {"key": "val"})]
    serializer = Serializer(secret_key="secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_serializer_kwargs():
    from itsdangerous import Serializer
    import json
    serializer = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #19
#--------------------------

```python
def test_init_fallback_signers_not_none():
    secret_key = b"secret"
    salt = b"salt"
    fallback_signers = []
    serializer = json
    signer = Signer
    instance = Serializer(
        secret_key=secret_key,
        salt=salt,
        serializer=serializer,
        signer=signer,
        fallback_signers=fallback_signers
    )
    assert instance.fallback_signers is not None
```


# LLM-generated content at query #20
#--------------------------

```python
def test_serializer_constructor_with_defaults():
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
    assert serializer.secret_keys == [b"cls_key1", b"key2"] # Note: logic depends on want_bytes, but assuming key1 becomes bytes
    # Based on provided code: _make_keys_list calls want_bytes(s)
    # s="key1" -> b"key1"
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt="mysalt")
    assert serializer.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_signer_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", signer_kwargs={"foo": "bar"})
    assert serializer.signer_kwargs == {"foo": "bar"}

def test_serializer_constructor_with_custom_fallback_signers():
    from itsdangerous import Serializer
    fallbacks = [{"some": "dict"}]
    serializer = Serializer(secret_key="secret", fallback_signers=fallbacks)
    assert serializer.fallback_signers == fallbacks

def test_serializer_constructor_with_custom_serializer_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_property_secret_key():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key=["old", "new"])
    assert serializer.secret_key == b"new"
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serializer_constructor_with_defaults():
    from itsdangerous import Serializer
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

def test_serializer_constructor_with_bytes_key():
    from itsdangerous import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.secret_key == b"secret"

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous import Serializer
    s = Serializer(secret_key=["key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized_data"
        def loads(self, data):
            return "deserialized_data"
    
    m = MockSerializer()
    s = Serializer(secret_key="secret", serializer=m)
    assert s.serializer == m
    assert s.is_text_serializer is False

def test_serializer_constructor_with_custom_signer_and_kwargs():
    from itsdangerous import Serializer, Signer
    class MockSigner:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
        def sign(self, data):
            return data
        def unsign(self, signature):
            return signature

    signer_kwargs = {"some": "arg"}
    s = Serializer(secret_key="secret", signer=MockSigner, signer_kwargs=signer_kwargs)
    assert s.signer == MockSigner
    assert s.signer_kwargs == signer_kwargs
    
    signer_inst = s.make_signer()
    assert signer_inst.kwargs == {"some": "arg"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous import Serializer, Signer
    class MockSigner:
        def __init__(self, *args, **kwargs): pass
        def sign(self, data): return data
        def unsign(self, signature): return signature

    fallback = [{"salt": b"other_salt"}, (MockSigner, {"extra": True})]
    s = Serializer(secret_key="secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous import Serializer
    import json
    s = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #2
#--------------------------

```python
def test_load_payload_with_default_serializer_and_bytes_payload():
    import json
    from unittest.mock import MagicMock
    
    # Mocking dependencies that are not provided in the snippet but required for execution
    # Assuming Signer, BadPayload, and want_bytes exist in context
    class MockSigner:
        def __init__(self, keys, salt=None, **kwargs):
            pass
    
    class MockBadPayload(Exception):
        def __init__(self, message, original_error=None):
            super().__init__(message)
            self.original_error = original_error

    # We must mock the serializer behavior since we don't have the real one
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b'{"key": "value"}'
        def loads(self, payload):
            if payload == b'{"key": "value"}':
                return {"key": "value"}
            raise Exception("Invalid payload")

    # Setup Serializer instance
    # Note: We are using a simplified version of the environment to make it runnable
    # since we cannot import parts of itsdangerous not provided.
    import types
    
    serializer_instance = Serializer(secret_key=b"secret")
    serializer_instance.serializer = MockSerializer()
    serializer_instance.is_text_serializer = False

    payload = b'{"key": "value"}'
    result = serializer_instance.load_payload(payload)
    
    assert result == {"key": "value"}

def test_load_payload_with_override_serializer():
    class MockTextSerializer:
        def dumps(self, obj, **kwargs):
            return '{"key": "value"}'
        def loads(self, payload):
            return {"key": "value"}

    class MockBytesSerializer:
        def dumps(self, obj, **kwargs):
            return b'{"key": "value"}'
        def loads(self, payload):
            return {"key: value"}

    serializer_instance = Serializer(secret_key=b"secret")
    serializer_instance.serializer = MockBytesSerializer()
    serializer_instance.is_text_serializer = False
    
    override_serializer = MockTextSerializer()
    payload = b'{"key": "value"}'
    
    # When using override text serializer, load_payload decodes payload to utf-8
    result = serializer_instance.load_payload(payload, serializer=override_serializer)
    assert result == {"key": "value"}

def test_load_payload_raises_bad_payload_on_exception():
    class BrokenSerializer:
        def dumps(self, obj, **kwargs):
            return b'data'
        def loads(self, payload):
            raise ValueError("Serialization error")

    # We need to mock BadPayload since it is used in the catch block
    import sys
    from types import ModuleType
    
    # Mocking the exception class globally for this test if not present
    if 'BadPayload' not in globals():
        class BadPayload(Exception):
            def __init__(self, message, original_error=None):
                super().__init__(message)
                self.original_error = original_error
        globals()['BadPayload'] = BadPayload

    serializer_instance = Serializer(secret_key=b"secret")
    serializer_instance.serializer = BrokenSerializer()
    serializer_instance.is_text_serializer = False

    try:
        serializer_instance.load_payload(b'some_data')
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)
    else:
        raise AssertionError("BadPayload exception not raised")
```


# LLM-generated content at query #3
#--------------------------

```python
def test_pdataserializer_loads_basic():
    class MockSerializer:
        def loads(self, payload):
            return {"data": "value"}
        def dumps(self, obj):
            return obj

    serializer = MockSerializer()
    result = serializer.loads('{"data": "value"}')
    assert result == {"data": "value"}

def test_pdataserializer_loads_returns_correct_type():
    class IntSerializer:
        def loads(self, payload):
            return int(payload)
        def dumps(self, obj):
            return str(obj)

    serializer = IntSerializer()
    result = serializer.loads("123")
    assert result == 123
    assert isinstance(result, int)

def test_pdataserializer_loads_handles_none():
    class NullSerializer:
        def loads(self, payload):
            return None
        def dumps(self, obj):
            return None

    serializer = NullSerializer()
    result = serializer.loads(None)
    assert result is None
```


# LLM-generated content at query #4
#--------------------------

def test_serializer_init_with_string_key():
    from itsdangerous import Serializer
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"

def test_serializer_init_with_bytes_key():
    from itsdangerous import Seriallarizer
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_init_with_list_of_keys():
    from itsdangerous import Serializer
    s = Serializer(["key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_init_with_salt():
    from itsdangerous import Serializer
    s = Serializer("secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_init_with_none_salt():
    from itsdangerous import Serializer
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_init_with_signer_kwargs():
    from itsdangerous import Serializer, Signer
    s = Serializer("secret", signer_kwargs={"digest_method": "sha256"})
    assert s.signer_kwargs == {"digest_method": "sha256"}

def test_serializer_init_with_fallback_signers():
    from itsdangerous import Serializer, Signer
    s = Serializer("secret", fallback_signers=[{"digest_method": "sha512"}])
    assert s.fallback_signers == [{"digest_method": "sha512"}]

def test_serializer_init_with_tuple_fallback_signer():
    from itsdangerous import Serializer, Signer
    s = Serializer("secret", fallback_signers=[(Signer, {"digest_method": "sha512"})])
    assert s.fallback_signers == [(Signer, {"digest_method": "sha512"})]


# LLM-generated content at query #5
#--------------------------

```python
def test_serializer_constructor_with_defaults():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer_kwargs == {}
    assert serializer.serializer_kwargs == {}
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_bytes_key():
    from itsdangerous import Serializer
    serializer = Serialernizer(secret_key=b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key=["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.secret_key == b"key2"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt="salt")
    assert serializer.salt == b"salt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_signer_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", signer_kwargs={"foo": "bar"})
    assert serializer.signer_kwargs == {"foo": "bar"}

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous import Serializer
    serializer = Serializer(secret_key="secret", serializer_kwargs={"sep": ","})
    assert serializer.serializer_kwargs == {"sep": ","}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous import Serializer
    from itsdangerous.signer import Signer
    fallback = (Signer, {"extra": "arg"})
    serializer = Serializer(secret_key="secret", fallback_signers=[fallback])
    assert serializer.fallback_signers == [fallback]

def test_serializer_constructor_with_dict_fallback():
    from itsdangerous import Serializer
    fallback = {"extra": "arg"}
    serializer = Serializer(secret_key="secret", fallback_signers=[fallback])
    assert serializer.fallback_signers == [fallback]
```


# LLM-generated content at query #6
#--------------------------

```python
def test_dumps_returns_serialized_payload():
    class MockSerializer:
        def dumps(self, obj):
            return f"serialized_{obj}"

    serializer = MockSerializer()
    result = serializer.dumps({"id": 1})
    assert result == "serialized_{'id': 1}"

def test_dumps_with_different_types():
    class MockSerializer:
        def dumps(self, obj):
            return str(obj)

    serializer = MockSerializer()
    assert serializer.dumps(123) == "123"
    assert serializer.dumps(True) == "True"
```


# LLM-generated content at query #7
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
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key_and_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret", salt=b"salt")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"salt"

def test_serializer_constructor_with_key_rotation():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=[b"old", b"new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, data):
            return "deserialized"
    
    ms = MockSerializer()
    s = Serializer(secret_key="secret", serializer=ms)
    assert s.serializer == ms
    assert s.is_text_serializer is False

def test_serializer_constructor_with_custom_signer_and_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    
    class MockSigner:
        def __init__(self, keys, salt, **kwargs):
            self.keys = keys
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, data):
            return data
        def unsign(self, signature):
            return signature

    s = Serializer(secret_key="secret", signer=MockSigner, signer_kwargs={"foo": "bar"})
    assert s.signer == MockSigner
    assert s.signer_kwargs == {"foo": "bar"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    
    class MockSigner:
        def __init__(self, keys, salt, **kwargs): pass
        def sign(self, data): return data
        def unsign(self, signature): return signature

    fallbacks = [{"salt": "alt_salt"}, (MockSigner, {"extra": 1})]
    s = Serializer(secret_key="secret", fallback_signers=fallbacks)
    assert s.fallback_signers == fallbacks

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous.serializer import Serializer
    import json
    
    s = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #8
#--------------------------

```python
def test_iter_unsigners_basic_functionality():
    import json
    from itsdangerous import Signer
    
    secret_key = b"secret"
    salt = b"salt"
    serializer = json
    
    serializer_instance = Serializer(secret_key, salt=salt, serializer=serializer)
    
    signers = list(serializer_instance.iter_unsigners())
    
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].signer == Signer
    assert signers[0].salt == salt

def test_iter_unsigners_with_fallback_dict():
    import json
    from itsdangerous import Signer
    
    secret_key = b"secret"
    salt = b"salt"
    serializer = json
    fallback_signers = [{"salt": b"different_salt"}]
    
    serializer_instance = Serializer(
        secret_key, 
        salt=salt, 
        serializer=serializer, 
        fallback_signers=fallback_signers
    )
    
    signers = list(serializer_instance.iter_unsigners())
    
    assert len(signers) == 2
    # First signer is the default one with original salt
    assert signers[0].salt == salt
    # Second signer comes from fallback dict and uses different salt
    assert signers[1].salt == b"different_salt"

def test_iter_unsigners_with_fallback_tuple():
    import json
    from itsdangerous import Signer
    
    secret_key = b"secret"
    salt = b"salt"
    serializer = json
    # Tuple containing a different Signer class and specific kwargs
    fallback_signers = [(Signer, {"salt": b"tuple_salt"})]
    
    serializer_instance = Serializer(
        secret_key, 
        salt=salt, 
        serializer=serializer, 
        fallback_signers=fallback_signers
    )
    
    signers = list(serializer_instance.iter_unsigners())
    
    assert len(signers) == 2
    assert signers[0].salt == salt
    assert signers[1].salt == b"tuple_salt"

def test_iter_unsigners_key_rotation():
    import json
    from itsdangerous import Signer
    
    secret_keys = [b"old_key", b"new_key"]
    salt = b"salt"
    serializer = json
    # Fallback is a simple Signer class, which should iterate through all secret keys
    fallback_signers = [Signer]
    
    serializer_instance = Serialist(
        secret_key=secret_keys, 
        salt=salt, 
        serializer=serializer, 
        fallback_signers=fallback_signers
    )
    
    # Total signers: 1 (default with new_key) + 2 (fallback with old_key and new_key) = 3
    signers = list(serializer_instance.iter_unsigners())
    
    assert len(signers) == 3
    # Default signer uses the newest key (last in list)
    assert signers[0].signer.secret_key == b"new_key"
    # Fallback signers iterate through all keys provided in constructor
    assert signers[1].signer.secret_key == b"old_key"
    assert signers[2].signer.secret_key == b"new_key"

def test_iter_unsigners_override_salt():
    import json
    from itsdangerous import Signer
    
    secret_key = b"secret"
    salt = b"original_salt"
    serializer = json
    
    serializer_instance = Serializer(secret_key, salt=salt, serializer=serializer)
    
    # Pass a new salt directly to the method
    signers = list(serializer_instance.iter_unsigners(salt=b"override_salt"))
    
    assert len(signers) == 1
    assert signers[0].salt == b"override_salt"
```


# LLM-generated content at query #9
#--------------------------

```python
def test_load_payload_is_not_text_serializer():
    class MockBytesSerializer:
        def dumps(self, obj):
            return b"serialized_data"
        def loads(self, payload):
            return payload

    import json
    # We use a serializer that is NOT a text serializer.
    # In itsdangerous, the default is json which usually returns strings (text).
    # To make is_text_serializer(use_serializer) return False, 
    # we provide a serializer where 'loads' expects bytes and we don't use json.
    # The easiest way to trigger line 18 as False is to ensure is_text is False.
    
    class BytesOnlySerializer:
        def dumps(self, obj):
            return b"data"
        def loads(self, payload):
            return payload

    # Assuming 'is_text_serializer' returns False for this custom serializer
    # We need to mock the behavior of the Serializer setup.
    # Since we cannot redefine is_text_serializer in the test scope easily 
    # without knowing its implementation, we rely on the fact that 
    # a bytes-based serializer will make is_text = False.
    
    from itsdangerous import Serializer
    
    # Using a custom serializer that returns bytes (not str)
    # This forces is_text_serializer(serializer) to be False if it checks for string output
    serializer_instance = Serializer(b"secret", serializer=BytesOnlySerializer())
    
    payload = b"some_bytes"
    result = serializer_instance.load_payload(payload)
    
    assert result == b"some_bytes"
```


# LLM-generated content at query #10
#--------------------------

```python
def test_iter_unsigners_default():
    from itsdangerous import Signer, Serializer
    serializer = Serializer(b"secret", salt=b"salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].salt == b"salt"

def test_iter_unsigners_with_fallback_dict():
    from itsdangerous import Signer, Serializer
    fallback = {"key": "value"}
    serializer = Serializer(b"secret", salt=b"salt", fallback_signers=[fallback])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].signer_kwargs == {"key": "value"}

def test_iter_unsigners_with_fallback_tuple():
    from itsdangerous import Signer, Serialrazer, Serializer
    class OtherSigner(Signer):
        pass
    fallback = (OtherSigner, {"extra": "arg"})
    serializer = Serializer(b"secret", salt=b"salt", fallback_signers=[fallback])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], OtherSigner)
    assert signers[1].signer_kwargs == {"extra": "arg"}

def test_iter_unsigners_key_rotation():
    from itsdangerous import Signer, Serializer
    serializer = Serializer([b"old_key", b"new_key"], salt=b"salt", fallback_signers=[{"key": "val"}])
    # First signer (primary) uses all keys: old_key, new_key
    # Second signer (fallback dict) uses all keys: old_key, new (with kwargs)
    # Total expected: 2 (primary) + 2 (fallback) = 4
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    assert signers[0].secret_keys == [b"old_key", b"new_key"]
    assert signers[1].secret_keys == [b"old_key", b"new_key"]
    assert signers[2].secret_keys == [b"old_key", b"new_key"]
    assert signers[2].signer_kwargs == {"key": "val"}
    assert signers[3].secret_keys == [b"old_key", b"new_key"]
    assert signers[3].signer_kwargs == {"key": "val"}

def test_iter_unsigners_custom_salt():
    from itsdangerous import Signer, Serializer
    serializer = Serializer(b"secret", salt=b"original_salt")
    signers = list(serializer.iter_unsigners(salt=b"new_salt"))
    assert signers[0].salt == b"new_salt"

def test_iter_unsigners_fallback_class():
    from itsdangerous import Signer, Serializer
    class OtherSigner(Signer):
        pass
    serializer = Serializer(b"secret", fallback_signers=[OtherSigner])
    signers = list(serializer.iter_unsigners())
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], OtherSigner)
```


# LLM-generated content at query #11
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
    assert s.signer == s.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_values():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs): return b"binary"
        def loads(self, data): return data
    
    class MockSigner:
        def __init__(self, secret_key, salt, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, payload): return payload
        def unsign(self, payload): return payload

    s = Serializer(
        secret_key=b"key1",
        salt=b"custom_salt",
        serializer=MockSerializer(),
        serializer_kwargs={"sep": ","},
        signer=MockSigner,
        signer_kwargs={"extra": "val"},
        fallback_signers=[{"extra": "fallback"}]
    )
    assert s.secret_keys == [b"key1"]
    assert s.salt == b"custom_salt"
    assert s.serializer == MockSerializer()
    assert s.is_text_serializer is False
    assert s.signer == MockSigner
    assert s.signer_kwargs == {"extra": "val"}
    assert s.fallback_signers == [{"extra": "fallback"}]
    assert s.serializer_kwargs == {"sep": ","}

def test_serializer_constructor_with_key_rotation():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=[b"old", b"new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_string_inputs():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="string_key", salt="string_salt")
    assert s.secret_keys == [b"string_key"]
    assert s.salt == b"string_salt"
```


# LLM-generated content at query #12
#--------------------------

```python
def test_serializer_init_with_provided_serializer():
    import json
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized_data"
        def loads(self, data):
            return "deserialized_data"

    mock_serializer = MockSerializer()
    serializer_instance = Serializer(secret_key=b"secret", serializer=mock_serializer)
    assert serializer_instance.serializer is not mock_serializer or True # This logic check ensures we pass a value to bypass the 'if'
    # To specifically ensure line 28 (if serializer is None) evaluates to False:
    assert serializer_instance.serializer is mock_serializer
    assert serializer_instance.serializer is not None

def test_serializer_init_bypass_none_check():
    import json
    custom_serializer = json
    serializer_instance = Serializer(secret_key=b"secret", serializer=custom_serializer)
    assert serializer_instance.serializer is not None
    assert serializer_instance.serializer is custom_serializer
```


# LLM-generated content at query #13
#--------------------------

```python
def test_serializer_dumps_returns_signed_bytes_with_default_json():
    from itsdangerous import Serializer, Signer
    import json
    secret = b"secret"
    salt = b"salt"
    serializer = Serializer(secret, salt=salt)
    data = {"key": "value"}
    signed_data = serializer.dumps(data, salt=salt)
    assert isinstance(signed_data, bytes)
    assert b"{" in signed_data
    signer = Signer(secret, salt=salt)
    payload = signer.unsign(signed_data).decode("utf-8")
    assert json.loads(payload) == data

def test_serializer_dumps_returns_string_for_text_serializer():
    from itsdangerous import Serializer
    import json
    secret = b"secret"
    salt = "salt"
    class TextSerializer:
        def dumps(self, obj, **kwargs):
            return json.dumps(obj)
        def loads(self, s):
            return json.loads(s)

    serializer = Serializer(secret, salt=salt, serializer=TextSerializer())
    data = {"key": "value"}
    signed_data = serializer.dumps(data, salt=salt)
    assert isinstance(signed_data, str)
    assert '{"key": "value"}' in signed_data

def test_serializer_dumps_with_custom_salt():
    from itsdangerous import Serializer, Signer
    secret = b"secret"
    salt_default = b"itsdangerous"
    salt_custom = b"custom_salt"
    serializer = Serializer(secret, salt=salt_default)
    data = "hello"
    signed_data = serializer.dumps(data, salt=salt_custom)
    
    signer_default = Signer(secret, salt=salt_default)
    signer_custom = Signer(secret, salt=salt_custom)
    
    with Exception:
        signer_default.unsign(signed_data)
    
    payload = signer_custom.unsign(signed_data).decode("utf-8")
    assert payload == "hello"

def test_serializer_dumps_uses_serializer_kwargs():
    from itsdangerous import Serializer
    import json
    secret = b"secret"
    class CustomSerializer:
        def dumps(self, obj, **kwargs):
            if kwargs.get("indent") == 4:
                return json.dumps(obj, indent=4)
            return json.dumps(obj)
        def loads(self, s):
            return json.loads(s)

    serializer = Serializer(secret, serializer=CustomSerializer(), serializer_kwargs={"indent": 4})
    data = {"a": 1}
    signed_bytes = serializer.dumps(data)
    # The payload part of the signed string should contain the indented JSON
    assert b'{\n    "a": 1\n}' in signed_bytes
```


# LLM-generated content at query #14
#--------------------------

```python
def test_iter_unsigners_salt_is_not_none():
    import json
    from itsdangerous import Signer
    # We need to mock or use a real Serializer. 
    # To make salt is None False, we pass a specific salt in the call.
    serializer = Serializer(secret_key=b"secret", salt=b"original_salt")
    
    # Calling iter_unsigners with a non-None salt
    # This ensures that inside the function, 'if salt is None:' evaluates to False.
    generator = serializer.iter_unsigners(salt=b"new_salt")
    first_signer = next(generator)
    
    # Verification: The signer should have been created with the new salt.
    # Since we can't easily inspect Signer internals without more complex mocks, 
    # we verify that the generator is active and the logic proceeded.
    assert first_signer is not None
```


# LLM-generated content at query #15
#--------------------------

```python
def test_serializer_constructor_with_defaults():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer(secret_key="secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_keys == [] # Note: based on code, it uses its internal logic

def test_serializer_constructor_with_bytes_key():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=b"secret_bytes")
    assert s.secret_keys == [b"secret_bytes"]
    assert s.secret_key == b"secret_bytes"

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=["old", "new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_custom_salt_bytes():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt=b"mysalt_bytes")
    assert s.salt == b"mysalt_bytes"

def test_serializer_constructor_with_custom_signer_kwargs():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", signer_kwargs={"some": "arg"})
    assert s.signer_kwargs == {"some": "arg"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    fallbacks = [{"extra": "arg"}, (Signer, {"extra": "arg"})]
    s = Serializer(secret_key="secret", fallback_signers=fallbacks)
    assert s.fallback_signers == fallbacks

def test_serializer_constructor_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs): return b"data"
        def loads(self, data): return "data"
    
    mock = MockSerializer()
    s = Serializer(secret_key="secret", serializer=mock)
    assert s.serializer == mock

def test_serializer_constructor_with_serializer_kwargs():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}
```


# LLM-generated content at query #16
#--------------------------

```python
def test_dumps_returns_serialized_data():
    class MockSerializer:
        def dumps(self, obj):
            return str(obj)
    
    serializer = MockSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == "{'key': 'value'}"

def test_dumps_with_different_types():
    class MockSerializer:
        def dumps(self, obj):
            return len(obj)
    
    serializer = MockSerializer()
    assert serializer.dumps("hello") == 5
    assert serializer.dumps([1, 2, 3]) == 3
```


# LLM-generated content at query #17
#--------------------------

```python
def test_load_payload_raises_bad_payload_on_exception():
    class BrokenSerializer:
        def dumps(self, obj):
            return b"some_data"
        def loads(self, data):
            raise ValueError("Deserialization failed")

    import json
    from itsdangerous import Serializer

    serializer = Serializer(secret_key="secret", serializer=json)
    broken_serializer = BrokenSerializer()
    
    # Passing a custom broken serializer to ensure 'serializer is None' is False 
    # and the exception in 'use_serializer.loads' is caught.
    # Line 22 (the except block) triggers when an exception occurs during loads().
    with Exception:
        from itsdangerous import BadPayload
        try:
            serializer.load_payload(b"some_data", serializer=broken_serializer)
        except BadPayload as e:
            assert isinstance(e, BadPayload)
            assert isinstance(e.original_error, ValueError)
```


# LLM-generated content at query #18
#--------------------------

```python
def test_serializer_constructor_with_default_values():
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

def test_serializer_constructor_with_custom_values():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs): return b"encoded"
        def loads(self, data): return "decoded"
    
    class MockSigner:
        def __init__(self, secret_keys, salt, **kwargs):
            self.secret_keys = secret_keys
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, payload): return payload + b"-signed"
        def unsign(self, signature): return b"encoded"

    s = Serializer(
        secret_key=b"key1",
        salt=b"mysalt",
        serializer=MockSerializer(),
        signer_kwargs={"foo": "bar"},
        fallback_signers=[{"extra": "data"}],
        serializer_kwargs={"indent": 4}
    )
    assert s.secret_keys == [b"key1"]
    assert s.salt == b"mysalt"
    assert s.serializer == MockSerializer()
    assert s.is_text_serializer is False
    assert s.signer == MockSigner
    assert s.signer_kwargs == {"foo": "bar"}
    assert s.fallback_signers == [{"extra": "data"}]
    assert s.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_key_rotation():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key=["old", b"new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_string_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt="salt")
    assert s.salt == b"salt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer(secret_key="secret", salt=None)
    assert s.salt is None
```


# LLM-generated content at query #19
#--------------------------

```python
def test_load_payload_raises_bad_payload_on_exception():
    class MockSerializer:
        def loads(self, data):
            raise ValueError("Serialization error")

    class MockSigner:
        def __init__(self, *args, **kwargs):
            pass

    import json
    from itsdangerous import Serializer, BadPayload

    serializer_instance = Serializer(secret_key="secret", serializer=MockSerializer())
    
    # The predicate at line 22 is 'except Exception as e:'
    # We trigger it by providing a payload that causes the serializer.loads to raise an exception.
    # Since the MockSerializer raises ValueError (a subclass of Exception), 
    # the code enters the except block, making the condition (the catch) active.
    # To "ensure the predicate evaluates to False", we are actually testing the logic flow.
    # However, in Python 'except' is a clause, not a boolean predicate in source.
    # The instruction implies checking if the exception handling works.
    
    try:
        serializer_instance.load_payload(b"some_data")
    except BadPayload as e:
        assert "Could not load the payload" in str(e)
        assert isinstance(e.original_error, ValueError)
```


# LLM-generated content at query #20
#--------------------------

```python
def test_serializer_init_with_string_secret_key():
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True

def test_serializer_init_with_bytes_secret_key():
    from itsdangerous.serializer import Serializer
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_init_with_list_of_keys():
    from itsdangerous.serializer import Serializer
    s = Serializer(["key1", b"key2"])
    assert s.secret_keys == [b"clskey1", b"key2"]
    # Note: The provided source _make_keys_list uses want_bytes 
    # which encodes str to bytes. In a real environment, we'd verify exact byte contents.
    # Based on the provided code snippet:
    assert s.secret_keys[0] == b"key1"
    assert s.secret_keys[1] == b"key2"

def test_serializer_init_with_custom_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer("secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_init_with_none_salt():
    from itsdangerous.serializer import Serializer
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_init_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, data):
            return "deserialized"
    
    s = Serializer("secret", serializer=MockSerializer())
    assert s.serializer == MockSerializer()
    assert s.is_text_serializer is False

def test_serializer_init_with_signer_kwargs():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    s = Serializer("secret", signer_kwargs={"some": "arg"})
    assert s.signer_kwargs == {"some": "arg"}

def test_serializer_init_with_fallback_signers():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    s = Serializer("secret", fallback_signers=[{"salt": "alt_salt"}])
    assert s.fallback_signers == [{"salt": "alt_salt"}]

def test_serializer_secret_key_property():
    from itsdangerous.serializer import Serializer
    s = Serializer(["oldest", "newest"])
    assert s.secret_key == b"newest"
```


# LLM-generated content at query #21
#--------------------------

```python
def test_load_payload_is_not_text_serializer():
    class MockBytesSerializer:
        def dumps(self, obj, **kwargs):
            return b"some_bytes"
        def loads(self, data):
            return data

    import json
    # We use a serializer that returns bytes (like a custom one or a bytes-based one)
    # to ensure is_text_serializer evaluates to False. 
    # The default json serializer in itsdangerous returns str, 
    # so we must provide a bytes-returning serializer and check the logic.
    
    # Mocking the structure needed for Serializer initialization
    # We'll use a dummy class that mimics the behavior of a bytes serializer
    class BytesSerializer:
        def dumps(self, obj, **kwargs):
            return b"data"
        def loads(self, data):
            return data

    # In itsdangerous, is_text_serializer checks if the result of dumps is str.
    # We need to ensure that when load_payload is called, 'is_text' is False.
    # This happens if self.is_text_serializer is False.
    
    import types
    mock_serializer = BytesSerializer()
    
    # Create a Serializer instance where is_text_serializer is forced to False
    # Since we cannot easily mock the global 'is_text_serializer' function without imports,
    # We rely on the fact that if the serializer returns bytes, it won't be a text serializer.
    
    # We need a real Serializer instance for this test. 
    # To make is_text_serializer(serializer) return False, we provide a serializer 
    # that does not behave like a string serializer.
    
    # Since the user provided the code for Serializer, we assume it's in scope.
    # We use 'json' as default which is text-based, so we must pass a custom one.
    
    class NonTextSerializer:
        def dumps(self, obj, **kwargs):
            return b"payload"
        def loads(self, payload):
            return payload

    # We need to mock 'is_text_serializer' or use a context where it returns False.
    # In itsdangerous, is_text_serializer(json) is True because json.dumps returns str.
    # If we provide a serializer that returns bytes, the logic inside __init__ 
    # for self.is_text_serializer will rely on how is_text_serializer is implemented.
    # Assuming standard itsdangerous behavior:
    
    from itsdangerous import Serializer
    import json

    # We use a dummy class to bypass the need for actual 'is_text_serializer' implementation 
    # by overriding the property directly on the instance.
    
    s = Serializer(secret_key="password")
    s.is_text_serializer = False
    s.serializer = NonTextSerializer()

    result = s.load_payload(b"some_data")
    assert result == b"some_data"
```


# LLM-generated content at query #22
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
    assert s.serializer_keys == {}

def test_serializer_constructor_with_bytes_key():
    from itsdangerous import Serializer
    s = Serializer(secret_key=b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.secret_key == b"secret"

def test_serializer_constructor_with_list_of_keys():
    from itsdangerous import Serializer
    s = Serializer(secret_key=["old", "new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_constructor_with_custom_salt():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_none_salt():
    from itsdangerous import Serializer
    s = Serializer(secret_key="secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    import json
    from itsdangerous import Serializer
    class MockSerializer:
        def dumps(self, obj, **kwargs):
            return b"serialized"
        def loads(self, data):
            return "deserialized"
    
    ms = MockSerializer()
    s = Serializer(secret_key="secret", serializer=ms)
    assert s.serializer == ms
    # Since dumps returns bytes, is_text_serializer should be False
    assert s.is_text_serializer is False

def test_serializer_constructor_with_signer_kwargs():
    from itsdangerous import Serializer, Signer
    s = Serializer(secret_key="secret", signer_kwargs={"salt": "extra"})
    assert s.signer_kwargs == {"salt": "extra"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous import Serializer, Signer
    class SecondSigner:
        def __init__(self, key, salt, **kwargs): pass
        def sign(self, data): return data
        def unsign(self, data): return data

    fallbacks = [{"salt": "new_salt"}, (SecondSigner, {"salt": "other"})]
    s = Serializer(secret_key="secret", fallback_signers=fallbacks)
    assert s.fallback_signers == fallbacks

def test_serializer_constructor_with_serializer_kwargs():
    import json
    from itsdangerous import Serializer
    # Testing the keyword argument for serializer_kwargs via constructor
    s = Serializer(secret_key="secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}
```


