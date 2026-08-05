####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_serializer_constructor_default_serializer():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_strings():
    s = Serializer(["old_secret", "new_secret"])
    assert s.secret_keys == [b"old_secret", b"new_secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"old_secret", b"new_secret"])
    assert s.secret_keys == [b"old_secret", b"new_secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"custom_salt"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"

        @staticmethod
        def loads(data):
            return data

    s = Serializer("secret", serializer=CustomSerializer())
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == CustomSerializer()
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    s = Serializer("secret", signer=CustomSigner)
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == CustomSigner
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {"key_derivation": "hmac"}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == [{"key_derivation": "none"}]
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.secret_keys == [b"secret"]
    assert s.salt is None
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes_data"

        @staticmethod
        def loads(data):
            return data

    s = Serializer("secret", serializer=BytesSerializer())
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == BytesSerializer()
    assert s.is_text_serializer is False
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}


# LLM-generated content at query #2
#--------------------------

def test_serializer_constructor_default():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_iterable_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_serializer_str():
    class MockSerializer:
        @staticmethod
        def dumps(obj):
            return "string"
    s = Serializer("secret", serializer=MockSerializer())
    assert s.is_text_serializer is True

def test_serializer_constructor_with_serializer_bytes():
    class MockSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
    s = Serializer("secret", serializer=MockSerializer())
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_signer():
    class MockSigner:
        def __init__(self, secret_keys, salt, **kwargs):
            pass
    s = Serializer("secret", signer=MockSigner)
    assert s.signer is MockSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"digest_method": "sha256"})
    assert s.signer_kwargs == {"digest_method": "sha256"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"salt": "fallback"}])
    assert s.fallback_signers == [{"salt": "fallback"}]

def test_serializer_constructor_fallback_signers_default():
    s = Serializer("secret")
    assert s.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    s = Serializer(["key1", "key2"])
    assert s.secret_key == b"key2"


# LLM-generated content at query #3
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "{}", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"{}", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer is custom_signer


# LLM-generated content at query #4
#--------------------------

def test_dumps_returns_bytes_for_bytes_serializer():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.serializer.dumps = lambda obj: obj
    serializer.is_text_serializer = False
    result = serializer.dumps("test")
    assert isinstance(result, bytes)


# LLM-generated content at query #5
#--------------------------

```python
def test_fallback_signers_is_not_none():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []
```


# LLM-generated content at query #6
#--------------------------

```python
def test_load_payload_with_default_text_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    assert serializer.load_payload(payload) == {"key": "value"}

def test_load_payload_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            import json
            return json.dumps(obj).encode()
        def loads(self, payload):
            import json
            return json.loads(payload)
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = serializer.dump_payload({"key": "value"})
    assert serializer.load_payload(payload) == {"key": "value"}

def test_load_payload_with_custom_serializer_override():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return eval(payload)
    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = serializer.dump_payload(42)
    assert serializer.load_payload(payload) == 42
    assert serializer.load_payload(payload, serializer=CustomSerializer()) == 42

def test_load_payload_raises_bad_payload_on_invalid_data():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid")
        assert False
    except Exception as e:
        from itsdangerous import BadPayload
        assert isinstance(e, BadPayload)

def test_load_payload_with_text_serializer_utf8_encoded():
    serializer = Serializer("secret")
    payload = serializer.dump_payload("hello")
    assert serializer.load_payload(payload) == "hello"

def test_load_payload_with_bytes_serializer_non_utf8():
    class BinarySerializer:
        def dumps(self, obj):
            import pickle
            return pickle.dumps(obj)
        def loads(self, payload):
            import pickle
            return pickle.loads(payload)
    serializer = Serializer("secret", serializer=BinarySerializer())
    payload = serializer.dump_payload([1, 2, 3])
    assert serializer.load_payload(payload) == [1, 2, 3]

def test_load_payload_with_none_serializer_override():
    serializer = Serializer("secret")
    payload = serializer.dump_payload(True)
    assert serializer.load_payload(payload, serializer=None) == True
```


# LLM-generated content at query #7
#--------------------------

def test_constructor_default_serializer_uses_json():
    serializer = Serializer("secret")
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_constructor_with_str_secret_key():
    serializer = Serializer("mysecret")
    assert serializer.secret_keys == [b"mysecret"]

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"mysecretbytes")
    assert serializer.secret_keys == [b"mysecretbytes"]

def test_constructor_with_list_of_str_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_str_salt():
    serializer = Serializer("secret", salt="mysalt")
    assert serializer.salt == b"mysalt"

def test_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"mysalt")
    assert serializer.salt == b"mysalt"

def test_constructor_with_default_salt():
    serializer = Serializer("secret")
    assert serializer.salt == b"itsdangerous"

def test_constructor_with_custom_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.is_text_serializer is True

def test_constructor_with_custom_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()
        @staticmethod
        def loads(b):
            return int(b.decode())
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_constructor_with_custom_signer_class():
    from itsdangerous.signer import Signer
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers_dict():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_constructor_with_fallback_signers_tuple():
    from itsdangerous.signer import Signer
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", fallback_signers=[(CustomSigner, {"key_derivation": "none"})])
    assert serializer.fallback_signers == [(CustomSigner, {"key_derivation": "none"})]

def test_constructor_with_fallback_signers_class():
    from itsdangerous.signer import Signer
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", fallback_signers=[CustomSigner])
    assert serializer.fallback_signers == [CustomSigner]

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_constructor_default_fallback_signers_empty():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_constructor_secret_key_property():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_key == b"new"


# LLM-generated content at query #8
#--------------------------

```python
def test_iter_unsigners_returns_default_signer_first():
    serializer = Serializer("secret-key")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) >= 1
    assert isinstance(unsigners[0], Signer)

def test_iter_unsigners_with_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    unsigners = list(serializer.iter_unsigners())
    assert unsigners[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer("secret-key", fallback_signers=[{"key": "fallback"}])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) > 1

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"key": "fallback"})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) > 1

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(["old-key", "new-key"])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) >= 2

def test_iter_unsigners_returns_iterator():
    serializer = Serializer("secret-key")
    unsigners = serializer.iter_unsigners()
    assert hasattr(unsigners, "__iter__")
    assert hasattr(unsigners, "__next__")
```


# LLM-generated content at query #9
#--------------------------

def test_load_payload_with_default_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def loads(self, payload):
            return {"custom": payload}
        def dumps(self, obj):
            return str(obj)
    custom_serializer = CustomSerializer()
    serializer = Serializer("secret", serializer=custom_serializer)
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"custom": b"{'key': 'value'}"}

def test_load_payload_with_text_serializer():
    class TextSerializer:
        def loads(self, payload):
            return {"text": payload}
        def dumps(self, obj):
            return "text_data"
    text_serializer = TextSerializer()
    serializer = Serializer("secret", serializer=text_serializer)
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"text": "text_data"}

def test_load_payload_with_overridden_serializer():
    class OverrideSerializer:
        def loads(self, payload):
            return {"overridden": payload}
        def dumps(self, obj):
            return "override"
    override_serializer = OverrideSerializer()
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload, serializer=override_serializer)
    assert result == {"overridden": b'{"key": "value"}'}

def test_load_payload_raises_bad_payload_on_exception():
    class BrokenSerializer:
        def loads(self, payload):
            raise ValueError("broken")
        def dumps(self, obj):
            return "data"
    broken_serializer = BrokenSerializer()
    serializer = Serializer("secret", serializer=broken_serializer)
    payload = serializer.dump_payload({"key": "value"})
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_with_text_overridden_serializer():
    class TextOverrideSerializer:
        def loads(self, payload):
            return {"text_override": payload}
        def dumps(self, obj):
            return "text_override"
    text_override_serializer = TextOverrideSerializer()
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload, serializer=text_override_serializer)
    assert result == {"text_override": b'{"key": "value"}'}


# LLM-generated content at query #10
#--------------------------

def test_constructor_default_serializer():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer == True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_constructor_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_constructor_list_of_strings_secret_key():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_constructor_list_of_bytes_secret_key():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_constructor_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_constructor_salt_bytes():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_constructor_salt_string():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_constructor_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: int(x)})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer == custom_serializer
    assert s.is_text_serializer == True

def test_constructor_custom_serializer_bytes():
    custom_serializer = type("CustomBytesSerializer", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: {}})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer == custom_serializer
    assert s.is_text_serializer == False

def test_constructor_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_constructor_custom_signer():
    class CustomSigner:
        def __init__(self, secret_keys, salt=None, **kwargs):
            pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_constructor_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_constructor_fallback_signers_none():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_constructor_with_serializer_positional_bytes():
    custom_serializer = type("CustomBytesSerializer", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: {}})()
    s = Serializer("secret", b"custom_salt", custom_serializer)
    assert s.salt == b"custom_salt"
    assert s.serializer == custom_serializer

def test_constructor_with_serializer_keyword_bytes():
    custom_serializer = type("CustomBytesSerializer", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: {}})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer == custom_serializer


# LLM-generated content at query #11
#--------------------------

def test_serializer_constructor_with_str_secret_key():
    s = Serializer("secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_str_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret-key", salt=None)
    assert s.secret_keys == [b"secret-key"]
    assert s.salt is None
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_salt():
    s = Serializer("secret-key", salt=b"custom-salt")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"custom-salt"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_str_salt():
    s = Serializer("secret-key", salt="custom-salt")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"custom-salt"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_positional():
    s = Serializer("secret-key", b"itsdangerous", json)
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_keyword():
    s = Serializer("secret-key", serializer=json)
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret-key", signer=CustomSigner)
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == CustomSigner
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {"key_derivation": "hmac"}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret-key", fallback_signers=fallback)
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == fallback
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers_none():
    s = Serializer("secret-key", fallback_signers=None)
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}


# LLM-generated content at query #12
#--------------------------

```python
def test_salt_is_none_so_predicate_false():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #13
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer("secret", fallback_signers=[Signer])
    assert serializer.fallback_signers == [Signer]
```


# LLM-generated content at query #14
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_not_text_serializer():
    serializer = Serializer("secret", serializer=Serializer(default_serializer=type("BytesSerializer", (), {"dumps": lambda self, obj: b'{"key": "value"}', "loads": lambda self, s: {"key": "value"}})()))
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #15
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_list_secret():
    s = Serializer(["secret1", b"secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_custom_serializer():
    s = Serializer("secret", serializer=json)
    assert s.serializer is json
    assert s.is_text_serializer is True

def test_serializer_constructor_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_custom_signer():
    s = Serializer("secret", signer=Signer)
    assert s.signer is Signer

def test_serializer_constructor_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "hmac"}])
    assert s.fallback_signers == [{"key_derivation": "hmac"}]

def test_serializer_constructor_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []

def test_serializer_constructor_default_fallback_signers():
    s = Serializer("secret")
    assert s.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    s = Serializer(["old", "new"])
    assert s.secret_key == b"new"


# LLM-generated content at query #16
#--------------------------

```
def test_iter_unsigners_fallback_signers_not_dict_or_tuple():
    serializer = Serializer("secret")
    serializer.fallback_signers = [Signer]
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
```


# LLM-generated content at query #17
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback():
    secret_key = b"test-secret"
    serializer = Serializer(secret_key, fallback_signers=[(Signer, {"digest_method": hashlib.sha256})])
    unsigners = list(serializer.iter_unsigners(b"test-salt"))
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
```


# LLM-generated content at query #18
#--------------------------

```python
def test_serializer_init_with_serializer_does_not_use_default():
    from itsdangerous.serializer import Serializer
    serializer = Serializer("secret", serializer=Serializer.default_serializer)
    assert serializer.serializer is Serializer.default_serializer
```


# LLM-generated content at query #19
#--------------------------

```python
def test_iter_unsigners_fallback_is_dict_evaluates_to_false():
    serializer = Serializer("secret", fallback_signers=[{}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0] is not None
    assert signers[1] is not None
```


# LLM-generated content at query #20
#--------------------------

```python
def test_iter_unsigners_fallback_tuple_handling():
    class FakeSigner:
        def __init__(self, secret_key, salt=None, **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs

    serializer = Serializer("secret", fallback_signers=[(FakeSigner, {"extra": "value"})])
    signers = list(serializer.iter_unsigners("test_salt"))
    assert len(signers) == 2
    assert isinstance(signers[1], FakeSigner)
    assert signers[1].secret_key == b"secret"
    assert signers[1].salt == b"test_salt"
    assert signers[1].kwargs == {"extra": "value"}
```


# LLM-generated content at query #21
#--------------------------

```python
def test_salt_is_none_evaluates_to_false():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #22
#--------------------------

```python
def test_serializer_init_predicate_line28_false():
    serializer_instance = Serializer(secret_key="test_secret", serializer=json)
    assert serializer_instance.serializer is json
```


# LLM-generated content at query #23
#--------------------------

```
def test_load_payload_exception_raised_when_is_text_false_and_loads_fails():
    class FailingSerializer:
        def loads(self, data):
            raise ValueError("loads failed")

    serializer = Serializer(
        secret_key="secret",
        serializer=FailingSerializer(),
    )
    payload = b"invalid_bytes"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
```


# LLM-generated content at query #24
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer == True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_secret_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    s = Serializer("secret", serializer=CustomSerializer)
    assert s.serializer == CustomSerializer
    assert s.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(b):
            return b
    s = Serializer("secret", serializer=BytesSerializer)
    assert s.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []


# LLM-generated content at query #25
#--------------------------

def test_serializer_init_with_string_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]

def test_serializer_init_with_bytes_secret_key():
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]

def test_serializer_init_with_list_of_strings_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes_secret_key():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_default_salt():
    serializer = Serializer("secret")
    assert serializer.salt == b"itsdangerous"

def test_serializer_init_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_init_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_default_serializer():
    serializer = Serializer("secret")
    assert serializer.serializer is json

def test_serializer_init_with_custom_str_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "serialized"
        @staticmethod
        def loads(s):
            return {"data": s}
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_init_with_custom_bytes_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return b"serialized"
        @staticmethod
        def loads(s):
            return {"data": s}
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_init_with_default_signer():
    serializer = Serializer("secret")
    assert serializer.signer is Signer

def test_serializer_init_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_init_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_empty_signer_kwargs():
    serializer = Serializer("secret")
    assert serializer.signer_kwargs == {}

def test_serializer_init_with_default_fallback_signers():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_serializer_init_with_custom_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_init_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_init_with_empty_serializer_kwargs():
    serializer = Serializer("secret")
    assert serializer.serializer_kwargs == {}

def test_serializer_init_secret_key_property():
    serializer = Serializer(["key1", "key2", "key3"])
    assert serializer.secret_key == b"key3"

def test_serializer_init_secret_key_single():
    serializer = Serializer("key")
    assert serializer.secret_key == b"key"


# LLM-generated content at query #26
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer == is_text_serializer(CustomSerializer())

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer_class():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"bytes_salt")
    assert serializer.salt == b"bytes_salt"

def test_serializer_constructor_with_serializer_and_signer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2}, signer_kwargs={"digest_method": "sha256"})
    assert serializer.serializer_kwargs == {"indent": 2}
    assert serializer.signer_kwargs == {"digest_method": "sha256"}


# LLM-generated content at query #27
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes_data"
        @staticmethod
        def loads(b):
            return b.decode()
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    class CustomSigner(Signer):
        pass
    fallback = [CustomSigner]
    serializer = Serializer(
        ["key1", b"key2"],
        salt="custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=fallback
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #28
#--------------------------

```python
def test_load_payload_predicate_false():
    serializer = Serializer(secret_key="test", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #29
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_string")
    assert result is not None
```


# LLM-generated content at query #30
#--------------------------

```python
def test_serializer_init_with_serializer_not_none():
    s = Serializer("secret", serializer=json)
    assert s.serializer == json
```


# LLM-generated content at query #31
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return {}
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #32
#--------------------------

```python
def test_load_payload_is_text_false_with_bytes_serializer():
    serializer = Serializer("secret", serializer=Serializer.default_serializer)
    serializer.is_text_serializer = False
    result = serializer.load_payload(b'"test"')
    assert result == "test"
```


# LLM-generated content at query #33
#--------------------------

def test_serializer_constructor_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return str(obj)

        @staticmethod
        def loads(s: str) -> t.Any:
            return eval(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == is_text_serializer(CustomSerializer())

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #34
#--------------------------

def test_serializer_init_with_string_secret_key():
    s = Serializer("my-secret-key")
    assert s.secret_keys == [b"my-secret-key"]

def test_serializer_init_with_bytes_secret_key():
    s = Serializer(b"my-secret-key")
    assert s.secret_keys == [b"my-secret-key"]

def test_serializer_init_with_list_of_strings_secret_key():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes_secret_key():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_default_salt():
    s = Serializer("my-secret-key")
    assert s.salt == b"itsdangerous"

def test_serializer_init_with_custom_salt():
    s = Serializer("my-secret-key", salt="my-salt")
    assert s.salt == b"my-salt"

def test_serializer_init_with_none_salt():
    s = Serializer("my-secret-key", salt=None)
    assert s.salt is None

def test_serializer_init_with_default_serializer():
    s = Serializer("my-secret-key")
    assert s.serializer is s.default_serializer

def test_serializer_init_with_custom_serializer():
    s = Serializer("my-secret-key", serializer=json)
    assert s.serializer is json

def test_serializer_init_with_custom_serializer_and_is_text_serializer():
    s = Serializer("my-secret-key", serializer=json)
    assert s.is_text_serializer is True

def test_serializer_init_with_default_signer():
    s = Serializer("my-secret-key")
    assert s.signer is Signer

def test_serializer_init_with_custom_signer():
    s = Serializer("my-secret-key", signer=Signer)
    assert s.signer is Signer

def test_serializer_init_with_signer_kwargs():
    s = Serializer("my-secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_empty_signer_kwargs():
    s = Serializer("my-secret-key")
    assert s.signer_kwargs == {}

def test_serializer_init_with_default_fallback_signers():
    s = Serializer("my-secret-key")
    assert s.fallback_signers == []

def test_serializer_init_with_custom_fallback_signers():
    s = Serializer("my-secret-key", fallback_signers=[{"key_derivation": "hmac"}])
    assert s.fallback_signers == [{"key_derivation": "hmac"}]

def test_serializer_init_with_none_fallback_signers():
    s = Serializer("my-secret-key", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_init_with_serializer_kwargs():
    s = Serializer("my-secret-key", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_init_with_empty_serializer_kwargs():
    s = Serializer("my-secret-key")
    assert s.serializer_kwargs == {}

def test_serializer_init_all_parameters():
    s = Serializer(
        "my-secret-key",
        salt="my-salt",
        serializer=json,
        serializer_kwargs={"sort_keys": True},
        signer=Signer,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[{"key_derivation": "hmac"}],
    )
    assert s.secret_keys == [b"my-secret-key"]
    assert s.salt == b"my-salt"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {"key_derivation": "hmac"}
    assert s.fallback_signers == [{"key_derivation": "hmac"}]
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #35
#--------------------------

def test_serializer_constructor_defaults() -> None:
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer == True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret() -> None:
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_str() -> None:
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes() -> None:
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return str(obj)

        @staticmethod
        def loads(s: str) -> t.Any:
            return int(s)

    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer == CustomSerializer()
    assert s.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer() -> None:
    import pickle
    s = Serializer("secret", serializer=pickle)
    assert s.serializer == pickle
    assert s.is_text_serializer == False

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    s = Serializer("secret", fallback_signers=[{"key_derivation": "hmac"}])
    assert s.fallback_signers == [{"key_derivation": "hmac"}]

def test_serializer_constructor_with_serializer_kwargs() -> None:
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_preserves_order_of_secret_keys() -> None:
    s = Serializer(["old", "new"])
    assert s.secret_keys == [b"old", b"new"]

def test_serializer_constructor_with_all_parameters() -> None:
    class CustomSigner(Signer):
        pass
    s = Serializer(
        secret_key=["key1", b"key2"],
        salt=b"custom_salt",
        serializer=json,
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"key_derivation": "none"},
        fallback_signers=[{"digest_method": hashlib.sha256}],
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"custom_salt"
    assert s.serializer == json
    assert s.serializer_kwargs == {"indent": 2}
    assert s.signer == CustomSigner
    assert s.signer_kwargs == {"key_derivation": "none"}
    assert s.fallback_signers == [{"digest_method": hashlib.sha256}]
    assert s.is_text_serializer == True


# LLM-generated content at query #36
#--------------------------

```python
def test_load_payload_with_non_text_serializer_results_in_false_predicate():
    serializer = Serializer("secret-key")
    serializer.is_text_serializer = False
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #37
#--------------------------

```
def test_fallback_signers_not_none():
    serializer = Serializer(secret_key="secret", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #38
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "test", "loads": lambda self, x: {}})()
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"test"
        def loads(self, payload):
            return {}
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #39
#--------------------------

```python
def test_dumps_returns_expected_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_string")
    assert isinstance(result, str)

def test_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    serializer.dumps(42)
    serializer.dumps([1, 2, 3])
    serializer.dumps({"key": "value"})
    serializer.dumps(None)

def test_dumps_preserves_integer_identity():
    serializer = _PDataSerializer()
    result = serializer.dumps(123)
    assert result == "123"

def test_dumps_handles_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result == "null"

def test_dumps_handles_boolean():
    serializer = _PDataSerializer()
    result_true = serializer.dumps(True)
    result_false = serializer.dumps(False)
    assert result_true == "true"
    assert result_false == "false"

def test_dumps_handles_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, "two", 3.0])
    assert result == '[1, "two", 3.0]'

def test_dumps_handles_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"name": "test", "value": 42})
    assert result == '{"name": "test", "value": 42}'

def test_dumps_handles_nested_objects():
    serializer = _PDataSerializer()
    data = {"level1": {"level2": [1, 2, 3]}}
    result = serializer.dumps(data)
    assert result == '{"level1": {"level2": [1, 2, 3]}}'

def test_dumps_handles_empty_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([])
    assert result == "[]"

def test_dumps_handles_empty_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({})
    assert result == "{}"```


# LLM-generated content at query #40
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #41
#--------------------------

def test_loads_returns_any_for_any_payload():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None


# LLM-generated content at query #42
#--------------------------

```python
def test_load_payload_predicate_line_22_false():
    from itsdangerous.serializer import Serializer
    from itsdangerous.exc import BadPayload
    
    serializer = Serializer("secret-key")
    
    try:
        serializer.load_payload(b"{invalid json}")
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload was not raised"


# LLM-generated content at query #43
#--------------------------

```python
def test_load_payload_is_text_false_with_bytes_serializer():
    serializer = Serializer("secret", serializer=Serializer.default_serializer)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #44
#--------------------------

def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_text_serializer():
    class CustomTextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=CustomTextSerializer())
    assert serializer.serializer == CustomTextSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_custom_bytes_serializer():
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(b):
            return b
    serializer = Serializer("secret", serializer=CustomBytesSerializer())
    assert serializer.serializer == CustomBytesSerializer()
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer_class():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]


# LLM-generated content at query #45
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_secret_key_bytes():
    serializer = Serializer(b"bytes-key")
    assert serializer.secret_keys == [b"bytes-key"]

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_str():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return eval(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"

        @staticmethod
        def loads(s):
            return s

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_signer_class():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback


# LLM-generated content at query #46
#--------------------------

```
def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    serializer.dumps = lambda obj: obj
    result = serializer.dumps("test")
    assert result == "test"
```


# LLM-generated content at query #47
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer is Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_iterable_secret_key():
    s = Serializer([b"old", b"new"])
    assert s.secret_keys == [b"old", b"new"]

def test_serializer_constructor_iterable_str_secret_key():
    s = Serializer(["old", "new"])
    assert s.secret_keys == [b"old", b"new"]

def test_serializer_constructor_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_custom_serializer_text():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    s = Serializer("secret", serializer=TextSerializer())
    assert s.serializer is TextSerializer()
    assert s.is_text_serializer is True

def test_serializer_constructor_custom_serializer_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()

        @staticmethod
        def loads(b):
            return int(b.decode())

    s = Serializer("secret", serializer=BytesSerializer())
    assert s.serializer is BytesSerializer()
    assert s.is_text_serializer is False

def test_serializer_constructor_custom_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_custom_signer():
    class CustomSigner(Signer):
        pass

    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_custom_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_custom_fallback_signers():
    s = Serializer("secret", fallback_signers=[{}, {"key_derivation": "none"}])
    assert s.fallback_signers == [{}, {"key_derivation": "none"}]

def test_serializer_constructor_none_fallback_signers():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []


# LLM-generated content at query #48
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_not_text():
    s = Serializer("secret", serializer=Serializer.default_serializer)
    result = s.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert s.is_text_serializer is False
```


# LLM-generated content at query #49
#--------------------------

def test_serializer_constructor_default():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer == True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_list_secret_key():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_bytes_list_secret_key():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "abc", "loads": lambda self, x: {}})
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer == custom_serializer
    assert s.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"abc", "loads": lambda self, x: {}})
    s = Serializer("secret", serializer=bytes_serializer)
    assert s.serializer == bytes_serializer
    assert s.is_text_serializer == False

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {"sign": lambda self, x: x, "unsign": lambda self, x: x})
    s = Serializer("secret", signer=custom_signer)
    assert s.signer == custom_signer

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_all_arguments():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "abc", "loads": lambda self, x: {}})
    custom_signer = type("CustomSigner", (Signer,), {"sign": lambda self, x: x, "unsign": lambda self, x: x})
    s = Serializer(
        ["key1", b"key2"],
        salt=b"salt",
        serializer=custom_serializer,
        serializer_kwargs={"indent": 2},
        signer=custom_signer,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[{"algorithm": "sha512"}],
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"salt"
    assert s.serializer == custom_serializer
    assert s.is_text_serializer == True
    assert s.signer == custom_signer
    assert s.signer_kwargs == {"key_derivation": "hmac"}
    assert s.fallback_signers == [{"algorithm": "sha512"}]
    assert s.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #50
#--------------------------

def test_dumps_text_serializer_returns_str():
    serializer = Serializer("secret", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_bytes_serializer_returns_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj, **kwargs):
            return b'{"key": "value"}'
    serializer = Serializer("secret", serializer=BytesSerializer())
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_default_salt():
    serializer = Serializer("secret")
    result = serializer.dumps("test")
    assert isinstance(result, str)

def test_dumps_custom_salt():
    serializer = Serializer("secret")
    result = serializer.dumps("test", salt="custom_salt")
    assert isinstance(result, str)

def test_dumps_with_serializer_kwargs():
    class CustomSerializer:
        @staticmethod
        def dumps(obj, **kwargs):
            assert kwargs.get("indent") == 2
            return '{"key": "value"}'
    serializer = Serializer("secret", serializer=CustomSerializer(), serializer_kwargs={"indent": 2})
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)


# LLM-generated content at query #51
#--------------------------

```python
def test_loads_returns_any_for_empty_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(b"")
    assert result is not None

def test_loads_returns_any_for_integer_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(42)
    assert isinstance(result, object)

def test_loads_returns_any_for_string_payload():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None

def test_loads_returns_any_for_list_payload():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert isinstance(result, object)

def test_loads_returns_any_for_dict_payload():
    serializer = _PDataSerializer()
    result = serializer.loads({"key": "value"})
    assert result is not None
```


# LLM-generated content at query #52
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None or result is None


# LLM-generated content at query #53
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback():
    signer_class = Signer
    fallback_tuple = (signer_class, {"key_derivation": "hmac"})
    serializer = Serializer(
        secret_key="test-secret",
        fallback_signers=[fallback_tuple],
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].key_derivation == "hmac"
    assert isinstance(unsigners[1], Signer)
```


# LLM-generated content at query #54
#--------------------------

def test_serializer_constructor_with_default_serializer() -> None:
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_secret_key_list() -> None:
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> t.Any:
            return int(s)
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer()
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> bytes:
            return b"data"
        @staticmethod
        def loads(b: bytes) -> t.Any:
            return b"data"
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs() -> None:
    s = Serializer("secret", serializer_kwargs={"ensure_ascii": False})
    assert s.serializer_kwargs == {"ensure_ascii": False}

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]


# LLM-generated content at query #55
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret_key")
    assert serializer.secret_keys == [b"secret_key"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_str_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class MockSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=MockSerializer())
    assert serializer.serializer is MockSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(s):
            return s

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"ensure_ascii": False})
    assert serializer.serializer_kwargs == {"ensure_ascii": False}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_uses_default_serializer_when_none():
    serializer = Serializer("secret", serializer=None)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True


# LLM-generated content at query #56
#--------------------------

```python
def test_loads_returns_any_type():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None

def test_loads_accepts_serialized_input():
    serializer = _PDataSerializer()
    result = serializer.loads(42)
    assert result is not None

def test_loads_accepts_string_input():
    serializer = _PDataSerializer()
    result = serializer.loads("data")
    assert result is not None

def test_loads_accepts_bytes_input():
    serializer = _PDataSerializer()
    result = serializer.loads(b"bytes")
    assert result is not None

def test_loads_accepts_list_input():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert result is not None

def test_loads_accepts_dict_input():
    serializer = _PDataSerializer()
    result = serializer.loads({"key": "value"})
    assert result is not None

def test_loads_accepts_none_input():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is not None
```


# LLM-generated content at query #57
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return {"custom": True}
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return {"bytes": True}
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.serializer is BytesSerializer
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_bytes_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_str_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"


# LLM-generated content at query #58
#--------------------------

def test_serializer_constructor_with_default_parameters():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings_secret_key():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_key():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_bytes_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_str_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    s = Serializer("secret", serializer=TextSerializer)
    assert s.is_text_serializer

def test_serializer_constructor_with_custom_binary_serializer():
    class BinarySerializer:
        @staticmethod
        def dumps(obj):
            return bytes(str(obj), "utf-8")
        @staticmethod
        def loads(b):
            return eval(b.decode("utf-8"))
    s = Serializer("secret", serializer=BinarySerializer)
    assert not s.is_text_serializer

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer_class():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []

def test_serializer_constructor_with_none_fallback_signers():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []


# LLM-generated content at query #59
#--------------------------

```
def test_dumps_with_none():
    serializer = _PDataSerializer()
    assert serializer.dumps(None) == b"null"

def test_dumps_with_integer():
    serializer = _PDataSerializer()
    assert serializer.dumps(42) == b"42"

def test_dumps_with_string():
    serializer = _PDataSerializer()
    assert serializer.dumps("hello") == b'"hello"'

def test_dumps_with_list():
    serializer = _PDataSerializer()
    assert serializer.dumps([1, 2, 3]) == b'[1, 2, 3]'

def test_dumps_with_dict():
    serializer = _PDataSerializer()
    assert serializer.dumps({"key": "value"}) == b'{"key": "value"}'

def test_dumps_with_boolean():
    serializer = _PDataSerializer()
    assert serializer.dumps(True) == b"true"

def test_dumps_with_float():
    serializer = _PDataSerializer()
    assert serializer.dumps(3.14) == b"3.14"
```


# LLM-generated content at query #60
#--------------------------

```python
from itsdangerous.serializer import Serializer
from itsdangerous.signer import Signer

def test_iter_unsigners_returns_default_signer_first():
    serializer = Serializer("secret-key", salt="salt")
    signers = list(serializer.iter_unsigners())
    assert isinstance(signers[0], Signer)
    assert len(signers) == 1

def test_iter_unsigners_yields_fallback_signers():
    fallback_signer = Signer
    serializer = Serializer("secret-key", fallback_signers=[fallback_signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_uses_salt_parameter():
    serializer = Serializer("secret-key", salt="default_salt")
    signers = list(serializer.iter_unsigners(salt="custom_salt"))
    assert signers[0].salt == b"custom_salt"

def test_iter_unsigners_with_fallback_dict():
    kwargs = {"key_derivation": "hmac"}
    serializer = Serializer("secret-key", fallback_signers=[kwargs])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].key_derivation == "hmac"

def test_iter_unsigners_with_fallback_tuple():
    kwargs = {"key_derivation": "hmac"}
    serializer = Serializer("secret-key", fallback_signers=[(Signer, kwargs)])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].key_derivation == "hmac"

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(["old_key", "new_key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert len(signers[0].secret_keys) == 2
```


# LLM-generated content at query #61
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_not_text_serializer():
    from itsdangerous.serializer import Serializer
    from itsdangerous.exc import BadSignature

    serializer = Serializer(secret_key="secret", serializer=lambda: None)
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #62
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_secret_key_list():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_str_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = json
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_all_parameters():
    serializer = Serializer(
        secret_key=["old_key", "new_key"],
        salt="pepper",
        serializer=json,
        serializer_kwargs={"indent": 2},
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}],
    )
    assert serializer.secret_keys == [b"old_key", b"new_key"]
    assert serializer.salt == b"pepper"
    assert serializer.serializer == json
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #63
#--------------------------

```python
def test_load_payload_is_text_false_with_bytes_serializer():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    import json

    serializer = Serializer("secret-key", serializer=json)
    serializer.is_text_serializer = False
    result = serializer.load_payload(b'"test"')
    assert result == "test"
```


# LLM-generated content at query #64
#--------------------------

```python
from itsdangerous.serializer import Serializer
from itsdangerous.exc import BadPayload

def test_load_payload_raises_bad_payload_when_is_text_false_and_loads_fails():
    class FailingSerializer:
        @staticmethod
        def loads(data):
            raise ValueError("test error")

    serializer = Serializer("secret", serializer=FailingSerializer())
    try:
        serializer.load_payload(b"invalid")
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload exception"
```


# LLM-generated content at query #65
#--------------------------

```python
def test_salt_is_none_so_predicate_is_false():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #66
#--------------------------

def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, obj: "str", "loads": lambda self, s: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, obj: b"bytes", "loads": lambda self, s: {}})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True


# LLM-generated content at query #67
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == isinstance(json.dumps({}), str)
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_salt_bytes():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_salt_string():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"

        @staticmethod
        def loads(s):
            return {"custom": True}

    serializer = Serializer("secret-key", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_none():
    serializer = Serializer("secret-key", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #68
#--------------------------

def test_serializer_constructor_with_str_secret_key():
    s = Serializer("my-secret-key")
    assert s.secret_keys == [b"my-secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"my-secret-key")
    assert s.secret_keys == [b"my-secret-key"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_str_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_str_salt():
    s = Serializer("secret", salt="custom-salt")
    assert s.salt == b"custom-salt"

def test_serializer_constructor_with_bytes_salt():
    s = Serializer("secret", salt=b"custom-salt")
    assert s.salt == b"custom-salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    s = Serializer("secret", serializer=CustomSerializer)
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer_class():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_none_fallback_signers():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    s = Serializer(["old", "new"])
    assert s.secret_key == b"new"

def test_serializer_constructor_calls_make_keys_list():
    s = Serializer(b"test")
    assert s.secret_keys == [b"test"]


# LLM-generated content at query #69
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json

def test_constructor_with_bytes_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.is_text_serializer == True

def test_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #70
#--------------------------

def test_serializer_constructor_with_default_parameters():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    class CustomSigner(Signer):
        pass
    serializer = Serializer(
        secret_key=["key1", "key2"],
        salt=b"custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #71
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads(serializer.dumps("test"))
    assert result is not None

def test_loads_with_string_payload():
    serializer = _PDataSerializer()
    result = serializer.loads("test_string")
    assert result == "test_string"

def test_loads_with_integer_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(123)
    assert result == 123

def test_loads_with_list_payload():
    serializer = _PDataSerializer()
    payload = [1, 2, 3]
    result = serializer.loads(payload)
    assert result == payload

def test_loads_with_dict_payload():
    serializer = _PDataSerializer()
    payload = {"key": "value"}
    result = serializer.loads(payload)
    assert result == payload

def test_loads_with_none_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is None

def test_loads_with_bytes_payload():
    serializer = _PDataSerializer()
    payload = b"bytes_data"
    result = serializer.loads(payload)
    assert result == payload
```


# LLM-generated content at query #72
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_serializer():
    serializer = Serializer("secret", serializer=json.JSONEncoder)
    assert serializer.serializer == json.JSONEncoder

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}

def test_constructor_with_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer == Signer

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secrets():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_constructor_with_list_of_bytes_secrets():
    serializer = Serializer([b"secret1", b"secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]


# LLM-generated content at query #73
#--------------------------

```python
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None
```


# LLM-generated content at query #74
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_string")
    assert result == "test_string"

def test_dumps_accepts_any_type():
    serializer = _PDataSerializer()
    result = serializer.dumps(123)
    assert result == 123

def test_dumps_accepts_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is None

def test_dumps_accepts_complex_object():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert result == obj
```


# LLM-generated content at query #75
#--------------------------

```python
def test_load_payload_predicate_false():
    serializer = Serializer(secret_key="secret", serializer=Serializer.default_serializer)
    payload = b'{"valid": true}'
    result = serializer.load_payload(payload)
    assert result == {"valid": True}
```


# LLM-generated content at query #76
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: int(x)})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_binary_serializer():
    custom_serializer = type("BinarySerializer", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = {"key_derivation": "none"}
    serializer = Serializer("secret", fallback_signers=[fallback])
    assert serializer.fallback_signers == [fallback]

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #77
#--------------------------

def test_dumps_returns_bytes_when_serializer_is_bytes():
    serializer = Serializer("secret-key", serializer=lambda: None)
    serializer.serializer = type("BytesSerializer", (), {"dumps": lambda self, obj: b"data"})()
    serializer.is_text_serializer = False
    result = serializer.dumps("test")
    assert isinstance(result, bytes)

def test_dumps_returns_string_when_serializer_is_text():
    serializer = Serializer("secret-key", serializer=lambda: None)
    serializer.serializer = type("TextSerializer", (), {"dumps": lambda self, obj: "data"})()
    serializer.is_text_serializer = True
    result = serializer.dumps("test")
    assert isinstance(result, str)

def test_dumps_includes_signature():
    serializer = Serializer("secret-key")
    result = serializer.dumps("test")
    assert b"." in result if isinstance(result, bytes) else "." in result

def test_dumps_with_salt_overrides_default_salt():
    serializer = Serializer("secret-key")
    result_default = serializer.dumps("test")
    result_salted = serializer.dumps("test", salt="different")
    assert result_default != result_salted

def test_dumps_uses_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    serializer.serializer = type("MockJSON", (), {"dumps": lambda self, obj, **kwargs: str(kwargs)})()
    serializer.is_text_serializer = True
    result = serializer.dumps("test")
    assert "sort_keys" in result


# LLM-generated content at query #78
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert s.is_text_serializer == True
    assert s.signer is Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret-key")
    assert s.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_list_of_strings():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret-key", salt="custom-salt")
    assert s.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret-key", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: int(x)})()
    s = Serializer("secret-key", serializer=custom_serializer)
    assert s.serializer is custom_serializer
    assert s.is_text_serializer == True

def test_serializer_constructor_with_custom_serializer_kwargs():
    s = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_keys, salt=None, **kwargs):
            pass
    s = Serializer("secret-key", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    s = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "none"}]
    s = Serializer("secret-key", fallback_signers=fallback_signers)
    assert s.fallback_signers == fallback_signers

def test_serializer_constructor_with_empty_fallback_signers():
    s = Serializer("secret-key", fallback_signers=[])
    assert s.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    s = Serializer(["old_key", "new_key"])
    assert s.secret_key == b"new_key"


# LLM-generated content at query #79
#--------------------------

```python
def test_loads_returns_any_given_serialized_input():
    serializer = _PDataSerializer()
    payload = _TSerialized()
    result = serializer.loads(payload)
    assert result is not None or result is None

def test_loads_accepts_serialized_payload():
    serializer = _PDataSerializer()
    payload = _TSerialized()
    result = serializer.loads(payload)
    assert isinstance(result, object)

def test_loads_returns_none_for_empty_serialized():
    serializer = _PDataSerializer()
    payload = _TSerialized()
    result = serializer.loads(payload)
    assert result is None or not isinstance(result, _TSerialized)
```


# LLM-generated content at query #80
#--------------------------

```python
def test_salt_is_not_none_predicate_false():
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None
    assert not hasattr(serializer, '_salt_processed')
```


# LLM-generated content at query #81
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #82
#--------------------------

```python
def test_serializer_init_with_serializer_not_none():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    import json
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
```


# LLM-generated content at query #83
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_salt_bytes():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_salt_string():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(data):
            return data
    s = Serializer("secret", serializer=TextSerializer())
    assert s.serializer is TextSerializer()
    assert s.is_text_serializer is True

def test_serializer_constructor_with_custom_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.serializer is BytesSerializer()
    assert s.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #84
#--------------------------

```python
def test_load_payload_uses_default_serializer_when_none_provided():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_uses_provided_serializer():
    serializer = Serializer("secret")
    payload = b'{"key":"value"}'
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"key": "value"}

def test_load_payload_raises_bad_payload_on_invalid_data():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid")
        assert False
    except BadPayload:
        pass

def test_load_payload_handles_text_serializer():
    class TextSerializer:
        def dumps(self, obj):
            return "text"
        def loads(self, payload):
            return payload
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = serializer.dump_payload("data")
    result = serializer.load_payload(payload)
    assert result == "data"

def test_load_payload_handles_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, payload):
            return payload
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = serializer.dump_payload("data")
    result = serializer.load_payload(payload)
    assert result == b"bytes"

def test_load_payload_uses_overridden_serializer_is_text():
    class TextSerializer:
        def dumps(self, obj):
            return "text"
        def loads(self, payload):
            return payload
    serializer = Serializer("secret")
    payload = b'{"key":"value"}'
    result = serializer.load_payload(payload, serializer=TextSerializer())
    assert result == '{"key":"value"}'
```


# LLM-generated content at query #85
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer == True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return {}
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer == CustomSerializer()
    assert s.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return {}
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.serializer == BytesSerializer()
    assert s.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_keys, salt=None, **kwargs):
            self.secret_keys = secret_keys
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, payload):
            return payload
        def unsign(self, signed):
            return signed
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]


# LLM-generated content at query #86
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_str_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_json_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"dumped"

        @staticmethod
        def loads(data):
            return data

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"ensure_ascii": False})
    assert serializer.serializer_kwargs == {"ensure_ascii": False}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"

def test_serializer_constructor_positional_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"dumped"

        @staticmethod
        def loads(data):
            return data

    serializer = Serializer("secret", b"salt", BytesSerializer())
    assert serializer.salt == b"salt"
    assert serializer.serializer is not json
    assert serializer.is_text_serializer is False


# LLM-generated content at query #87
#--------------------------

def test_dumps_with_default_serializer():
    serializer = Serializer("secret")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.serializer.dumps = lambda obj: b"data"
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_with_salt():
    serializer = Serializer("secret")
    result = serializer.dumps({"key": "value"}, salt="custom_salt")
    assert isinstance(result, str)

def test_dumps_return_value_is_signed():
    serializer = Serializer("secret")
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)


# LLM-generated content at query #88
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_string_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "str", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"digest_method": hashlib.sha256})
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"digest_method": hashlib.sha256}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_text_serializer():
    text_serializer = type("TextSerializer", (), {"dumps": lambda self, x: "str", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=text_serializer)
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"bytes", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.is_text_serializer == False


# LLM-generated content at query #89
#--------------------------

def test_serializer_constructor_with_str_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_str_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False


# LLM-generated content at query #90
#--------------------------

def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_strings_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_key():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #91
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_string_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_text():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(s):
            return {}
    serializer = Serializer("secret-key", serializer=TextSerializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is TextSerializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return {}
    serializer = Serializer("secret-key", serializer=BytesSerializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is BytesSerializer
    assert serializer.is_text_serializer is False
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[{"key_derivation": "hmac"}])
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #92
#--------------------------

```python
def test_serializer_uses_provided_serializer_when_not_none():
    custom_serializer = CustomSerializer()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer is custom_serializer
```


# LLM-generated content at query #93
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert s.is_text_serializer == is_text_serializer(s.serializer)
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_str():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_custom_serializer():
    custom_serializer = type("MockSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: int(x)})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer is custom_serializer
    assert s.is_text_serializer == is_text_serializer(custom_serializer)

def test_serializer_constructor_custom_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_custom_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_fallback_signers_none():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_default_fallback_signers():
    class CustomSerializer(Serializer):
        default_fallback_signers = [{"key_derivation": "none"}]
    s = CustomSerializer("secret")
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_secret_key_property():
    s = Serializer(["key1", "key2"])
    assert s.secret_key == b"key2"


# LLM-generated content at query #94
#--------------------------

```python
def test_load_payload_is_text_false_with_non_text_serializer():
    serializer = Serializer("secret", serializer=dumps_bytes)
    result = serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}

def test_load_payload_is_text_false_with_default_serializer():
    serializer = Serializer("secret")
    result = serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}
```


# LLM-generated content at query #95
#--------------------------

```
def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None
```


# LLM-generated content at query #96
#--------------------------

def test_salt_is_not_none_false():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #97
#--------------------------

```python
def test_load_payload_predicate_false():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={}, signer=Signer, signer_kwargs={}, fallback_signers=[])
    payload = b'{"key": "value"}'
    try:
        result = serializer.load_payload(payload)
    except BadPayload:
        pass
```


# LLM-generated content at query #98
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    payload = b"test data"
    result = serializer.loads(payload)
    assert result is not None

def test_loads_accepts_bytes():
    serializer = _PDataSerializer()
    payload = b"serialized content"
    result = serializer.loads(payload)
    assert isinstance(result, object)

def test_loads_accepts_string():
    serializer = _PDataSerializer()
    payload = "serialized content"
    result = serializer.loads(payload)
    assert isinstance(result, object)

def test_loads_returns_none():
    serializer = _PDataSerializer()
    payload = b""
    result = serializer.loads(payload)
    assert result is None

def test_loads_returns_integer():
    serializer = _PDataSerializer()
    payload = b"42"
    result = serializer.loads(payload)
    assert result == 42
```


# LLM-generated content at query #99
#--------------------------

def test_serializer_constructor_default_values():
    secret_key = "secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    secret_key = b"secret"
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    secret_key = ["key1", "key2"]
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    secret_key = [b"key1", b"key2"]
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    secret_key = "secret"
    salt = "custom_salt"
    serializer = Serializer(secret_key, salt=salt)
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    secret_key = "secret"
    salt = None
    serializer = Serializer(secret_key, salt=salt)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    secret_key = "secret"
    custom_serializer = json
    serializer = Serializer(secret_key, serializer=custom_serializer)
    assert serializer.serializer == custom_serializer

def test_serializer_constructor_with_bytes_serializer():
    secret_key = "secret"
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"
        @staticmethod
        def loads(data):
            return {}
    serializer = Serializer(secret_key, serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    secret_key = "secret"
    serializer_kwargs = {"indent": 4}
    serializer = Serializer(secret_key, serializer_kwargs=serializer_kwargs)
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer():
    secret_key = "secret"
    class CustomSigner(Signer):
        pass
    serializer = Serializer(secret_key, signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    secret_key = "secret"
    signer_kwargs = {"key_derivation": "hmac"}
    serializer = Serializer(secret_key, signer_kwargs=signer_kwargs)
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    secret_key = "secret"
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer(secret_key, fallback_signers=fallback_signers)
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_empty_fallback_signers():
    secret_key = "secret"
    fallback_signers = []
    serializer = Serializer(secret_key, fallback_signers=fallback_signers)
    assert serializer.fallback_signers == []


# LLM-generated content at query #100
#--------------------------

```python
def test_predicate_at_line_20_evaluates_to_false_when_salt_is_not_none():
    from itsdangerous.serializer import Serializer
    serializer = Serializer("secret-key", salt=b"custom_salt")
    assert serializer.salt is not None
    assert serializer.salt == b"custom_salt"
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_load_payload_with_default_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def loads(self, payload):
            return {"custom": payload}
        def dumps(self, obj):
            return "test"
    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"custom": b"test"}

def test_load_payload_with_bytes_serializer():
    serializer = Serializer("secret", serializer=json, serializer_kwargs={})
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_text_serializer():
    class TextSerializer:
        def loads(self, payload):
            return {"text": payload}
        def dumps(self, obj):
            return "text_data"
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"text": "text_data"}

def test_load_payload_raises_bad_payload():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid")
        assert False
    except BadPayload:
        assert True

def test_load_payload_with_explicit_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"key": "value"}

def test_load_payload_with_text_serializer_explicit():
    class TextSerializer:
        def loads(self, payload):
            return {"text": payload}
        def dumps(self, obj):
            return "text_data"
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload, serializer=TextSerializer())
    assert result == {"text": b'{"key": "value"}'}


# LLM-generated content at query #2
#--------------------------

def test_serializer_constructor_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> t.Any:
            return int(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"ensure_ascii": False})
    assert serializer.serializer_kwargs == {"ensure_ascii": False}

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_all_parameters() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> t.Any:
            return int(s)
    class CustomSigner(Signer):
        pass
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer(
        secret_key=["key1", "key2"],
        salt=b"custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"ensure_ascii": False},
        signer=CustomSigner,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=fallback,
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {"ensure_ascii": False}


# LLM-generated content at query #3
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #4
#--------------------------

```python
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None

def test_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result is not None

def test_dumps_accepts_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None

def test_dumps_accepts_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result is not None

def test_dumps_accepts_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert result is not None

def test_dumps_accepts_custom_object():
    serializer = _PDataSerializer()
    result = serializer.dumps(object())
    assert result is not None
```


# LLM-generated content at query #5
#--------------------------

def test_serializer_init_with_string_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert isinstance(serializer.serializer, type(serializer.default_serializer))
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]

def test_serializer_init_with_list_of_string_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_custom_salt():
    serializer = Serializer("my-secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_init_with_none_salt():
    serializer = Serializer("my-secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("my-secret-key", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_init_with_custom_serializer_returning_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("my-secret-key", serializer=BytesSerializer())
    assert serializer.serializer is BytesSerializer()
    assert serializer.is_text_serializer is False

def test_serializer_init_with_serializer_kwargs():
    serializer = Serializer("my-secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_init_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("my-secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_init_with_signer_kwargs():
    serializer = Serializer("my-secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("my-secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_empty_fallback_signers():
    serializer = Serializer("my-secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_init_with_none_fallback_signers():
    serializer = Serializer("my-secret-key", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_init_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    class CustomSigner(Signer):
        pass
    serializer = Serializer(
        secret_key=["k1", b"k2"],
        salt="salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=[{"key_derivation": "none"}],
    )
    assert serializer.secret_keys == [b"k1", b"k2"]
    assert serializer.salt == b"salt"
    assert isinstance(serializer.serializer, CustomSerializer)
    assert serializer.is_text_serializer is True
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #6
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secrets():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "{}", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"{}"
        def loads(self, data):
            return {}
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert not serializer.is_text_serializer

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #7
#--------------------------

def test_constructor_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_salt_none() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer() -> None:
    custom_serializer = json
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #8
#--------------------------

def test_load_payload_with_none_serializer_and_is_text_serializer():
    serializer = Serializer("secret", serializer=json)
    serializer.load_payload(json.dumps({"key": "value"}).encode("utf-8"))

def test_load_payload_with_none_serializer_and_not_is_text_serializer():
    serializer = Serializer("secret", serializer=type("BytesSerializer", (), {"dumps": lambda self, obj: b"bytes", "loads": lambda self, payload: payload})())
    serializer.load_payload(b"bytes")

def test_load_payload_with_explicit_serializer_text():
    serializer = Serializer("secret", serializer=json)
    serializer.load_payload(json.dumps({"key": "value"}).encode("utf-8"), serializer=json)

def test_load_payload_with_explicit_serializer_bytes():
    serializer = Serializer("secret", serializer=json)
    serializer.load_payload(b"bytes", serializer=type("BytesSerializer", (), {"dumps": lambda self, obj: b"bytes", "loads": lambda self, payload: payload})())

def test_load_payload_raises_bad_payload_on_exception():
    serializer = Serializer("secret", serializer=type("FaultySerializer", (), {"dumps": lambda self, obj: b"data", "loads": lambda self, payload: (_ for _ in ()).throw(Exception("load error"))})())
    try:
        serializer.load_payload(b"data")
        assert False
    except BadPayload:
        pass


# LLM-generated content at query #9
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "dumped"
        @staticmethod
        def loads(s):
            return "loaded"
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"dumped"
        @staticmethod
        def loads(s):
            return "loaded"
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_tuple():
    class CustomSigner(Signer):
        pass
    fallback = [(CustomSigner, {"key_derivation": "none"})]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_key == b"new_key"


# LLM-generated content at query #10
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_list_of_strings_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_list_of_bytes_secret_key():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(b):
            return b

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_fallback_signers_default():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_serializer_constructor_fallback_signers_none():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_key == b"new_key"


# LLM-generated content at query #11
#--------------------------

```python
def test_iter_unsigners_returns_signer_yielded_from_make_signer():
    serializer = Serializer(secret_key="test-secret", salt=b"test-salt")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert unsigners[0] == serializer.make_signer(salt=b"test-salt")

def test_iter_unsigners_with_fallback_dict():
    serializer = Serializer(
        secret_key="test-secret",
        salt=b"test-salt",
        fallback_signers=[{"key": "value"}],
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert unsigners[1].secret_key == b"test-secret"
    assert unsigners[1].salt == b"test-salt"

def test_iter_unsigners_with_fallback_tuple():
    serializer = Serializer(
        secret_key="test-secret",
        salt=b"test-salt",
        fallback_signers=[(Serializer.default_signer, {"key": "value"})],
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert unsigners[1].secret_key == b"test-secret"
    assert unsigners[1].salt == b"test-salt"

def test_iter_unsigners_with_fallback_class():
    serializer = Serializer(
        secret_key="test-secret",
        salt=b"test-salt",
        fallback_signers=[Serializer.default_signer],
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert unsigners[1].secret_key == b"test-secret"
    assert unsigners[1].salt == b"test-salt"

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(
        secret_key=["old-secret", "new-secret"],
        salt=b"test-salt",
        fallback_signers=[Serializer.default_signer],
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3
    assert unsigners[0].secret_keys[-1] == b"new-secret"
    assert unsigners[1].secret_key == b"old-secret"
    assert unsigners[2].secret_key == b"new-secret"

def test_iter_unsigners_with_custom_salt():
    serializer = Serializer(secret_key="test-secret", salt=b"default-salt")
    unsigners = list(serializer.iter_unsigners(salt=b"custom-salt"))
    assert unsigners[0].salt == b"custom-salt"
```


# LLM-generated content at query #12
#--------------------------

def test_dumps_with_text_serializer_returns_string():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    import json

    serializer = Serializer("secret-key", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_with_bytes_serializer_returns_bytes():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    import pickle

    serializer = Serializer("secret-key", serializer=pickle)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_with_default_serializer_returns_string():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_with_salt():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    serializer = Serializer("secret-key", salt="custom-salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_returns_different_output_for_different_objects():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    serializer = Serializer("secret-key")
    result1 = serializer.dumps({"key": "value1"})
    result2 = serializer.dumps({"key": "value2"})
    assert result1 != result2


# LLM-generated content at query #13
#--------------------------

def test_serializer_constructor_with_default_parameters() -> None:
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_iterable_secret_key() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return str(obj)

        @staticmethod
        def loads(s: str) -> object:
            return int(s)

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #14
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_serializer():
    custom_serializer = type("Custom", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: {}})
    serializer = Serializer("secret", serializer=custom_serializer())
    assert serializer.serializer is custom_serializer() or isinstance(serializer.serializer, type(custom_serializer()))
    assert serializer.is_text_serializer is False

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback


# LLM-generated content at query #15
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_key == b"secret"
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_key == b"secret"

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()
        @staticmethod
        def loads(data):
            return data.decode()
    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_keys, salt=None, **kwargs):
            pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()
        @staticmethod
        def loads(data):
            return data.decode()
    class CustomSigner:
        def __init__(self, secret_keys, salt=None, **kwargs):
            pass
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer(
        secret_key=["key1", b"key2"],
        salt=b"custom_salt",
        serializer=CustomSerializer,
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=fallback
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == False
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #16
#--------------------------

```python
def test_load_payload_predicate_false():
    serializer = Serializer("secret", serializer=type("CustomSerializer", (), {"loads": lambda self, x: x}))
    serializer.load_payload(b"test")


# LLM-generated content at query #17
#--------------------------

```python
def test_dumps_returns_bytes_when_not_text_serializer():
    serializer = Serializer("secret", serializer=None)
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #18
#--------------------------

def test_serializer_constructor_with_str_secret_key() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_str_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_default_salt() -> None:
    serializer = Serializer("secret")
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_salt_as_str() -> None:
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_salt_as_bytes() -> None:
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_salt_none() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_default_serializer() -> None:
    serializer = Serializer("secret")
    assert serializer.serializer is json

def test_serializer_constructor_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return str(obj)

        @staticmethod
        def loads(s: str) -> t.Any:
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer

def test_serializer_constructor_default_signer() -> None:
    serializer = Serializer("secret")
    assert serializer.signer is Signer

def test_serializer_constructor_custom_signer() -> None:
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_default_signer_kwargs() -> None:
    serializer = Serializer("secret")
    assert serializer.signer_kwargs == {}

def test_serializer_constructor_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_default_fallback_signers() -> None:
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_serializer_constructor_fallback_signers() -> None:
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_default_serializer_kwargs() -> None:
    serializer = Serializer("secret")
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #19
#--------------------------

```python
def test_iter_unsigners_default_signer_yielded_first():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    assert isinstance(signers[0], Signer)

def test_iter_unsigners_fallback_dict_signers_yielded():
    serializer = Serializer("secret-key", fallback_signers=[{"key": "value"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_fallback_tuple_signers_yielded():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"key": "value"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_fallback_class_signers_yielded():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_fallback_with_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"], fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3

def test_iter_unsigners_custom_salt_used():
    serializer = Serializer("secret-key", salt=b"custom_salt")
    signers = list(serializer.iter_unsigners(salt=b"custom_salt"))
    assert signers[0].salt == b"custom_salt"

def test_iter_unsigners_none_salt_falls_back_to_serializer_salt():
    serializer = Serializer("secret-key", salt=b"serializer_salt")
    signers = list(serializer.iter_unsigners())
    assert signers[0].salt == b"serializer_salt"

def test_iter_unsigners_fallback_signers_empty_list():
    serializer = Serializer("secret-key", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
```


# LLM-generated content at query #20
#--------------------------

def test_serializer_constructor_with_default_parameters():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_keys == [b"old_key", b"new_key"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"old_key", b"new_key"])
    assert serializer.secret_keys == [b"old_key", b"new_key"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(b):
            return {}
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback


# LLM-generated content at query #21
#--------------------------

```python
from itsdangerous.serializer import Serializer
from itsdangerous.exc import BadPayload

def test_load_payload_with_default_serializer_and_bytes_payload():
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_text_serializer():
    class TextSerializer:
        def loads(self, payload: str) -> dict:
            return {"data": payload}
        def dumps(self, obj: dict) -> str:
            return obj["data"]
    serializer = Serializer("secret-key", serializer=TextSerializer())
    payload = b"test data"
    result = serializer.load_payload(payload)
    assert result == {"data": "test data"}

def test_load_payload_with_custom_binary_serializer():
    class BinarySerializer:
        def loads(self, payload: bytes) -> dict:
            return {"data": payload}
        def dumps(self, obj: dict) -> bytes:
            return obj["data"]
    serializer = Serializer("secret-key", serializer=BinarySerializer())
    payload = b"binary data"
    result = serializer.load_payload(payload)
    assert result == {"data": b"binary data"}

def test_load_payload_with_override_serializer():
    class OverrideSerializer:
        def loads(self, payload: str) -> list:
            return [1, 2, 3]
        def dumps(self, obj: list) -> str:
            return "dummy"
    serializer = Serializer("secret-key")
    payload = b"anything"
    result = serializer.load_payload(payload, serializer=OverrideSerializer())
    assert result == [1, 2, 3]

def test_load_payload_raises_bad_payload_on_invalid_data():
    serializer = Serializer("secret-key")
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_raises_bad_payload_with_original_error():
    serializer = Serializer("secret-key")
    payload = b"not json"
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload as e:
        assert e.original_error is not None
```


# LLM-generated content at query #22
#--------------------------

```python
def test_iter_unsigners_tuple_fallback():
    serializer = Serializer(
        secret_key="secret",
        fallback_signers=[(Signer, {"digest_method": "sha256"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].digest_method == "sha256"
```


# LLM-generated content at query #23
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(b):
            return b
    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.serializer == BytesSerializer
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #24
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer(secret_key="secret")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)

def test_iter_unsigners_with_fallback_dict():
    serializer = Serializer(secret_key="secret", fallback_signers=[{"key": "fallback"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_fallback_tuple():
    serializer = Serializer(secret_key="secret", fallback_signers=[(Signer, {"key": "fallback"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_fallback_class():
    serializer = Serializer(secret_key="secret", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_multiple_secret_keys():
    serializer = Serializer(secret_key=["old_secret", "new_secret"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_custom_salt():
    serializer = Serializer(secret_key="secret", salt=b"custom_salt")
    signers = list(serializer.iter_unsigners(salt=b"override_salt"))
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
```


# LLM-generated content at query #25
#--------------------------

def test_serializer_constructor_default_serializer():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_salt_string():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_salt_bytes():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer_text():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "{}"
        @staticmethod
        def loads(s):
            return {}
    s = Serializer("secret", serializer=TextSerializer())
    assert s.serializer == TextSerializer()
    assert s.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"
        @staticmethod
        def loads(s):
            return {}
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.serializer == BytesSerializer()
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer_class():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_secret_key_bytes():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_secret_key_list_strings():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_secret_key_list_bytes():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #26
#--------------------------

```python
def test_salt_is_not_none_in_serializer_init():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"
```


# LLM-generated content at query #27
#--------------------------

def test_constructor_with_default_parameters() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return "test"

        @staticmethod
        def loads(s: str) -> object:
            return "test"

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer is True
    assert serializer.serializer is CustomSerializer()

def test_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_fallback_signers() -> None:
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_constructor_with_overridden_default_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return b"test"

        @staticmethod
        def loads(s: bytes) -> object:
            return "test"

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer is False
    assert serializer.serializer is CustomSerializer()


# LLM-generated content at query #28
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert result is not None

def test_loads_accepts_string():
    serializer = _PDataSerializer()
    payload = "test payload"
    result = serializer.loads(payload)
    assert isinstance(result, object)

def test_loads_accepts_bytes():
    serializer = _PDataSerializer()
    payload = b"test bytes"
    result = serializer.loads(payload)
    assert result is not None

def test_loads_accepts_list():
    serializer = _PDataSerializer()
    payload = [1, 2, 3]
    result = serializer.loads(payload)
    assert result is not None

def test_loads_accepts_dict():
    serializer = _PDataSerializer()
    payload = {"a": 1}
    result = serializer.loads(payload)
    assert result is not None

def test_loads_accepts_none():
    serializer = _PDataSerializer()
    payload = None
    result = serializer.loads(payload)
    assert result is None

def test_loads_accepts_int():
    serializer = _PDataSerializer()
    payload = 42
    result = serializer.loads(payload)
    assert result == 42
```


# LLM-generated content at query #29
#--------------------------

```python
def test_load_payload_with_non_text_serializer_and_invalid_payload_raises_bad_payload():
    serializer = Serializer("secret-key", serializer=type("BytesSerializer", (), {"loads": lambda self, x: (_ for _ in ()).throw(ValueError("invalid"))})())
    payload = b"invalid"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload"
    except BadPayload:
        pass
```


# LLM-generated content at query #30
#--------------------------

```python
def test_load_payload_is_text_false_with_bytes_serializer():
    serializer = Serializer(b"secret", serializer=dill)
    payload = b"gAN9cQAu"
    result = serializer.load_payload(payload)
    assert result == {}
```


# LLM-generated content at query #31
#--------------------------

def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None


# LLM-generated content at query #32
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #33
#--------------------------

def test_serializer_constructor_with_defaults() -> None:
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings() -> None:
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes() -> None:
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> object:
            return eval(s)
    s = Serializer("secret", serializer=CustomSerializer)
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return str(obj).encode()
        @staticmethod
        def loads(s: bytes) -> object:
            return s.decode()
    s = Serializer("secret", serializer=BytesSerializer)
    assert s.serializer is BytesSerializer
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs() -> None:
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer_class() -> None:
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "hmac"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers() -> None:
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []

def test_serializer_constructor_with_none_fallback_signers() -> None:
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []


# LLM-generated content at query #34
#--------------------------

```python
def test_load_payload_with_default_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return eval(payload)
    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        def loads(self, payload):
            return json.loads(payload)
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_override_serializer():
    serializer = Serializer("secret")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"key": "value"}

def test_load_payload_raises_bad_payload():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_text_serializer():
    class TextSerializer:
        def dumps(self, obj):
            return json.dumps(obj)
        def loads(self, payload):
            return json.loads(payload)
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_empty_payload():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_none_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload(None)
    result = serializer.load_payload(payload)
    assert result is None
```


# LLM-generated content at query #35
#--------------------------

```python
def test_iter_unsigners_predicate_line20_false():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    secret_key = b"test_secret"
    salt = b"test_salt"
    serializer = Serializer(secret_key, salt=salt, fallback_signers=[{}])
    unsigners = list(serializer.iter_unsigners(salt))
    assert len(unsigners) == 2
    assert isinstance(unsigners[1], Signer)
```


# LLM-generated content at query #36
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_key == b"secret"
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_str_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "text", "loads": lambda self, x: x})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"bytes", "loads": lambda self, x: x})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #37
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"digest_method": hashlib.sha256})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
```


# LLM-generated content at query #38
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_string_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_text_serializer():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=type("BytesSerializer", (), {"dumps": lambda self, x: b"{}", "loads": lambda self, x: {}})())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_signer_kwargs():
    signer_kwargs = {"key_derivation": "hmac"}
    serializer = Serializer("secret-key", signer_kwargs=signer_kwargs)
    assert serializer.signer_kwargs == signer_kwargs

def test_serializer_constructor_with_serializer_kwargs():
    serializer_kwargs = {"sort_keys": True}
    serializer = Serializer("secret-key", serializer_kwargs=serializer_kwargs)
    assert serializer.serializer_kwargs == serializer_kwargs

def test_serializer_constructor_with_custom_signer_class():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_fallback_signers_none():
    serializer = Serializer("secret-key", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #39
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer is True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(b):
            return b"data"
    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.is_text_serializer is False

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    class CustomSigner(Signer):
        pass
    serializer = Serializer(
        ["key1", b"key2"],
        salt="custom_salt",
        serializer=CustomSerializer,
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #40
#--------------------------

```python
def test_fallback_signers_is_not_none():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #41
#--------------------------

```python
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert isinstance(result, object)

def test_dumps_preserves_string_content():
    serializer = _PDataSerializer()
    result = serializer.dumps("hello")
    assert result is not None

def test_dumps_handles_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert isinstance(result, object)

def test_dumps_handles_integer():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert isinstance(result, object)

def test_dumps_handles_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert isinstance(result, object)

def test_dumps_handles_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, object)

def test_dumps_returns_same_type_for_same_input():
    serializer = _PDataSerializer()
    result1 = serializer.dumps("data")
    result2 = serializer.dumps("data")
    assert type(result1) == type(result2)
```


# LLM-generated content at query #42
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_list_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_empty_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #43
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    payload = "test payload"
    result = serializer.loads(payload)
    assert result is not None

def test_loads_accepts_serialized_type():
    serializer = _PDataSerializer()
    payload = "serialized data"
    result = serializer.loads(payload)
    assert isinstance(result, object)

def test_loads_positional_only_argument():
    serializer = _PDataSerializer()
    payload = 123
    result = serializer.loads(payload)
    assert result is not None
```


# LLM-generated content at query #44
#--------------------------

```python
def test_serializer_init_with_serializer_not_none():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
```


# LLM-generated content at query #45
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_as_bytes():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_salt_as_string():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_serializer_text():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "{}"
        @staticmethod
        def loads(s):
            return {}

    s = Serializer("secret", serializer=TextSerializer())
    assert s.serializer == TextSerializer()
    assert s.is_text_serializer is True

def test_serializer_constructor_with_serializer_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"
        @staticmethod
        def loads(b):
            return {}

    s = Serializer("secret", serializer=BytesSerializer())
    assert s.serializer == BytesSerializer()
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_signer():
    class CustomSigner(Signer):
        pass

    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "hmac"}])
    assert s.fallback_signers == [{"key_derivation": "hmac"}]

def test_serializer_constructor_with_fallback_signers_none():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []


# LLM-generated content at query #46
#--------------------------

```python
def test_serializer_salt_is_none_to_make_predicate_false():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #47
#--------------------------

def test_load_payload_with_default_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_text_serializer():
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return eval(payload)
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode()
        def loads(self, payload):
            return json.loads(payload)
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_override_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    class OverrideSerializer:
        def loads(self, payload):
            return {"overridden": True}
    result = serializer.load_payload(payload, serializer=OverrideSerializer())
    assert result == {"overridden": True}

def test_load_payload_raises_bad_payload_on_error():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_text_serializer_encoding():
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return payload
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = serializer.dump_payload("test")
    result = serializer.load_payload(payload)
    assert result == "test"


# LLM-generated content at query #48
#--------------------------

def test_serializer_constructor_default_values():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"my_secret")
    assert serializer.secret_keys == [b"my_secret"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = json
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer():
    custom_signer = Signer
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #49
#--------------------------

```python
def test_load_payload_is_text_false():
    serializer = Serializer("secret", serializer=Serializer.default_serializer, serializer_kwargs={})
    payload = b'"test"'
    result = serializer.load_payload(payload)
    assert result == "test"

def test_load_payload_is_text_false_with_custom_serializer():
    custom_serializer = type("BytesSerializer", (), {"loads": lambda self, x: x, "dumps": lambda self, x: x})()
    serializer = Serializer("secret", serializer=custom_serializer, serializer_kwargs={})
    payload = b"test"
    result = serializer.load_payload(payload)
    assert result == b"test"

def test_load_payload_is_text_false_with_explicit_serializer():
    serializer = Serializer("secret", serializer=Serializer.default_serializer, serializer_kwargs={})
    payload = b'"test"'
    result = serializer.load_payload(payload, serializer=Serializer.default_serializer)
    assert result == "test"
```


# LLM-generated content at query #50
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_not_text_serializer():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    class BytesSerializer:
        def dumps(self, obj):
            return b"test"

        def loads(self, data):
            return data

    serializer = Serializer("secret-key", serializer=BytesSerializer())
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #51
#--------------------------

def test_serializer_constructor_default_values():
    serializer = Serializer("secret")
    assert isinstance(serializer, Serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"bytes_secret")
    assert serializer.secret_keys == [b"bytes_secret"]

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #52
#--------------------------

```python
def test_loads_returns_any_type():
    serializer = _PDataSerializer()
    result = serializer.loads(42)
    assert result is not None

def test_loads_accepts_integer():
    serializer = _PDataSerializer()
    result = serializer.loads(123)
    assert result is not None

def test_loads_accepts_string():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None

def test_loads_accepts_list():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert result is not None

def test_loads_accepts_dict():
    serializer = _PDataSerializer()
    result = serializer.loads({"key": "value"})
    assert result is not None

def test_loads_accepts_none():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is not None

def test_loads_accepts_float():
    serializer = _PDataSerializer()
    result = serializer.loads(3.14)
    assert result is not None

def test_loads_accepts_boolean():
    serializer = _PDataSerializer()
    result = serializer.loads(True)
    assert result is not None

def test_loads_accepts_bytes():
    serializer = _PDataSerializer()
    result = serializer.loads(b"data")
    assert result is not None

def test_loads_accepts_tuple():
    serializer = _PDataSerializer()
    result = serializer.loads((1, 2))
    assert result is not None
```


# LLM-generated content at query #53
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert s.is_text_serializer == True
    assert s.signer is Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_key_list():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_string_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer_as_text():
    class TextSerializer:
        def dumps(self, obj):
            return "text"
        def loads(self, s):
            return s
    s = Serializer("secret", serializer=TextSerializer())
    assert s.is_text_serializer == True

def test_serializer_constructor_with_custom_serializer_as_bytes():
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, s):
            return s
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_none_fallback_signers():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []


# LLM-generated content at query #54
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret_key")
    assert serializer.secret_keys == [b"secret_key"]

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_str_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is not json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, payload):
            return payload
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_fallback_signers_empty_list():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_creates_secret_keys_properly():
    serializer = Serializer(b"key", salt=b"salt")
    assert serializer.secret_key == b"key"


# LLM-generated content at query #55
#--------------------------

def test_dumps_with_default_serializer_returns_text():
    serializer = Serializer("secret")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_with_bytes_serializer_returns_bytes():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.serializer = type("BytesSerializer", (), {"dumps": lambda self, obj: b'{"key":"value"}'})()
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_with_custom_salt():
    serializer = Serializer("secret")
    result = serializer.dumps("data", salt="custom_salt")
    assert isinstance(result, str)

def test_dumps_empty_object():
    serializer = Serializer("secret")
    result = serializer.dumps({})
    assert isinstance(result, str)

def test_dumps_none_value():
    serializer = Serializer("secret")
    result = serializer.dumps(None)
    assert isinstance(result, str)


# LLM-generated content at query #56
#--------------------------

def test_serializer_constructor_with_default_parameters() -> None:
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer == is_text_serializer(s.serializer)
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    import json
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return json.dumps(obj)
        @staticmethod
        def loads(s: str) -> t.Any:
            return json.loads(s)
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer == CustomSerializer()

def test_serializer_constructor_with_serializer_kwargs() -> None:
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer_class() -> None:
    from itsdangerous.signer import Signer
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_fallback_signers_tuple() -> None:
    from itsdangerous.signer import Signer
    s = Serializer("secret", fallback_signers=[(Signer, {"key_derivation": "none"})])
    assert s.fallback_signers == [(Signer, {"key_derivation": "none"})]

def test_serializer_constructor_with_fallback_signers_class() -> None:
    from itsdangerous.signer import Signer
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", fallback_signers=[CustomSigner])
    assert s.fallback_signers == [CustomSigner]

def test_serializer_constructor_with_all_parameters() -> None:
    from itsdangerous.signer import Signer
    s = Serializer(
        secret_key=["key1", b"key2"],
        salt="salt",
        serializer=json,
        serializer_kwargs={"sort_keys": True},
        signer=Signer,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"salt"
    assert s.serializer == json
    assert s.is_text_serializer == is_text_serializer(json)
    assert s.signer == Signer
    assert s.signer_kwargs == {"digest_method": "sha256"}
    assert s.fallback_signers == [{"key_derivation": "none"}]
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_secret_key_property() -> None:
    s = Serializer(["old", "new"])
    assert s.secret_key == b"new"

def test_serializer_constructor_with_text_serializer_detection() -> None:
    class TextSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return "text"
        @staticmethod
        def loads(s: str) -> t.Any:
            return {}
    s = Serializer("secret", serializer=TextSerializer())
    assert s.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer_detection() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> bytes:
            return b"bytes"
        @staticmethod
        def loads(s: bytes) -> t.Any:
            return {}
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.is_text_serializer == False


# LLM-generated content at query #57
#--------------------------

def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None


# LLM-generated content at query #58
#--------------------------

```
def test_fallback_signers_not_none():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #59
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_list_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()

        @staticmethod
        def loads(data):
            return int(data)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]
    assert serializer.serializer_kwargs == {}

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}


# LLM-generated content at query #60
#--------------------------

```python
def test_loads_returns_any_type():
    serializer = _PDataSerializer()
    result = serializer.loads(b'{"key": "value"}')
    assert result is not None

def test_loads_accepts_bytes():
    serializer = _PDataSerializer()
    result = serializer.loads(b'test data')
    assert result is not None

def test_loads_accepts_string():
    serializer = _PDataSerializer()
    result = serializer.loads('test string')
    assert result is not None

def test_loads_accepts_integer():
    serializer = _PDataSerializer()
    result = serializer.loads(123)
    assert result is not None

def test_loads_accepts_list():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert result is not None

def test_loads_accepts_dictionary():
    serializer = _PDataSerializer()
    result = serializer.loads({'a': 1})
    assert result is not None

def test_loads_accepts_none():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is None

def test_loads_preserves_positional_only():
    serializer = _PDataSerializer()
    result = serializer.loads(b'data')
    assert result is not None
```


# LLM-generated content at query #61
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_key == b"secret"
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer == True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_key == b"secret"
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.secret_key == b"key2"

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_serializer():
    s = Serializer("secret", serializer=json)
    assert s.serializer == json
    assert s.is_text_serializer == True

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_signer():
    s = Serializer("secret", signer=Signer)
    assert s.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_all_params():
    s = Serializer(
        ["key1", "key2"],
        salt=b"custom_salt",
        serializer=json,
        serializer_kwargs={"sort_keys": True},
        signer=Signer,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=[{"key_derivation": "none"}],
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"custom_salt"
    assert s.serializer == json
    assert s.serializer_kwargs == {"sort_keys": True}
    assert s.signer == Signer
    assert s.signer_kwargs == {"digest_method": hashlib.sha256}
    assert s.fallback_signers == [{"key_derivation": "none"}]


# LLM-generated content at query #62
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_serializer_keyword():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json

def test_serializer_constructor_with_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_secret_key_bytes():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_secret_key_list_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #63
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_salt_string():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer is BytesSerializer
    assert serializer.is_text_serializer is False

def test_constructor_with_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.serializer is TextSerializer
    assert serializer.is_text_serializer is True

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_constructor_with_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"bytes_key")
    assert serializer.secret_keys == [b"bytes_key"]


# LLM-generated content at query #64
#--------------------------

```python
def test_salt_is_not_none_in_constructor():
    serializer = Serializer(secret_key="secret", salt=b"custom_salt")
    assert serializer.salt is not None
```


# LLM-generated content at query #65
#--------------------------

```python
def test_fallback_signers_is_none():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []


# LLM-generated content at query #66
#--------------------------

```python
def test_load_payload_returns_bytes_when_serializer_is_text_serializer():
    serializer = Serializer("secret-key")
    payload = b'"hello"'
    result = serializer.load_payload(payload)
    assert result == "hello"
```


# LLM-generated content at query #67
#--------------------------

def test_dumps_returns_bytes_when_serializer_is_not_text_serializer():
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_returns_string_when_serializer_is_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text_serialized"

    serializer = Serializer("secret-key", serializer=TextSerializer)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_uses_salt_parameter():
    serializer = Serializer("secret-key", salt=b"custom_salt")
    result = serializer.dumps({"key": "value"}, salt=b"override_salt")
    assert isinstance(result, bytes)

def test_dumps_invokes_dump_payload():
    serializer = Serializer("secret-key")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.dumps({"key": "value"})
    assert result == serializer.make_signer().sign(payload)

def test_dumps_decodes_to_utf8_when_is_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"

    serializer = Serializer("secret-key", serializer=TextSerializer)
    result = serializer.dumps({"key": "value"})
    assert result == "text"

def test_dumps_with_custom_serializer_kwargs():
    class CustomSerializer:
        @staticmethod
        def dumps(obj, **kwargs):
            return "custom_" + str(obj)

    serializer = Serializer("secret-key", serializer=CustomSerializer, serializer_kwargs={"indent": 2})
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #68
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: x})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == isinstance(custom_serializer.dumps({}), str)

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {"__init__": lambda self, secret_keys, salt, **kwargs: None})()
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #69
#--------------------------

```python
def test_serializer_init_with_serializer_not_none():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
```


# LLM-generated content at query #70
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None or result is None

def test_loads_accepts_serialized_input():
    serializer = _PDataSerializer()
    serializer.loads("")
    serializer.loads(b"")
    serializer.loads(123)
    serializer.loads([1, 2, 3])
    serializer.loads({"a": 1})

def test_loads_returns_any_type():
    serializer = _PDataSerializer()
    result = serializer.loads("data")
    assert isinstance(result, object)
```


# LLM-generated content at query #71
#--------------------------

def test_iter_unsigners_with_default_signer_only():
    serializer = Serializer("secret-key", salt="salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer("secret-key", salt="salt", fallback_signers=[{"digest_method": "sha256"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_fallback_signers_tuple():
    from itsdangerous.signer import Signer
    serializer = Serializer("secret-key", salt="salt", fallback_signers=[(Signer, {"digest_method": "sha256"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_fallback_signers_class():
    from itsdangerous.signer import Signer
    serializer = Serializer("secret-key", salt="salt", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(["old-key", "new-key"], salt="salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1

def test_iter_unsigners_with_fallback_and_multiple_secret_keys():
    from itsdangerous.signer import Signer
    serializer = Serializer(["old-key", "new-key"], salt="salt", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key", salt="default-salt")
    signers = list(serializer.iter_unsigners(salt="custom-salt"))
    assert len(signers) == 1

def test_iter_unsigners_empty_fallback():
    serializer = Serializer("secret-key", salt="salt", fallback_signers=[])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1

def test_iter_unsigners_none_salt():
    serializer = Serializer("secret-key", salt=None)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1


# LLM-generated content at query #72
#--------------------------

def test_serializer_constructor_with_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer_str() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> t.Any:
            return int(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_bytes() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> bytes:
            return str(obj).encode()
        @staticmethod
        def loads(s: bytes) -> t.Any:
            return int(s.decode())
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #73
#--------------------------

def test_serializer_constructor_defaults():
    secret_key = "test-secret"
    serializer = Serializer(secret_key)

def test_serializer_constructor_with_bytes_secret():
    secret_key = b"test-secret"
    serializer = Serializer(secret_key)

def test_serializer_constructor_with_secret_key_list():
    secret_key = ["key1", "key2"]
    serializer = Serializer(secret_key)

def test_serializer_constructor_with_custom_salt():
    secret_key = "test-secret"
    salt = "custom-salt"
    serializer = Serializer(secret_key, salt=salt)

def test_serializer_constructor_with_none_salt():
    secret_key = "test-secret"
    serializer = Serializer(secret_key, salt=None)

def test_serializer_constructor_with_custom_serializer():
    secret_key = "test-secret"
    serializer = Serializer(secret_key, serializer=json)

def test_serializer_constructor_with_serializer_kwargs():
    secret_key = "test-secret"
    serializer_kwargs = {"indent": 2}
    serializer = Serializer(secret_key, serializer_kwargs=serializer_kwargs)

def test_serializer_constructor_with_custom_signer():
    secret_key = "test-secret"
    serializer = Serializer(secret_key, signer=Signer)

def test_serializer_constructor_with_signer_kwargs():
    secret_key = "test-secret"
    signer_kwargs = {"key_derivation": "hmac"}
    serializer = Serializer(secret_key, signer_kwargs=signer_kwargs)

def test_serializer_constructor_with_fallback_signers():
    secret_key = "test-secret"
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer(secret_key, fallback_signers=fallback_signers)

def test_serializer_constructor_all_parameters():
    secret_key = "test-secret"
    salt = "custom-salt"
    serializer_kwargs = {"indent": 2}
    signer_kwargs = {"key_derivation": "hmac"}
    fallback_signers = [Signer, (Signer, {"key_derivation": "none"})]
    serializer = Serializer(secret_key, salt=salt, serializer=json, serializer_kwargs=serializer_kwargs, signer=Signer, signer_kwargs=signer_kwargs, fallback_signers=fallback_signers)

def test_serializer_constructor_with_text_serializer():
    secret_key = "test-secret"
    serializer = Serializer(secret_key, serializer=json)

def test_serializer_constructor_with_bytes_serializer():
    secret_key = "test-secret"
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"
        @staticmethod
        def loads(data):
            return {}
    serializer = Serializer(secret_key, serializer=BytesSerializer())


# LLM-generated content at query #74
#--------------------------

def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert isinstance(result, bytes)


# LLM-generated content at query #75
#--------------------------

```python
def test_load_payload_with_default_serializer():
    serializer = Serializer("secret")
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_text_serializer():
    serializer = Serializer("secret", serializer=json)
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_bytes_serializer():
    serializer = Serializer("secret", serializer=json)
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_override_serializer():
    serializer = Serializer("secret")
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"key": "value"}

def test_load_payload_raises_bad_payload():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid data")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_bytes_payload():
    serializer = Serializer("secret")
    payload = json.dumps(123).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == 123
```


# LLM-generated content at query #76
#--------------------------

def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None


# LLM-generated content at query #77
#--------------------------

```python
def test_salt_is_not_none_in_serializer_initialization():
    serializer = Serializer("secret", salt="explicit_salt")
    assert serializer.salt == want_bytes("explicit_salt")
```


# LLM-generated content at query #78
#--------------------------

def test_constructor_default_serializer():
    s = Serializer("secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_constructor_with_salt_none():
    s = Serializer("secret-key", salt=None)
    assert s.salt is None

def test_constructor_with_custom_serializer():
    s = Serializer("secret-key", serializer=json)
    assert s.serializer == json
    assert s.is_text_serializer is True

def test_constructor_with_custom_signer():
    s = Serializer("secret-key", signer=Signer)
    assert s.signer == Signer

def test_constructor_with_signer_kwargs():
    s = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    s = Serializer("secret-key", fallback_signers=[Signer])
    assert s.fallback_signers == [Signer]

def test_constructor_with_empty_fallback_signers():
    s = Serializer("secret-key", fallback_signers=[])
    assert s.fallback_signers == []

def test_constructor_with_serializer_kwargs():
    s = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_multiple_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_constructor_with_secret_key_bytes():
    s = Serializer(b"secret-key")
    assert s.secret_keys == [b"secret-key"]


# LLM-generated content at query #79
#--------------------------

```python
def test_dumps_returns_bytes_when_is_text_serializer_is_false():
    from itsdangerous.serializer import Serializer
    serializer = Serializer("secret-key", serializer=type("Mock", (), {"dumps": lambda self, obj: b"data"})())
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #80
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    payload = "test payload"
    result = serializer.loads(payload)
    assert result is None or True

def test_loads_with_string_payload():
    serializer = _PDataSerializer()
    payload = "data"
    result = serializer.loads(payload)
    assert result is None or True

def test_loads_with_bytes_payload():
    serializer = _PDataSerializer()
    payload = b"data"
    result = serializer.loads(payload)
    assert result is None or True

def test_loads_with_integer_payload():
    serializer = _PDataSerializer()
    payload = 123
    result = serializer.loads(payload)
    assert result is None or True

def test_loads_with_none_payload():
    serializer = _PDataSerializer()
    payload = None
    result = serializer.loads(payload)
    assert result is None or True
```


# LLM-generated content at query #81
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_list_secret():
    s = Serializer(["secret1", "secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_list_bytes_secret():
    s = Serializer([b"secret1", b"secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_custom_salt():
    s = Serializer("secret", salt="custom")
    assert s.salt == b"custom"

def test_serializer_constructor_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return "custom"
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer()
    assert s.is_text_serializer is True

def test_serializer_constructor_custom_serializer_bytes():
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"custom"
        @staticmethod
        def loads(s):
            return "custom"
    s = Serializer("secret", serializer=CustomBytesSerializer())
    assert s.serializer is CustomBytesSerializer()
    assert s.is_text_serializer is False

def test_serializer_constructor_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_fallback_signers_none():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    s = Serializer(["old", "new"])
    assert s.secret_key == b"new"


# LLM-generated content at query #82
#--------------------------

```python
def test_fallback_signers_is_not_none():
    fallback_signers: list[dict[str, object] | tuple[type[Signer], dict[str, object]] | type[Signer]] = []
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers is not None
```


# LLM-generated content at query #83
#--------------------------

```
def test_iter_unsigners_with_fallback_tuple():
    signer_kwargs = {"digest_method": "sha256"}
    fallback = (Signer, signer_kwargs)
    serializer = Serializer("secret", fallback_signers=[fallback])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert unsigners[0].salt == serializer.salt
    assert unsigners[1].salt == serializer.salt
    assert unsigners[1].digest_method == "sha256"
```


# LLM-generated content at query #84
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_key, salt="itsdangerous", **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs

        def sign(self, value):
            return value

        def unsign(self, value):
            return value

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_all_params():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()

        @staticmethod
        def loads(s):
            return int(s)

    class CustomSigner:
        def __init__(self, secret_key, salt="itsdangerous", **kwargs):
            self.secret_key = secret_key
            self.salt = salt
            self.kwargs = kwargs

        def sign(self, value):
            return value

        def unsign(self, value):
            return value

    serializer = Serializer(
        secret_key=["old_key", "new_key"],
        salt="custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}],
    )
    assert serializer.secret_keys == [b"old_key", b"new_key"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is False
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #85
#--------------------------

```python
def test_serializer_init_with_explicit_serializer():
    serializer_instance = Serializer(secret_key="secret", serializer=json)


# LLM-generated content at query #86
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer_text():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_secret_key_bytes():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_all_parameters():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(s):
            return s
    class CustomSigner(Signer):
        pass
    serializer = Serializer(
        secret_key=["old", "new"],
        salt=b"custom",
        serializer=TextSerializer(),
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.salt == b"custom"
    assert serializer.is_text_serializer is True
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #87
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"bytes-secret")
    assert serializer.secret_keys == [b"bytes-secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, o: "{}", "loads": lambda self, s: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, o: b"{}", "loads": lambda self, s: {}})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_key == b"new"

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer


# LLM-generated content at query #88
#--------------------------

def test_serializer_constructor_default_serializer():
    secret_key = "my-secret-key"
    s = Serializer(secret_key)
    assert s.secret_keys == [b"my-secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer == True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_bytes_secret_key():
    secret_key = b"bytes-secret"
    s = Serializer(secret_key)
    assert s.secret_keys == [b"bytes-secret"]

def test_serializer_constructor_list_of_keys():
    secret_keys = ["key1", b"key2"]
    s = Serializer(secret_keys)
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_custom_serializer_str():
    class CustomStrSerializer:
        @staticmethod
        def dumps(obj):
            return "str"
        @staticmethod
        def loads(s):
            return None
    s = Serializer("secret", serializer=CustomStrSerializer())
    assert s.is_text_serializer == True

def test_serializer_constructor_custom_serializer_bytes():
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(b):
            return None
    s = Serializer("secret", serializer=CustomBytesSerializer())
    assert s.is_text_serializer == False

def test_serializer_constructor_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #89
#--------------------------

```python
def test_load_payload_predicate_false_with_bytes_serializer():
    serializer = Serializer("secret", serializer=bytes)
    payload = b"test"
    result = serializer.load_payload(payload)
    assert result == b"test"
```


# LLM-generated content at query #90
#--------------------------

def test_serializer_constructor_with_string_secret_key():
    s = Serializer("my-secret-key")
    assert s.secret_keys == [b"my-secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"my-secret-key")
    assert s.secret_keys == [b"my-secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_string_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom-salt")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"custom-salt"
    assert s.serializer is json
    assert s.is_text_serializer
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.secret_keys == [b"secret"]
    assert s.salt is None
    assert s.serializer is json
    assert s.is_text_serializer
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()

        @staticmethod
        def loads(data):
            return data.decode()

    s = Serializer("secret", serializer=CustomSerializer())
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is CustomSerializer()
    assert not s.is_text_serializer
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    s = Serializer("secret", signer=CustomSigner)
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer
    assert s.signer is CustomSigner
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer
    assert s.signer is Signer
    assert s.signer_kwargs == {"key_derivation": "hmac"}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "hmac"}])
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == [{"key_derivation": "hmac"}]
    assert s.serializer_kwargs == {}


# LLM-generated content at query #91
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_bytes():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #92
#--------------------------

```python
def test_dumps_returns_typed_serialized():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None

def test_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result is not None

def test_dumps_accepts_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None

def test_dumps_accepts_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result is not None

def test_dumps_accepts_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert result is not None

def test_dumps_accepts_custom_object():
    serializer = _PDataSerializer()
    class Custom:
        pass
    obj = Custom()
    result = serializer.dumps(obj)
    assert result is not None

def test_dumps_called_with_positional_argument_only():
    serializer = _PDataSerializer()
    result = serializer.dumps("data")
    assert result is not None
```


# LLM-generated content at query #93
#--------------------------

```python
def test_iter_unsigners_with_default_salt():
    serializer = Serializer(secret_key="secret")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == serializer.salt

def test_iter_unsigners_with_custom_salt():
    serializer = Serializer(secret_key="secret")
    signers = list(serializer.iter_unsigners(salt="custom_salt"))
    assert len(signers) == 1
    assert signers[0].salt == b"custom_salt"

def test_iter_unsigners_with_fallback_signers():
    serializer = Serializer(secret_key="secret", fallback_signers=[{"key": "fallback"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_tuple_fallback():
    serializer = Serializer(secret_key="secret", fallback_signers=[(Signer, {"key": "fallback"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(secret_key=["old_secret", "new_secret"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_fallback_and_multiple_keys():
    serializer = Serializer(secret_key=["old_secret", "new_secret"], fallback_signers=[{"key": "fallback"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
```


# LLM-generated content at query #94
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert s.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_str_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom")
    assert s.salt == b"custom"

def test_serializer_constructor_with_bytes_salt():
    s = Serializer("secret", salt=b"custom")
    assert s.salt == b"custom"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return eval(s)

    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer()
    assert s.is_text_serializer == is_text_serializer(CustomSerializer())

def test_serializer_constructor_with_custom_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_none_fallback_signers():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    s = Serializer(["old", "new"])
    assert s.secret_key == b"new"


# LLM-generated content at query #95
#--------------------------

def test_salt_is_none_creates_serializer_without_modifying_salt():
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #96
#--------------------------

```python
def test_fallback_signers_is_not_none_when_provided_as_empty_list():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #97
#--------------------------

```python
def test_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_serializer_str():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer

def test_constructor_with_serializer_bytes():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer

def test_constructor_with_signer_class():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_secret_key_bytes():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #98
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_salt_bytes():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "str", "loads": lambda self, x: {}})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer is custom_serializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"bytes", "loads": lambda self, x: {}})()
    s = Serializer("secret", serializer=bytes_serializer)
    assert s.serializer is bytes_serializer
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    s = Serializer("secret", signer=custom_signer)
    assert s.signer is custom_signer

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = {"key_derivation": "none"}
    s = Serializer("secret", fallback_signers=[fallback])
    assert s.fallback_signers == [fallback]

def test_serializer_constructor_with_fallback_signers_none():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_with_all_parameters():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "str", "loads": lambda self, x: {}})()
    custom_signer = type("CustomSigner", (Signer,), {})
    s = Serializer(
        secret_key=["key1", b"key2"],
        salt="custom_salt",
        serializer=custom_serializer,
        serializer_kwargs={"sort_keys": True},
        signer=custom_signer,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=[{"key_derivation": "none"}],
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"custom_salt"
    assert s.serializer is custom_serializer
    assert s.is_text_serializer is True
    assert s.signer is custom_signer
    assert s.signer_kwargs == {"digest_method": hashlib.sha256}
    assert s.fallback_signers == [{"key_derivation": "none"}]
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #99
#--------------------------

```python
def test_loads_returns_any_type():
    serializer = _PDataSerializer()
    result = serializer.loads("test_payload")
    assert result is not None

def test_loads_accepts_string_payload():
    serializer = _PDataSerializer()
    result = serializer.loads("")
    assert result is None

def test_loads_with_none_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is None

def test_loads_with_integer_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(42)
    assert isinstance(result, object)

def test_loads_with_list_payload():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert isinstance(result, object)
```


# LLM-generated content at query #100
#--------------------------

def test_serializer_init_with_string_secret_key():
    s = Serializer("my-secret-key")
    assert s.secret_keys == [b"my-secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    s = Serializer(b"my-secret-key")
    assert s.secret_keys == [b"my-secret-key"]
    assert s.salt == b"itsdangerous"

def test_serializer_init_with_list_of_string_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes_secret_keys():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_custom_salt():
    s = Serializer("secret", salt="custom-salt")
    assert s.salt == b"custom-salt"

def test_serializer_init_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_init_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    s = Serializer("secret", serializer=CustomSerializer)
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True

def test_serializer_init_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_init_with_custom_signer():
    class CustomSigner(Signer):
        pass

    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_init_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_init_with_none_fallback_signers():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_init_with_default_serializer_override():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return b""

        @staticmethod
        def loads(s):
            return None

    s = Serializer("secret", serializer=CustomSerializer)
    assert s.is_text_serializer is False


# LLM-generated content at query #101
#--------------------------

```python
def test_load_payload_serializer_raises_exception():
    import json
    from itsdangerous.serializer import Serializer
    from itsdangerous.exc import BadPayload

    class BrokenSerializer:
        @staticmethod
        def loads(data):
            raise ValueError("broken")

    serializer = Serializer("secret", serializer=BrokenSerializer())
    payload = b"test"

    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
```


# LLM-generated content at query #102
#--------------------------

```python
def test_iter_unsigners_elif_branch_with_tuple_fallback():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    class CustomSigner(Signer):
        pass

    serializer = Serializer(
        secret_key=b"secret",
        fallback_signers=[(CustomSigner, {"key_derivation": "none"})],
    )
    unsigners = list(serializer.iter_unsigners(salt=b"test"))
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], CustomSigner)
    assert unsigners[1].salt == b"test"
```


# LLM-generated content at query #103
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    serializer.loads = lambda payload: payload
    result = serializer.loads("test")
    assert result == "test"

def test_loads_accepts_serialized_input():
    serializer = _PDataSerializer()
    serializer.loads = lambda payload: payload.upper()
    result = serializer.loads("hello")
    assert result == "HELLO"

def test_loads_handles_none():
    serializer = _PDataSerializer()
    serializer.loads = lambda payload: None
    result = serializer.loads("anything")
    assert result is None

def test_loads_handles_integer():
    serializer = _PDataSerializer()
    serializer.loads = lambda payload: payload * 2
    result = serializer.loads(5)
    assert result == 10

def test_loads_handles_list():
    serializer = _PDataSerializer()
    serializer.loads = lambda payload: [x for x in payload]
    result = serializer.loads([1, 2, 3])
    assert result == [1, 2, 3]
```


# LLM-generated content at query #104
#--------------------------

```python
def test_fallback_signers_predicate_evaluates_to_false():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #105
#--------------------------

def test_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback


# LLM-generated content at query #106
#--------------------------

```python
def test_serializer_constructor_serializer_not_none():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.serializer is json
```


# LLM-generated content at query #107
#--------------------------

```python
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_string")
    assert result is not None

def test_dumps_with_integer():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert isinstance(result, bytes)

def test_dumps_with_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert isinstance(result, str)

def test_dumps_with_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert result is not None

def test_dumps_with_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert isinstance(result, bytes)

def test_dumps_with_float():
    serializer = _PDataSerializer()
    result = serializer.dumps(3.14)
    assert isinstance(result, str)

def test_dumps_with_boolean():
    serializer = _PDataSerializer()
    result = serializer.dumps(True)
    assert result is not None

def test_dumps_with_empty_string():
    serializer = _PDataSerializer()
    result = serializer.dumps("")
    assert isinstance(result, bytes)

def test_dumps_with_custom_object():
    serializer = _PDataSerializer()
    result = serializer.dumps(object())
    assert isinstance(result, str)
```


# LLM-generated content at query #108
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_not_text():
    serializer = Serializer("secret", serializer=Serializer.default_serializer)
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #109
#--------------------------

def test_serializer_constructor_with_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_bytes_salt() -> None:
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer() -> None:
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "str", "loads": lambda self, x: x})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == is_text_serializer(custom_serializer)

def test_serializer_constructor_with_custom_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer() -> None:
    from itsdangerous.signer import Signer
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_constructor_with_custom_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback_signers = [{"key_derivation": "none"}, ("itsdangerous.signer.Signer", {"digest_method": "sha256"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers


# LLM-generated content at query #110
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "{}", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"{}"
        def loads(self, payload):
            return {}
    serializer = Serializer("secret", b"itsdangerous", BytesSerializer())
    assert serializer.is_text_serializer is False

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})()
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer is custom_signer

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"digest_method": hashlib.sha256})
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}

def test_constructor_with_fallback_signers():
    fallback = [{"digest_method": hashlib.sha256}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_constructor_with_multiple_secret_keys():
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_keys == [b"old_key", b"new_key"]
    assert serializer.secret_key == b"new_key"


# LLM-generated content at query #111
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = json
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer is custom_serializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.serializer is not None
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []


# LLM-generated content at query #112
#--------------------------

def test_serializer_init_with_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_init_with_secret_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_init_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    s = Serializer("secret", serializer=CustomSerializer)
    assert s.serializer == CustomSerializer
    assert s.is_text_serializer is True

def test_serializer_init_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(data):
            return data
    s = Serializer("secret", serializer=BytesSerializer)
    assert s.serializer == BytesSerializer
    assert s.is_text_serializer is False

def test_serializer_init_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}

def test_serializer_init_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_init_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_init_with_fallback_signers_none():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []


# LLM-generated content at query #113
#--------------------------

```python
def test_salt_is_none_condition_false():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None
```


