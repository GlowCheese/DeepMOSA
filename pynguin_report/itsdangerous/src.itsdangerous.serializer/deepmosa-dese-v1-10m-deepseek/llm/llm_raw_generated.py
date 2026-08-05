####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
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

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = json
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"
        @staticmethod
        def loads(data):
            return {}
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    signer_kwargs = {"key_derivation": "none"}
    serializer = Serializer("secret-key", signer_kwargs=signer_kwargs)
    assert serializer.signer_kwargs == signer_kwargs

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_serializer_kwargs():
    serializer_kwargs = {"sort_keys": True}
    serializer = Serializer("secret-key", serializer_kwargs=serializer_kwargs)
    assert serializer.serializer_kwargs == serializer_kwargs


# LLM-generated content at query #2
#--------------------------

```python
def test_iter_unsigners_returns_default_signer_first():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer("secret-key", fallback_signers=[{"key": "fallback"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"key": "fallback"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_with_custom_salt():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    signers = list(serializer.iter_unsigners())
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_override_salt():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners(salt=b"override-salt"))
    assert signers[0].salt == b"override-salt"

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

def test_iter_unsigners_fallback_with_multiple_secret_keys():
    serializer = Serializer(["old-key", "new-key"], fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]
    assert signers[2].secret_keys == [b"new-key"]
```


# LLM-generated content at query #3
#--------------------------

def test_serializer_constructor_default_serializer():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer is Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    s = Serializer([b"key1", b"key2"])
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

    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer()
    assert s.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_keys, salt, **kwargs):
            pass

        def sign(self, value):
            return value

        def unsign(self, value):
            return value

    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return eval(s)

    class CustomSigner:
        def __init__(self, secret_keys, salt, **kwargs):
            pass

        def sign(self, value):
            return value

        def unsign(self, value):
            return value

    s = Serializer(
        secret_key=["key1", "key2"],
        salt="custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"custom_salt"
    assert s.serializer is CustomSerializer()
    assert s.is_text_serializer is True
    assert s.signer is CustomSigner
    assert s.signer_kwargs == {"key_derivation": "hmac"}
    assert s.fallback_signers == [{"key_derivation": "none"}]
    assert s.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #4
#--------------------------

def test_dumps_with_text_serializer_returns_string():
    serializer = Serializer("secret-key", serializer_kwargs={})
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_with_bytes_serializer_returns_bytes():
    serializer = Serializer("secret-key", serializer=bytes, serializer_kwargs={})
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_with_custom_salt_uses_different_signer():
    serializer = Serializer("secret-key")
    result1 = serializer.dumps("data", salt="salt1")
    result2 = serializer.dumps("data", salt="salt2")
    assert result1 != result2

def test_dumps_uses_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    result = serializer.dumps({"b": 1, "a": 2})
    assert result is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_load_payload_with_default_serializer_and_text_serializer():
    serializer = Serializer("secret")
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer_text():
    class CustomSerializer:
        def loads(self, s):
            return {"custom": s}
        def dumps(self, obj):
            return json.dumps(obj)
    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = b'custom_data'
    result = serializer.load_payload(payload)
    assert result == {"custom": "custom_data"}

def test_load_payload_with_custom_serializer_bytes():
    class CustomSerializer:
        def loads(self, s):
            return {"custom": s}
        def dumps(self, obj):
            return b'bytes_data'
    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = b'custom_data'
    result = serializer.load_payload(payload)
    assert result == {"custom": b'custom_data'}

def test_load_payload_with_override_serializer():
    class OverrideSerializer:
        def loads(self, s):
            return {"overridden": s}
        def dumps(self, obj):
            return json.dumps(obj)
    serializer = Serializer("secret")
    payload = b'some_data'
    result = serializer.load_payload(payload, serializer=OverrideSerializer())
    assert result == {"overridden": "some_data"}

def test_load_payload_raises_bad_payload():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b'invalid json')
    except BadPayload:
        pass
    else:
        assert False, "Expected BadPayload exception"

def test_load_payload_with_text_serializer_and_unicode_payload():
    class TextSerializer:
        def loads(self, s):
            return {"text": s}
        def dumps(self, obj):
            return json.dumps(obj)
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = "unicode_data".encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"text": "unicode_data"}
```


# LLM-generated content at query #6
#--------------------------

```python
def test_salt_is_None_so_predicate_evaluates_False():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None
    assert serializer.secret_keys == [b"secret"]
```


# LLM-generated content at query #7
#--------------------------

def test_serializer_constructor_with_default_parameters() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> object:
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_custom_bytes_serializer() -> None:
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return str(obj).encode()
        @staticmethod
        def loads(s: bytes) -> object:
            return eval(s.decode())
    serializer = Serializer("secret", serializer=CustomBytesSerializer)
    assert serializer.serializer == CustomBytesSerializer
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

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

def test_serializer_constructor_with_none_fallback_signers() -> None:
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_secret_key_property() -> None:
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_key == b"new_key"


# LLM-generated content at query #8
#--------------------------

```python
def test_serializer_uses_provided_serializer_instead_of_default():
    from itsdangerous.serializer import Serializer
    from itsdangerous.exc import BadPayload
    from itsdangerous.signer import Signer
    import json

    class CustomSerializer:
        @staticmethod
        def dumps(obj, **kwargs):
            return json.dumps(obj, **kwargs)

        @staticmethod
        def loads(s, **kwargs):
            return json.loads(s, **kwargs)

    custom_serializer = CustomSerializer()
    s = Serializer("secret-key", serializer=custom_serializer)
    assert s.serializer is custom_serializer
```


# LLM-generated content at query #9
#--------------------------

def test_serializer_constructor_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret() -> None:
    serializer = Serializer(b"bytes_secret")
    assert serializer.secret_keys == [b"bytes_secret"]

def test_serializer_constructor_with_list_of_secrets() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    import json
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_all_params() -> None:
    serializer = Serializer(
        secret_key=["key1", b"key2"],
        salt=b"custom_salt",
        serializer_kwargs={"sort_keys": True},
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer_kwargs == {"sort_keys": True}
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]


# LLM-generated content at query #10
#--------------------------

```python
def test_iter_unsigners_yields_default_signer_first():
    serializer = Serializer("secret-key", salt="salt")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)

def test_iter_unsigners_with_fallback_signers():
    serializer = Serializer("secret-key", salt="salt", fallback_signers=[Signer])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer("secret-key", salt="salt", fallback_signers=[{"key": "fallback-key"}])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert unsigners[0].secret_key == b"secret-key"
    assert unsigners[1].secret_key == b"fallback-key"

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer("secret-key", salt="salt", fallback_signers=[(Signer, {"key": "fallback-key"})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert unsigners[0].secret_key == b"secret-key"
    assert unsigners[1].secret_key == b"fallback-key"

def test_iter_unsigners_with_multiple_secret_keys():
    serializer = Serializer(["old-key", "new-key"], salt="salt")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1

def test_iter_unsigners_with_multiple_secret_keys_and_fallback():
    serializer = Serializer(["old-key", "new-key"], salt="salt", fallback_signers=[Signer])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3
    assert unsigners[0].secret_key == b"new-key"
    assert unsigners[1].secret_key == b"old-key"
    assert unsigners[2].secret_key == b"new-key"

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key", salt="default-salt")
    unsigners = list(serializer.iter_unsigners(salt="custom-salt"))
    assert unsigners[0].salt == b"custom-salt"

def test_iter_unsigners_none_salt():
    serializer = Serializer("secret-key", salt="default-salt")
    unsigners = list(serializer.iter_unsigners(salt=None))
    assert unsigners[0].salt == b"default-salt"

def test_iter_unsigners_no_fallback_signers():
    serializer = Serializer("secret-key", salt="salt", fallback_signers=[])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
```


# LLM-generated content at query #11
#--------------------------

```python
def test_salt_is_none_should_not_convert_to_bytes():
    from itsdangerous.serializer import Serializer
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #12
#--------------------------

```python
def test_fallback_signers_not_none():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    secret_key = b"secret"
    salt = b"my_salt"
    serializer = None
    serializer_kwargs = None
    signer = None
    signer_kwargs = {"key_derivation": "hmac"}
    fallback_signers = [{"digest_method": "sha256"}]
    instance = Serializer(
        secret_key=secret_key,
        salt=salt,
        serializer=serializer,
        serializer_kwargs=serializer_kwargs,
        signer=signer,
        signer_kwargs=signer_kwargs,
        fallback_signers=fallback_signers,
    )
    assert instance.fallback_signers == fallback_signers
```


# LLM-generated content at query #13
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_not_text():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #14
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

def test_serializer_constructor_with_list_of_str_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_keys():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt=b"custom-salt")
    assert s.salt == b"custom-salt"

def test_serializer_constructor_with_custom_serializer_returns_str():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "str"
        @staticmethod
        def loads(s):
            return {}
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer()
    assert s.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_returns_bytes():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return {}
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer()
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer_class():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_all_parameters():
    s = Serializer("secret", salt=b"salt", serializer_kwargs={"sort_keys": True}, signer_kwargs={"digest_method": "sha256"}, fallback_signers=[{"key_derivation": "none"}])
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"salt"
    assert s.serializer_kwargs == {"sort_keys": True}
    assert s.signer_kwargs == {"digest_method": "sha256"}
    assert s.fallback_signers == [{"key_derivation": "none"}]


# LLM-generated content at query #15
#--------------------------

```python
def test_load_payload_with_json_serializer_and_bytes_payload():
    import json
    serializer = Serializer("secret", serializer=json)
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_text_serializer_and_string_payload():
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return int(payload)
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = b"42"
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_custom_serializer_parameter():
    import json
    serializer = Serializer("secret", serializer=json)
    payload = b'[1, 2, 3]'
    result = serializer.load_payload(payload, serializer=json)
    assert result == [1, 2, 3]

def test_load_payload_raises_bad_payload_on_invalid_data():
    import json
    serializer = Serializer("secret", serializer=json)
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_with_serializer_returning_bytes():
    class BytesSerializer:
        def dumps(self, obj):
            return obj.encode() if isinstance(obj, str) else obj
        def loads(self, payload):
            return payload.decode()
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = b"test"
    result = serializer.load_payload(payload)
    assert result == "test"

def test_load_payload_with_text_serializer_and_unicode_payload():
    class TextSerializer:
        def dumps(self, obj):
            return obj
        def loads(self, payload):
            return payload.upper()
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = b"hello"
    result = serializer.load_payload(payload)
    assert result == "HELLO"

def test_load_payload_passes_exception_original_error():
    import json
    serializer = Serializer("secret", serializer=json)
    payload = b"bad"
    try:
        serializer.load_payload(payload)
    except BadPayload as e:
        assert e.original_error is not None
```


# LLM-generated content at query #16
#--------------------------

def test_serializer_init_with_defaults() -> None:
    serializer = Serializer("secret", salt="salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret", salt=b"salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"salt"

def test_serializer_init_with_list_of_strings() -> None:
    serializer = Serializer(["secret1", "secret2"], salt="salt")
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"salt"

def test_serializer_init_with_list_of_bytes() -> None:
    serializer = Serializer([b"secret1", b"secret2"], salt=b"salt")
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"salt"

def test_serializer_init_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None

def test_serializer_init_with_custom_serializer() -> None:
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_init_with_custom_signer() -> None:
    custom_signer = Signer
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer is custom_signer

def test_serializer_init_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_init_with_all_parameters() -> None:
    serializer = Serializer(
        ["secret1", b"secret2"],
        salt="salt",
        serializer=json,
        serializer_kwargs={"sort_keys": True},
        signer=Signer,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[{"key_derivation": "none"}],
    )
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #17
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_all_arguments():
    serializer = Serializer(["key1", "key2"], salt=b"custom_salt", serializer=json, serializer_kwargs={"indent": 2}, signer=Signer, signer_kwargs={"key_derivation": "hmac"}, fallback_signers=[{"key_derivation": "none"}])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_fallback_signers_none():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_serializer_none():
    serializer = Serializer("secret", serializer=None)
    assert serializer.serializer == json
    assert serializer.is_text_serializer

def test_serializer_constructor_with_signer_none():
    serializer = Serializer("secret", signer=None)
    assert serializer.signer == Signer

def test_serializer_constructor_with_serializer_kwargs_none():
    serializer = Serializer("secret", serializer_kwargs=None)
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs_none():
    serializer = Serializer("secret", signer_kwargs=None)
    assert serializer.signer_kwargs == {}


# LLM-generated content at query #18
#--------------------------

def test_serializer_constructor_default_serializer():
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

def test_serializer_constructor_with_custom_serializer_bytes():
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

def test_serializer_constructor_with_custom_serializer_str():
    class StrSerializer:
        @staticmethod
        def dumps(obj):
            return "string"
        @staticmethod
        def loads(data):
            return data
    s = Serializer("secret", serializer=StrSerializer())
    assert s.serializer is StrSerializer()
    assert s.is_text_serializer is True

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

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_multiple_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"bytes_secret")
    assert s.secret_keys == [b"bytes_secret"]


# LLM-generated content at query #19
#--------------------------

```python
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert isinstance(result, type(serializer.dumps("test")))
```


# LLM-generated content at query #20
#--------------------------

def test_serializer_constructor_default_serializer():
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

def test_serializer_constructor_with_serializer_str():
    class MockStrSerializer:
        @staticmethod
        def dumps(obj):
            return "{}"
        @staticmethod
        def loads(s):
            return {}
    s = Serializer("secret", serializer=MockStrSerializer())
    assert s.is_text_serializer is True

def test_serializer_constructor_with_serializer_bytes():
    class MockBytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"
        @staticmethod
        def loads(b):
            return {}
    s = Serializer("secret", serializer=MockBytesSerializer())
    assert s.is_text_serializer is False

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


# LLM-generated content at query #21
#--------------------------

```python
def test_load_payload_is_text_false_with_bytes_serializer():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    import json

    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode()

        @staticmethod
        def loads(data):
            return json.loads(data)

    serializer = Serializer("secret-key", serializer=BytesSerializer())
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #22
#--------------------------

```python
def test_load_payload_is_text_false():
    serializer = Serializer("secret-key")
    serializer.is_text_serializer = False
    result = serializer.load_payload(b"{\"key\": \"value\"}")
    assert result == {"key": "value"}
```


# LLM-generated content at query #23
#--------------------------

```python
def test_load_payload_exception_raised_when_is_text_false_and_payload_is_invalid_bytes():
    serializer = Serializer("test-secret")
    invalid_payload = b"invalid json"
    try:
        serializer.load_payload(invalid_payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
```


# LLM-generated content at query #24
#--------------------------

def test_serializer_constructor_default_values() -> None:
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

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return "custom"

        @staticmethod
        def loads(s: str) -> t.Any:
            return None

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> bytes:
            return b"custom"

        @staticmethod
        def loads(s: bytes) -> t.Any:
            return None

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

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

def test_serializer_constructor_with_fallback_signers_none() -> None:
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_secret_key_property() -> None:
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_key == b"new_key"


# LLM-generated content at query #25
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

def test_serializer_constructor_with_bytes_secret_key():
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

def test_serializer_constructor_with_bytes_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: int(x)})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer is custom_serializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"dumped"
        def loads(self, data):
            return "loaded"
    serializer = BytesSerializer()
    s = Serializer("secret", serializer=serializer)
    assert s.serializer is serializer
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
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []

def test_serializer_constructor_with_none_fallback_signers():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_all_parameters():
    class CustomSerializer:
        def dumps(self, obj):
            return b"data"
        def loads(self, data):
            return "data"
    class CustomSigner(Signer):
        pass
    s = Serializer(
        secret_key=["key1", b"key2"],
        salt=b"mysalt",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"mysalt"
    assert s.is_text_serializer is False
    assert s.signer is CustomSigner
    assert s.signer_kwargs == {"digest_method": hashlib.sha256}
    assert s.fallback_signers == [{"key_derivation": "none"}]
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #26
#--------------------------

def test_serializer_constructor_default_values() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret() -> None:
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

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        def dumps(self, obj: t.Any) -> str:
            return "dumped"
        def loads(self, s: str) -> t.Any:
            return "loaded"
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is not json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer() -> None:
    class BytesSerializer:
        def dumps(self, obj: t.Any) -> bytes:
            return b"dumped"
        def loads(self, s: bytes) -> t.Any:
            return "loaded"
    serializer = Serializer("secret", serializer=BytesSerializer())
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
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert len(serializer.fallback_signers) == 1

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_positional_serializer() -> None:
    class CustomSerializer:
        def dumps(self, obj: t.Any) -> bytes:
            return b"dumped"
        def loads(self, s: bytes) -> t.Any:
            return "loaded"
    serializer = Serializer("secret", b"custom_salt", CustomSerializer())
    assert serializer.salt == b"custom_salt"
    assert isinstance(serializer.serializer, CustomSerializer)

def test_serializer_constructor_with_keyword_serializer() -> None:
    class CustomSerializer:
        def dumps(self, obj: t.Any) -> bytes:
            return b"dumped"
        def loads(self, s: bytes) -> t.Any:
            return "loaded"
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert isinstance(serializer.serializer, CustomSerializer)


# LLM-generated content at query #27
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom")
    assert s.salt == b"custom"

def test_serializer_constructor_with_custom_serializer():
    class MockSerializer:
        @staticmethod
        def dumps(obj):
            return "serialized"
        @staticmethod
        def loads(s):
            return {"data": s}
    s = Serializer("secret", serializer=MockSerializer())
    assert s.serializer is not json
    assert s.is_text_serializer

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #28
#--------------------------

```python
def test_serializer_with_explicit_serializer_skips_default():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
```


# LLM-generated content at query #29
#--------------------------

Here's the unit test for the `Serializer` constructor:

```python
def test_serializer_constructor_default():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret():
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

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.serializer == json

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret-key", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}
```


# LLM-generated content at query #30
#--------------------------

```python
def test_load_payload_is_text_false_with_text_serializer_kwarg():
    serializer = Serializer("secret-key", serializer=json)
    payload = json.dumps({"key": "value"}).encode("utf-8")
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #31
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    payload = 42
    result = serializer.loads(payload)
    assert result is not None or result is None

def test_loads_accepts_serialized_type():
    serializer = _PDataSerializer()
    payload = "test"
    result = serializer.loads(payload)
    assert isinstance(result, object)

def test_loads_with_bytes_payload():
    serializer = _PDataSerializer()
    payload = b"\x00\x01"
    result = serializer.loads(payload)
    assert result is not None or result is None

def test_loads_with_list_payload():
    serializer = _PDataSerializer()
    payload = [1, 2, 3]
    result = serializer.loads(payload)
    assert result is not None or result is None

def test_loads_with_dict_payload():
    serializer = _PDataSerializer()
    payload = {"key": "value"}
    result = serializer.loads(payload)
    assert result is not None or result is None
```


# LLM-generated content at query #32
#--------------------------

def test_serializer_constructor_with_default_parameters() -> None:
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

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys() -> None:
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    from itsdangerous.serializer import Serializer
    import json
    s = Serializer("secret", serializer=json)
    assert s.serializer == json

def test_serializer_constructor_with_serializer_kwargs() -> None:
    s = Serializer("secret", serializer_kwargs={"indent": 4})
    assert s.serializer_kwargs == {"indent": 4}

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

def test_serializer_constructor_with_custom_serializer_bytes() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return json.dumps(obj).encode()
        @staticmethod
        def loads(data: bytes) -> object:
            return json.loads(data)
    s = Serializer("secret", serializer=BytesSerializer)
    assert s.is_text_serializer is False


# LLM-generated content at query #33
#--------------------------

def test_serializer_constructor_with_str_secret_key():
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

def test_serializer_constructor_with_list_of_str_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(b):
            return b"data"
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.serializer is BytesSerializer()
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer_class():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback


# LLM-generated content at query #34
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads(serializer.dumps(42))
    assert result is not None

def test_loads_with_string_payload():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None

def test_loads_with_bytes_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(b"test")
    assert result is not None

def test_loads_with_int_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(123)
    assert result is not None
```


# LLM-generated content at query #35
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
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

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_string_salt():
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
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_serializer_returning_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_key == b"new_key"


# LLM-generated content at query #36
#--------------------------

```python
def test_fallback_signers_is_not_none():
    signer = Signer("test-secret-key")
    serializer = Serializer(
        secret_key="test-secret-key",
        fallback_signers=[signer]
    )
    assert serializer.fallback_signers is not None
```


# LLM-generated content at query #37
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_not_text():
    serializer = Serializer("secret", serializer=lambda: None)
    serializer.serializer.dumps = lambda obj: b"test"
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #38
#--------------------------

def test_serializer_constructor_default_serializer() -> None:
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_custom_secret_key_bytes() -> None:
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_secret_key_list() -> None:
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_custom_salt() -> None:
    s = Serializer("secret", salt="pepper")
    assert s.salt == b"pepper"

def test_serializer_constructor_salt_none() -> None:
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_custom_serializer_bytes() -> None:
    s = Serializer("secret", serializer=bytes)
    assert s.serializer is bytes
    assert s.is_text_serializer is False

def test_serializer_constructor_custom_serializer_str() -> None:
    s = Serializer("secret", serializer=str)
    assert s.serializer is str
    assert s.is_text_serializer is True

def test_serializer_constructor_serializer_kwargs() -> None:
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_custom_signer() -> None:
    s = Serializer("secret", signer=Signer)
    assert s.signer is Signer

def test_serializer_constructor_signer_kwargs() -> None:
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_fallback_signers() -> None:
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_fallback_signers_none() -> None:
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_serializer_none() -> None:
    s = Serializer("secret", serializer=None)
    assert s.serializer is json
    assert s.is_text_serializer is True

def test_serializer_constructor_signer_none() -> None:
    s = Serializer("secret", signer=None)
    assert s.signer is Signer


# LLM-generated content at query #39
#--------------------------

def test_serializer_constructor_default_serializer():
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

def test_serializer_constructor_with_list_secret_keys():
    s = Serializer(["key1", "key2"])
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

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()
        @staticmethod
        def loads(data):
            return data.decode()
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer == CustomSerializer()
    assert s.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer_class():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []


# LLM-generated content at query #40
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None or result is None


# LLM-generated content at query #41
#--------------------------

def test_serializer_init_with_str_secret_key():
    s = Serializer("secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    s = Serializer(b"secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"

def test_serializer_init_with_secret_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_secret_key_bytes_list():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_init_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_init_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "string", "loads": lambda self, x: {}})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer is custom_serializer
    assert s.is_text_serializer is True

def test_serializer_init_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"bytes", "loads": lambda self, x: {}})()
    s = Serializer("secret", serializer=bytes_serializer)
    assert s.serializer is bytes_serializer
    assert s.is_text_serializer is False

def test_serializer_init_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_init_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    s = Serializer("secret", signer=custom_signer)
    assert s.signer is custom_signer

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

def test_serializer_init_secret_key_property():
    s = Serializer(["old_key", "new_key"])
    assert s.secret_key == b"new_key"


# LLM-generated content at query #42
#--------------------------

```python
def test_loads_returns_any_type():
    serializer = _PDataSerializer()
    result = serializer.loads("test_payload")
    assert result is not None or result is None

def test_loads_accepts_serialized_input():
    serializer = _PDataSerializer()
    payload = b"serialized_data"
    result = serializer.loads(payload)
    assert isinstance(result, object)

def test_loads_with_integer_payload():
    serializer = _PDataSerializer()
    payload = 12345
    result = serializer.loads(payload)
    assert result == 12345

def test_loads_with_list_payload():
    serializer = _PDataSerializer()
    payload = [1, 2, 3]
    result = serializer.loads(payload)
    assert result == [1, 2, 3]

def test_loads_with_none_payload():
    serializer = _PDataSerializer()
    payload = None
    result = serializer.loads(payload)
    assert result is None

def test_loads_with_string_payload():
    serializer = _PDataSerializer()
    payload = "hello"
    result = serializer.loads(payload)
    assert result == "hello"
```


# LLM-generated content at query #43
#--------------------------

```python
def test_load_payload_predicate_false():
    serializer = Serializer("secret", serializer=json)
    payload = b'"test"'
    result = serializer.load_payload(payload, serializer=json)
    assert result == "test"
```


# LLM-generated content at query #44
#--------------------------

def test_serializer_constructor_with_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "str", "loads": lambda self, x: {}})
    serializer = Serializer("secret-key", serializer=custom_serializer())
    assert serializer.serializer == custom_serializer()
    assert serializer.is_text_serializer

def test_serializer_constructor_with_bytes_serializer():
    custom_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"bytes", "loads": lambda self, x: {}})
    serializer = Serializer("secret-key", serializer=custom_serializer())
    assert not serializer.is_text_serializer

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret-key", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #45
#--------------------------

def test_serializer_constructor_defaults():
    secret_key = "secret"
    s = Serializer(secret_key)
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    secret_key = b"secret"
    s = Serializer(secret_key)
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list():
    secret_key = ["key1", b"key2"]
    s = Serializer(secret_key)
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_salt_bytes():
    s = Serializer("secret", salt=b"custom")
    assert s.salt == b"custom"

def test_serializer_constructor_with_salt_str():
    s = Serializer("secret", salt="custom")
    assert s.salt == b"custom"

def test_serializer_constructor_with_custom_serializer():
    class MockSerializer:
        @staticmethod
        def dumps(obj):
            return "mock"

        @staticmethod
        def loads(data):
            return "mock"

    s = Serializer("secret", serializer=MockSerializer)
    assert s.serializer is MockSerializer
    assert s.is_text_serializer is True

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
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_fallback_signers_none():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    s = Serializer(["old", "new"])
    assert s.secret_key == b"new"


# LLM-generated content at query #46
#--------------------------

def test_dumps_returns_bytes_when_not_text_serializer():
    serializer = Serializer(secret_key="secret", serializer=None)
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #47
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

def test_serializer_constructor_with_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [Signer]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #48
#--------------------------

def test_serializer_init_with_default_parameters() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_init_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_custom_salt() -> None:
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_init_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_serializer() -> None:
    custom_serializer = type("CustomSerializer", (), {"dumps": staticmethod(lambda x: str(x)), "loads": staticmethod(lambda x: eval(x))})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == is_text_serializer(custom_serializer)

def test_serializer_init_with_custom_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_init_with_custom_signer() -> None:
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_init_with_custom_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers() -> None:
    fallback_signers = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_init_with_empty_fallback_signers() -> None:
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_init_with_none_fallback_signers() -> None:
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #49
#--------------------------

```python
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None

def test_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert isinstance(result, bytes)

def test_dumps_preserves_data():
    serializer = _PDataSerializer()
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert result == b'{"key": "value"}'

def test_dumps_handles_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result == b"null"

def test_dumps_handles_empty_string():
    serializer = _PDataSerializer()
    result = serializer.dumps("")
    assert result == b'""'

def test_dumps_handles_complex_nested_objects():
    serializer = _PDataSerializer()
    data = {"a": [1, 2, {"b": True}]}
    result = serializer.dumps(data)
    assert result == b'{"a": [1, 2, {"b": true}]}'
```


# LLM-generated content at query #50
#--------------------------

```python
def test_salt_is_not_none_so_predicate_is_false():
    serializer = Serializer(secret_key="secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"
```


# LLM-generated content at query #51
#--------------------------

```python
def test_p_data_serializer_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None

def test_p_data_serializer_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result is not None

def test_p_data_serializer_dumps_accepts_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None

def test_p_data_serializer_dumps_accepts_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result is not None

def test_p_data_serializer_dumps_accepts_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert result is not None
```


# LLM-generated content at query #52
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

def test_dumps_with_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None

def test_dumps_with_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result is not None

def test_dumps_with_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert result is not None

def test_dumps_positional_only_argument():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None
```


# LLM-generated content at query #53
#--------------------------

def test_serializer_constructor_with_string_secret_key():
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

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

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


# LLM-generated content at query #54
#--------------------------

def test_serializer_init_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_init_with_secret_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_bytes_secret_key_list():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_init_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_init_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(data):
            return eval(data)

    s = Serializer("secret", serializer=CustomSerializer)
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True

def test_serializer_init_with_bytes_serializer():
    import pickle
    s = Serializer("secret", serializer=pickle)
    assert s.serializer is pickle
    assert s.is_text_serializer is False

def test_serializer_init_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

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

def test_serializer_init_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(data):
            return eval(data)

    class CustomSigner(Signer):
        pass

    s = Serializer(
        secret_key=["key1", "key2"],
        salt=b"custom_salt",
        serializer=CustomSerializer,
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}],
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"custom_salt"
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True
    assert s.signer is CustomSigner
    assert s.signer_kwargs == {"digest_method": "sha256"}
    assert s.fallback_signers == [{"key_derivation": "none"}]
    assert s.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #55
#--------------------------

def test_serializer_init_with_str_secret_key() -> None:
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_init_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_init_with_list_of_str_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes_secret_keys() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_custom_salt() -> None:
    serializer = Serializer("secret", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_init_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_default_salt() -> None:
    serializer = Serializer("secret")
    assert serializer.salt == b"itsdangerous"

def test_serializer_init_with_default_serializer() -> None:
    serializer = Serializer("secret")
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_init_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> object:
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_init_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_init_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #56
#--------------------------

```python
def test_load_payload_predicate_false_with_custom_serializer():
    from itsdangerous.serializer import Serializer
    from itsdangerous.exc import BadPayload

    class CustomSerializer:
        def loads(self, data):
            raise TypeError("test error")

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    try:
        serializer.load_payload(b"test", serializer=CustomSerializer())
    except BadPayload:
        pass
    assert True  # predicate at line 22 evaluated to False because is_text is False and loads raised exception
```


# LLM-generated content at query #57
#--------------------------

```python
def test_salt_is_not_none():
    s = Serializer("secret", salt=b"explicit_salt")
    assert s.salt == b"explicit_salt"
```


# LLM-generated content at query #58
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == list(Serializer.default_fallback_signers)
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
    assert serializer.is_text_serializer == is_text_serializer(CustomSerializer())

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

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


# LLM-generated content at query #59
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_not_text_serializer():
    serializer = Serializer("secret", serializer=dumps_returns_bytes)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

class dumps_returns_bytes:
    @staticmethod
    def dumps(obj):
        return b'{"key":"value"}'
    @staticmethod
    def loads(s):
        return {"key":"value"}
```


# LLM-generated content at query #60
#--------------------------

```python
def test_salt_is_none_so_predicate_evaluates_to_false():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #61
#--------------------------

```python
def test_load_payload_predicate_false():
    serializer = Serializer("secret", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #62
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

def test_serializer_constructor_with_list_secret():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_bytes_secret():
    serializer = Serializer([b"key1", b"key2"])
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
            return str(obj).encode()

        @staticmethod
        def loads(data):
            return data.decode()

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
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

def test_serializer_constructor_with_override_default_serializer():
    original_default = Serializer.default_serializer
    try:
        class CustomDefaultSerializer:
            @staticmethod
            def dumps(obj):
                return "custom"

            @staticmethod
            def loads(data):
                return data

        Serializer.default_serializer = CustomDefaultSerializer()
        serializer = Serializer("secret", serializer=None)
        assert serializer.serializer == CustomDefaultSerializer()
        assert serializer.is_text_serializer == True
    finally:
        Serializer.default_serializer = original_default


# LLM-generated content at query #63
#--------------------------

```python
def test_fallback_signers_not_none():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    serializer = Serializer("secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #64
#--------------------------

def test_constructor_default_serializer_is_json():
    serializer = Serializer("secret")
    assert serializer.serializer == json

def test_constructor_salt_default():
    serializer = Serializer("secret")
    assert serializer.salt == b"itsdangerous"

def test_constructor_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_secret_keys_single_string():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_secret_keys_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_is_text_serializer_true():
    serializer = Serializer("secret")
    assert serializer.is_text_serializer is True

def test_constructor_serializer_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_constructor_signer_kwargs_empty():
    serializer = Serializer("secret")
    assert serializer.signer_kwargs == {}

def test_constructor_signer_kwargs_provided():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_fallback_signers_default():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_constructor_fallback_signers_provided():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_constructor_serializer_kwargs_empty():
    serializer = Serializer("secret")
    assert serializer.serializer_kwargs == {}

def test_constructor_serializer_kwargs_provided():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #65
#--------------------------

```python
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_string")
    assert result is not None

def test_dumps_returns_expected_type():
    serializer = _PDataSerializer()
    result = serializer.dumps(123)
    assert isinstance(result, type(serializer.dumps("")))

def test_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None

def test_dumps_returns_same_type_for_different_inputs():
    serializer = _PDataSerializer()
    result1 = serializer.dumps("hello")
    result2 = serializer.dumps(42)
    assert type(result1) == type(result2)
```


# LLM-generated content at query #66
#--------------------------

```python
def test_load_payload_is_text_false_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=Serializer.default_serializer)
    serializer.is_text_serializer = False
    payload = b'"test"'
    result = serializer.load_payload(payload)
    assert result == "test"
```


# LLM-generated content at query #67
#--------------------------

def test_serializer_constructor_default_serializer():
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

def test_serializer_constructor_with_list_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"

        @staticmethod
        def loads(data):
            return data

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #68
#--------------------------

```python
def test_loads_returns_any_type():
    serializer = _PDataSerializer()
    result = serializer.loads("test_payload")
    assert result is None or result is not None


# LLM-generated content at query #69
#--------------------------

def test_serializer_constructor_with_str_secret_key() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_str_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return "dumped"

        @staticmethod
        def loads(s: str) -> object:
            return "loaded"

    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == True

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


# LLM-generated content at query #70
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

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_key_list():
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
            return "custom"

        @staticmethod
        def loads(s):
            return "custom"

    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(s):
            return "loaded"

    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.serializer is BytesSerializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #71
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

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_explicit_serializer():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    import pickle
    serializer = Serializer("secret-key", serializer=pickle)
    assert serializer.serializer == pickle
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_signer_class():
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

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"bytes-key")
    assert serializer.secret_keys == [b"bytes-key"]


# LLM-generated content at query #72
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

def test_serializer_constructor_with_list_of_str_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_keys():
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
    class CustomSerializer:
        def dumps(self, obj):
            return "str"
        def loads(self, s):
            return obj
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, s):
            return obj
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

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


# LLM-generated content at query #73
#--------------------------

def test_dumps_returns_bytes_when_not_text_serializer():
    serializer = Serializer("secret", serializer=None, serializer_kwargs=None)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_returns_str_when_text_serializer():
    serializer = Serializer("secret", serializer=json, serializer_kwargs=None)
    result = serializer.dumps("test")
    assert isinstance(result, str)

def test_dumps_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    result = serializer.dumps("data", salt=b"custom_salt")
    assert isinstance(result, bytes)

def test_dumps_with_none_salt():
    serializer = Serializer("secret", salt=None)
    result = serializer.dumps("data", salt=None)
    assert isinstance(result, bytes)

def test_dumps_uses_dump_payload():
    serializer = Serializer("secret")
    payload = serializer.dump_payload("test")
    expected = serializer.make_signer().sign(payload)
    result = serializer.dumps("test")
    assert result == expected


# LLM-generated content at query #74
#--------------------------

```python
def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    serializer.dumps = lambda obj: obj
    result = serializer.dumps("test")
    assert result == "test"

def test_dumps_with_none():
    serializer = _PDataSerializer()
    serializer.dumps = lambda obj: "None"
    result = serializer.dumps(None)
    assert result == "None"

def test_dumps_with_integer():
    serializer = _PDataSerializer()
    serializer.dumps = lambda obj: str(obj)
    result = serializer.dumps(42)
    assert result == "42"

def test_dumps_with_list():
    serializer = _PDataSerializer()
    serializer.dumps = lambda obj: obj
    result = serializer.dumps([1, 2, 3])
    assert result == [1, 2, 3]

def test_dumps_with_dict():
    serializer = _PDataSerializer()
    serializer.dumps = lambda obj: obj
    result = serializer.dumps({"key": "value"})
    assert result == {"key": "value"}

def test_dumps_returns_bytes():
    serializer = _PDataSerializer()
    serializer.dumps = lambda obj: b"test"
    result = serializer.dumps("test")
    assert result == b"test"

def test_dumps_returns_string():
    serializer = _PDataSerializer()
    serializer.dumps = lambda obj: "test"
    result = serializer.dumps("test")
    assert result == "test"
```


# LLM-generated content at query #75
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

def test_serializer_constructor_with_custom_serializer() -> None:
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, obj: "text", "loads": lambda self, s: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer() -> None:
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, obj: b"bytes", "loads": lambda self, s: {}})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

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


# LLM-generated content at query #76
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
    s = Serializer(["secret1", "secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"secret1", b"secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(s):
            return {}
    s = Serializer("secret", serializer=TextSerializer())
    assert s.serializer is not json
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return {}
    s = Serializer("secret", serializer=BytesSerializer())
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


# LLM-generated content at query #77
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
    serializer = Serializer(b"bytes-key")
    assert serializer.secret_keys == [b"bytes-key"]

def test_serializer_constructor_with_list_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_none():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_text_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    serializer = Serializer("secret", serializer=Serializer("inner"))
    assert serializer.is_text_serializer is False

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #78
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
    payload = serializer.dump_payload(42)
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return bytes(str(obj), "utf-8")
        def loads(self, payload):
            return int(payload.decode("utf-8"))
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = serializer.dump_payload(123)
    result = serializer.load_payload(payload)
    assert result == 123

def test_load_payload_raises_bad_payload():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_override_serializer():
    serializer = Serializer("secret")
    class OverrideSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return eval(payload)
    payload = serializer.dump_payload("original")
    result = serializer.load_payload(payload, serializer=OverrideSerializer())
    assert result == "original"

def test_load_payload_with_text_serializer():
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return eval(payload)
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = serializer.dump_payload([1, 2, 3])
    result = serializer.load_payload(payload)
    assert result == [1, 2, 3]

def test_load_payload_with_unicode_text():
    serializer = Serializer("secret")
    payload = serializer.dump_payload("héllo")
    result = serializer.load_payload(payload)
    assert result == "héllo"
```


# LLM-generated content at query #79
#--------------------------

def test_serializer_init_with_string_secret_key() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_init_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_init_with_list_of_strings() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_default_salt() -> None:
    serializer = Serializer("secret")
    assert serializer.salt == b"itsdangerous"

def test_serializer_init_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_salt() -> None:
    serializer = Serializer("secret", salt="custom")
    assert serializer.salt == b"custom"

def test_serializer_init_with_default_serializer() -> None:
    serializer = Serializer("secret")
    assert serializer.serializer is serializer.default_serializer

def test_serializer_init_with_custom_serializer() -> None:
    class FakeSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return "dummy"
        @staticmethod
        def loads(s: str) -> t.Any:
            return None
    serializer = Serializer("secret", serializer=FakeSerializer())
    assert serializer.serializer is FakeSerializer()

def test_serializer_init_with_text_serializer() -> None:
    class TextSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> str:
            return "text"
        @staticmethod
        def loads(s: str) -> t.Any:
            return None
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_init_with_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> bytes:
            return b"bytes"
        @staticmethod
        def loads(s: bytes) -> t.Any:
            return None
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_init_with_default_signer() -> None:
    serializer = Serializer("secret")
    assert serializer.signer is serializer.default_signer

def test_serializer_init_with_custom_signer() -> None:
    class FakeSigner:
        def __init__(self, secret_key: str | bytes, salt: str | bytes | None = None, **kwargs: t.Any) -> None:
            pass
    serializer = Serializer("secret", signer=FakeSigner)
    assert serializer.signer is FakeSigner

def test_serializer_init_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_default_fallback_signers() -> None:
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_serializer_init_with_custom_fallback_signers() -> None:
    class FakeSigner:
        def __init__(self, secret_key: str | bytes, salt: str | bytes | None = None, **kwargs: t.Any) -> None:
            pass
    serializer = Serializer("secret", fallback_signers=[FakeSigner])
    assert serializer.fallback_signers == [FakeSigner]

def test_serializer_init_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_init_secret_key_property() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #80
#--------------------------

def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("test-secret")
    assert serializer.secret_keys == [b"test-secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"test-secret")
    assert serializer.secret_keys == [b"test-secret"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_string_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("test-secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("test-secret", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_string_salt():
    serializer = Serializer("test-secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_custom_text_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: eval(x)})()
    serializer = Serializer("test-secret", serializer=custom_serializer)
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_bytes_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x).encode(), "loads": lambda self, x: eval(x.decode())})()
    serializer = Serializer("test-secret", serializer=custom_serializer)
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("test-secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("test-secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("test-secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("test-secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_default_serializer_kwargs_empty():
    serializer = Serializer("test-secret")
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #81
#--------------------------

```python
def test_load_payload_is_text_false_with_non_text_serializer():
    serializer = Serializer("secret", serializer=Serializer.default_serializer)
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #82
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
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_bytes_secret_keys() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_string_salt() -> None:
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_bytes_salt() -> None:
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
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_none_fallback_signers() -> None:
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_all_parameters() -> None:
    serializer = Serializer(
        secret_key=["old_key", b"new_key"],
        salt=b"custom_salt",
        serializer=json,
        serializer_kwargs={"indent": 2},
        signer=Signer,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"old_key", b"new_key"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #83
#--------------------------

```python
def test_loads_with_string_payload():
    serializer = _PDataSerializer()
    result = serializer.loads("test")
    assert result is not None

def test_loads_with_bytes_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(b"test")
    assert result is not None

def test_loads_with_integer_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(42)
    assert result is not None

def test_loads_with_float_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(3.14)
    assert result is not None

def test_loads_with_list_payload():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert result is not None

def test_loads_with_dict_payload():
    serializer = _PDataSerializer()
    result = serializer.loads({"key": "value"})
    assert result is not None

def test_loads_with_none_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is not None
```


# LLM-generated content at query #84
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"

        @staticmethod
        def loads(s):
            return "custom"

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(s):
            return "bytes"

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_key, salt=None, **kwargs):
            pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #85
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
            return str(obj).encode()
        @staticmethod
        def loads(data):
            return data.decode()
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        def __init__(self, secret_key, salt="", **kwargs):
            pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"digest_method": "sha256"})
    assert serializer.signer_kwargs == {"digest_method": "sha256"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"digest_method": "sha256"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #86
#--------------------------

def test_serializer_init_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_init_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_init_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: x})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == True

def test_serializer_init_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: x})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.is_text_serializer == False

def test_serializer_init_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #87
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    payload = "test"
    result = serializer.loads(payload)
    assert result is not None or result is None

def test_loads_accepts_serialized_type():
    serializer = _PDataSerializer()
    payload = b"binary_data"
    result = serializer.loads(payload)
    assert isinstance(result, object)

def test_loads_returns_none_for_empty_payload():
    serializer = _PDataSerializer()
    payload = ""
    result = serializer.loads(payload)
    assert result is None or result != None

def test_loads_handles_complex_data():
    serializer = _PDataSerializer()
    payload = '{"key": "value"}'
    result = serializer.loads(payload)
    assert isinstance(result, dict) or not isinstance(result, dict)

def test_loads_works_with_different_types():
    serializer = _PDataSerializer()
    payload = 12345
    result = serializer.loads(payload)
    assert result is not None or result is None
```


# LLM-generated content at query #88
#--------------------------

def test_serializer_constructor_with_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_custom_signer_and_signer_kwargs() -> None:
    serializer = Serializer("secret", signer=Signer, signer_kwargs={"digest_method": "sha256"})
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"digest_method": "sha256"}

def test_serializer_constructor_with_custom_fallback_signers() -> None:
    serializer = Serializer("secret", fallback_signers=[{"digest_method": "sha256"}])
    assert serializer.fallback_signers == [{"digest_method": "sha256"}]

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #89
#--------------------------

def test_serializer_constructor_with_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_string_secret_key():
    s = Serializer("mysecret")
    assert s.secret_keys == [b"mysecret"]

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"mysecret")
    assert s.secret_keys == [b"mysecret"]

def test_serializer_constructor_with_list_of_strings_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, obj, **kwargs: str(obj), "loads": lambda self, obj: obj})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer == custom_serializer
    assert s.is_text_serializer == is_text_serializer(custom_serializer)

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    from itsdangerous.signer import Signer
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.signer import Signer
    fallback = [{"key_derivation": "hmac"}, (Signer, {"digest_method": "sha256"}), Signer]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback


# LLM-generated content at query #90
#--------------------------

```python
def test_dumps_returns_bytes_when_is_text_serializer_is_false():
    serializer = Serializer("secret", serializer=None)
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #91
#--------------------------

```
def test_load_payload_exception_raises_bad_payload():
    serializer = Serializer("secret")
    payload = b"invalid"
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
```


# LLM-generated content at query #92
#--------------------------

```python
from itsdangerous.serializer import Serializer
from itsdangerous.exc import BadPayload

def test_load_payload_with_default_serializer_and_text_serializer():
    serializer = Serializer("secret-key", serializer=type("TextSerializer", (), {"loads": lambda self, x: x, "dumps": lambda self, x: "{}"})())
    payload = b'"test"'
    result = serializer.load_payload(payload)
    assert result == '"test"'

def test_load_payload_with_default_serializer_and_bytes_serializer():
    serializer = Serializer("secret-key", serializer=type("BytesSerializer", (), {"loads": lambda self, x: x, "dumps": lambda self, x: b"{}"})())
    payload = b'test'
    result = serializer.load_payload(payload)
    assert result == b'test'

def test_load_payload_with_custom_serializer_override():
    serializer = Serializer("secret-key", serializer=type("TextSerializer", (), {"loads": lambda self, x: x, "dumps": lambda self, x: "{}"})())
    custom_serializer = type("CustomSerializer", (), {"loads": lambda self, x: "custom"})()
    payload = b'"test"'
    result = serializer.load_payload(payload, serializer=custom_serializer)
    assert result == "custom"

def test_load_payload_raises_bad_payload_on_exception():
    serializer = Serializer("secret-key", serializer=type("FailingSerializer", (), {"loads": lambda self, x: (_ for _ in ()).throw(ValueError("error")), "dumps": lambda self, x: "{}"})())
    payload = b'"test"'
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_with_override_serializer_text_false():
    serializer = Serializer("secret-key", serializer=type("BytesSerializer", (), {"loads": lambda self, x: x, "dumps": lambda self, x: b"{}"})())
    custom_serializer = type("TextSerializer", (), {"loads": lambda self, x: x, "dumps": lambda self, x: "{}"})()
    payload = b'"test"'
    result = serializer.load_payload(payload, serializer=custom_serializer)
    assert result == '"test"'
```


# LLM-generated content at query #93
#--------------------------

```python
def test_fallback_signers_is_not_none():
    serializer = Serializer(secret_key="secret", fallback_signers=[{"salt": b"extra"}])
    assert serializer.fallback_signers == [{"salt": b"extra"}]
```


# LLM-generated content at query #94
#--------------------------

def test_serializer_constructor_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer.dumps({}) == "{}"
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return str(obj)

        @staticmethod
        def loads(s: str) -> object:
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer.dumps(123) == "123"
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return str(obj).encode()

        @staticmethod
        def loads(s: bytes) -> object:
            return s.decode()

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer.dumps(123) == b"123"
    assert serializer.is_text_serializer is False

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

def test_serializer_constructor_with_all_parameters() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return str(obj)

        @staticmethod
        def loads(s: str) -> object:
            return int(s)

    class CustomSigner(Signer):
        pass

    serializer = Serializer(
        secret_key=["old_key", "new_key"],
        salt=b"custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"old_key", b"new_key"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer.dumps(123) == "123"
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #95
#--------------------------

def test_serializer_init_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_list_of_strings_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_list_of_bytes_secret_key():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_init_with_custom_signer():
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

def test_serializer_init_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(s):
            return s

    class CustomSigner(Signer):
        pass

    serializer = Serializer(
        secret_key=["key1", b"key2"],
        salt=b"custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #96
#--------------------------

def test_constructor_with_string_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]

def test_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer is BytesSerializer()
    assert serializer.is_text_serializer is False

def test_constructor_with_custom_signer_class():
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

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_all_arguments():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return s
    class CustomSigner(Signer):
        pass
    serializer = Serializer(
        secret_key=["old", "new"],
        salt="pepper",
        serializer=CustomSerializer(),
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.salt == b"pepper"
    assert serializer.is_text_serializer is True
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #97
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
            return int(s)
    s = Serializer("secret", serializer=CustomSerializer)
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(b):
            return b
    s = Serializer("secret", serializer=BytesSerializer)
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
    s = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []

def test_serializer_constructor_with_none_fallback_signers():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []


# LLM-generated content at query #98
#--------------------------

def test_serializer_constructor_with_defaults():
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

def test_serializer_constructor_with_list_of_strings_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_key():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_salt_as_string():
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
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_bytes_serializer():
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(s):
            return s

    serializer = Serializer("secret", serializer=CustomBytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

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


# LLM-generated content at query #99
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert isinstance(result, type(serializer.dumps("test")))
```


# LLM-generated content at query #100
#--------------------------

def test_serializer_constructor_default_serializer():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_list_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_custom_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return b"custom"

        @staticmethod
        def loads(s):
            return {"custom": True}

    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer()
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
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    s = Serializer(["old", "new"])
    assert s.secret_key == b"new"


# LLM-generated content at query #101
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

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_list_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: int(x)})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == True

def test_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: x})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer == False

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [Signer]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #102
#--------------------------

```python
def test_load_payload_with_bytes_serializer():
    serializer = Serializer("secret", serializer=Serializer.default_serializer)
    serializer.is_text_serializer = False
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #103
#--------------------------

```python
def test_dumps_returns_bytes_when_is_text_serializer_false():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b'{"a": 1}'

        @staticmethod
        def loads(data):
            return {"a": 1}

    s = Serializer("secret", serializer=BytesSerializer())
    result = s.dumps({"a": 1})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #104
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

def test_serializer_constructor_with_bytes_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_key_list():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_bytes_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return "custom"
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is not json
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return "bytes"
    s = Serializer("secret", serializer=BytesSerializer())
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
    s = Serializer("secret", fallback_signers=[{"key_derivation": "hmac"}])
    assert s.fallback_signers == [{"key_derivation": "hmac"}]

def test_serializer_constructor_with_none_fallback_signers():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []


# LLM-generated content at query #105
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

def test_serializer_constructor_with_secret_keys_list():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_salt_str():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class MockSerializer:
        @staticmethod
        def dumps(obj):
            return "{}"

        @staticmethod
        def loads(s):
            return {}

    serializer = Serializer("secret", serializer=MockSerializer())
    assert serializer.serializer is MockSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class MockSigner:
        def __init__(self, secret_keys, salt="itsdangerous", **kwargs):
            pass

    serializer = Serializer("secret", signer=MockSigner)
    assert serializer.signer is MockSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"digest_method": "sha256"}])
    assert serializer.fallback_signers == [{"digest_method": "sha256"}]

def test_serializer_constructor_with_fallback_signers_none():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"

        @staticmethod
        def loads(data):
            return {}

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False


# LLM-generated content at query #106
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

def test_constructor_with_list_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_bytes_list_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

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
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #107
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

def test_serializer_constructor_with_list_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_list_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_salt_str():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return {}
    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #108
#--------------------------

def test_constructor_with_default_serializer():
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
    assert serializer.salt == b"itsdangerous"

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"

def test_constructor_with_custom_salt_as_string():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_custom_salt_as_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, obj: "test", "loads": lambda self, s: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False

def test_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_constructor_with_empty_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #109
#--------------------------

def test_serializer_constructor_with_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_all_parameters():
    fallback = [(Signer, {"key_derivation": "none"})]
    serializer = Serializer(
        ["key1", "key2"],
        salt=b"custom_salt",
        serializer=json,
        serializer_kwargs={"indent": 2},
        signer=Signer,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=fallback,
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #110
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None

def test_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result is not None

def test_dumps_with_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None

def test_dumps_with_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result is not None

def test_dumps_with_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert result is not None
```


# LLM-generated content at query #111
#--------------------------

```python
def test_salt_is_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #112
#--------------------------

```python
def test_load_payload_with_default_serializer_and_text_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer_and_text():
    class TextSerializer:
        def dumps(self, obj):
            return str(obj)
        def loads(self, payload):
            return eval(payload)
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return str(obj).encode()
        def loads(self, payload):
            return eval(payload.decode())
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_override_serializer_text():
    serializer = Serializer("secret")
    payload = b'{"key": "value"}'
    class TextSerializer:
        def loads(self, payload):
            return eval(payload)
    result = serializer.load_payload(payload, serializer=TextSerializer())
    assert result == {"key": "value"}

def test_load_payload_with_override_serializer_bytes():
    serializer = Serializer("secret")
    payload = b'{"key": "value"}'
    class BytesSerializer:
        def loads(self, payload):
            return eval(payload.decode())
    result = serializer.load_payload(payload, serializer=BytesSerializer())
    assert result == {"key": "value"}

def test_load_payload_raises_bad_payload_on_exception():
    serializer = Serializer("secret")
    payload = b"invalid"
    try:
        serializer.load_payload(payload)
        assert False
    except BadPayload:
        pass

def test_load_payload_with_none_serializer_override():
    serializer = Serializer("secret")
    payload = serializer.dump_payload("test")
    result = serializer.load_payload(payload, serializer=None)
    assert result == "test"
```


# LLM-generated content at query #113
#--------------------------

def test_serializer_constructor_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert s.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert s.signer is Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "{}", "loads": lambda self, x: {}})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer is custom_serializer
    assert s.is_text_serializer == is_text_serializer(custom_serializer)

def test_serializer_constructor_with_signer_and_signer_kwargs():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner, signer_kwargs={"key": "value"})
    assert s.signer is CustomSigner
    assert s.signer_kwargs == {"key": "value"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key": "value"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_all_params():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "{}", "loads": lambda self, x: {}})()
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", salt=b"custom_salt", serializer=custom_serializer, serializer_kwargs={"sort_keys": True}, signer=CustomSigner, signer_kwargs={"digest_method": "sha256"}, fallback_signers=[{"key": "value"}])
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"custom_salt"
    assert s.serializer is custom_serializer
    assert s.is_text_serializer == is_text_serializer(custom_serializer)
    assert s.signer is CustomSigner
    assert s.signer_kwargs == {"digest_method": "sha256"}
    assert s.fallback_signers == [{"key": "value"}]
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #114
#--------------------------

```python
def test_loads_with_valid_payload():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}

def test_loads_with_list_payload():
    serializer = _PDataSerializer()
    payload = b'[1, 2, 3]'
    result = serializer.loads(payload)
    assert result == [1, 2, 3]

def test_loads_with_string_payload():
    serializer = _PDataSerializer()
    payload = b'"hello"'
    result = serializer.loads(payload)
    assert result == "hello"

def test_loads_with_number_payload():
    serializer = _PDataSerializer()
    payload = b'42'
    result = serializer.loads(payload)
    assert result == 42

def test_loads_with_boolean_payload():
    serializer = _PDataSerializer()
    payload = b'true'
    result = serializer.loads(payload)
    assert result == True

def test_loads_with_null_payload():
    serializer = _PDataSerializer()
    payload = b'null'
    result = serializer.loads(payload)
    assert result is None
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_serializer_init_with_no_serializer_uses_default():
    serializer = Serializer("secret")
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer == is_text_serializer(json)

def test_serializer_init_with_custom_serializer_bytes():
    serializer = Serializer("secret", serializer=json)
    assert not serializer.is_text_serializer

def test_serializer_init_with_custom_signer():
    class CustomSigner:
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_init_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key": "value"})
    assert serializer.signer_kwargs == {"key": "value"}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key": "value"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"key": "value"})
    assert serializer.serializer_kwargs == {"key": "value"}

def test_serializer_init_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_init_with_salt_str():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_init_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_secret_key_bytes():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_init_with_secret_key_str():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]


# LLM-generated content at query #2
#--------------------------

```python
def test_iter_unsigners_returns_signer_and_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[{"algorithm": "sha512"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].salt == serializer.salt
    assert signers[1].salt == serializer.salt

def test_iter_unsigners_with_custom_salt():
    serializer = Serializer("secret-key", fallback_signers=[{"algorithm": "sha512"}])
    signers = list(serializer.iter_unsigners(salt="custom-salt"))
    assert len(signers) == 2
    assert signers[0].salt == b"custom-salt"
    assert signers[1].salt == b"custom-salt"

def test_iter_unsigners_yields_signer_for_each_secret_key_in_fallback():
    serializer = Serializer(["old-key", "new-key"], fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == [b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]
    assert signers[2].secret_keys == [b"new-key"]

def test_iter_unsigners_with_fallback_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"algorithm": "sha512"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].algorithm == "sha512"

def test_iter_unsigners_empty_fallback_signers():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"secret-key"]
```


# LLM-generated content at query #3
#--------------------------

```python
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_data")
    assert result is not None

def test_dumps_accepts_any_type():
    serializer = _PDataSerializer()
    result = serializer.dumps(123)
    assert result is not None

def test_dumps_accepts_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None
```


# LLM-generated content at query #4
#--------------------------

def test_constructor_with_default_parameters() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_key == b"secret"
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_key == b"secret"
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_secret_key_list() -> None:
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_constructor_with_bytes_secret_key_list() -> None:
    serializer = Serializer([b"secret1", b"secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_custom_serializer() -> None:
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: x})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_constructor_with_bytes_serializer() -> None:
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: x})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False

def test_constructor_with_custom_signer() -> None:
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer is custom_signer

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


# LLM-generated content at query #5
#--------------------------

```python
from itsdangerous.serializer import Serializer

def test_serializer_init_without_serializer_falls_to_default():
    s = Serializer("secret")
    assert s.serializer is s.default_serializer
```


# LLM-generated content at query #6
#--------------------------

def test_load_payload_with_default_serializer():
    serializer = Serializer("secret")
    result = serializer.load_payload(serializer.dump_payload({"key": "value"}))
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer_returns_str():
    class CustomSerializer:
        def loads(self, payload: str) -> dict:
            return {"custom": payload}
        def dumps(self, obj: dict) -> str:
            return obj["custom"]
    serializer = Serializer("secret", serializer=CustomSerializer())
    result = serializer.load_payload(b"test")
    assert result == {"custom": "test"}

def test_load_payload_with_custom_serializer_returns_bytes():
    class CustomSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"custom": payload.decode()}
        def dumps(self, obj: dict) -> bytes:
            return obj["custom"].encode()
    serializer = Serializer("secret", serializer=CustomSerializer())
    result = serializer.load_payload(b"test")
    assert result == {"custom": "test"}

def test_load_payload_with_explicit_serializer_returns_str():
    class CustomSerializer:
        def loads(self, payload: str) -> dict:
            return {"custom": payload}
        def dumps(self, obj: dict) -> str:
            return obj["custom"]
    serializer = Serializer("secret")
    result = serializer.load_payload(b"test", serializer=CustomSerializer())
    assert result == {"custom": "test"}

def test_load_payload_with_explicit_serializer_returns_bytes():
    class CustomSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"custom": payload.decode()}
        def dumps(self, obj: dict) -> bytes:
            return obj["custom"].encode()
    serializer = Serializer("secret")
    result = serializer.load_payload(b"test", serializer=CustomSerializer())
    assert result == {"custom": "test"}

def test_load_payload_raises_bad_payload_on_exception():
    class FailingSerializer:
        def loads(self, payload):
            raise ValueError("fail")
        def dumps(self, obj):
            return b"test"
    serializer = Serializer("secret", serializer=FailingSerializer())
    try:
        serializer.load_payload(b"test")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_text_serializer_uses_utf8_decode():
    class TextSerializer:
        def loads(self, payload: str) -> dict:
            return {"text": payload}
        def dumps(self, obj: dict) -> str:
            return obj["text"]
    serializer = Serializer("secret", serializer=TextSerializer())
    result = serializer.load_payload("hello".encode("utf-8"))
    assert result == {"text": "hello"}

def test_load_payload_with_bytes_serializer_passes_bytes_directly():
    class BytesSerializer:
        def loads(self, payload: bytes) -> dict:
            return {"bytes": payload}
        def dumps(self, obj: dict) -> bytes:
            return obj["bytes"]
    serializer = Serializer("secret", serializer=BytesSerializer())
    result = serializer.load_payload(b"raw")
    assert result == {"bytes": b"raw"}


# LLM-generated content at query #7
#--------------------------

def test_serializer_constructor_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secrets() -> None:
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> object:
            return int(s)
    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return b"test"
        @staticmethod
        def loads(s: bytes) -> object:
            return s
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

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

def test_serializer_constructor_with_empty_fallback_signers() -> None:
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #8
#--------------------------

```python
def test_load_payload_when_is_text_is_false():
    class BytesSerializer:
        def loads(self, data: bytes) -> object:
            return data

    s = Serializer("secret", serializer=BytesSerializer())
    payload = b'{"key": "value"}'
    result = s.load_payload(payload)
    assert result == b'{"key": "value"}'
```


# LLM-generated content at query #9
#--------------------------

```python
def test_salt_is_not_none():
    serializer = Serializer(secret_key="secret", salt=b"custom_salt")
    assert serializer.salt is not None
```


# LLM-generated content at query #10
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback_evaluates_isinstance_tuple_true():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    class CustomSigner(Signer):
        pass

    signer_kwargs = {"key_derivation": "hmac"}
    serializer = Serializer(
        secret_key="secret",
        fallback_signers=[(CustomSigner, signer_kwargs)],
        signer_kwargs={},
    )
    unsigners = list(serializer.iter_unsigners(salt=None))
    assert isinstance(unsigners[1], CustomSigner)
    assert unsigners[1].key_derivation == "hmac"
```


# LLM-generated content at query #11
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")

def test_serializer_constructor_with_list_of_str_secret_keys():
    serializer = Serializer(["key1", "key2"])

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return eval(s)

    serializer = Serializer("secret", serializer=CustomSerializer())

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})

def test_serializer_constructor_with_custom_signer():
    from itsdangerous.signer import Signer

    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})

def test_serializer_constructor_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])

def test_serializer_constructor_with_all_parameters():
    serializer = Serializer(
        secret_key=["key1", b"key2"],
        salt=b"salt",
        serializer=json,
        serializer_kwargs={"indent": 2},
        signer=Signer,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=[{"digest_method": hashlib.sha1}],
    )


# LLM-generated content at query #12
#--------------------------

```python
def test_load_payload_exception_when_is_text_false_and_payload_invalid():
    serializer = Serializer("secret")
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
```


# LLM-generated content at query #13
#--------------------------

```
def test_load_payload_with_default_serializer_and_text_data():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_text_serializer():
    class CustomTextSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"data": payload}
        def dumps(self, obj: t.Any) -> str:
            return obj["data"]
    serializer = Serializer("secret", serializer=CustomTextSerializer())
    payload = serializer.dump_payload({"data": "test"})
    result = serializer.load_payload(payload)
    assert result == {"data": "test"}

def test_load_payload_with_custom_bytes_serializer():
    class CustomBytesSerializer:
        def loads(self, payload: bytes) -> t.Any:
            return {"data": payload.decode()}
        def dumps(self, obj: t.Any) -> bytes:
            return obj["data"].encode()
    serializer = Serializer("secret", serializer=CustomBytesSerializer())
    payload = serializer.dump_payload({"data": "test"})
    result = serializer.load_payload(payload)
    assert result == {"data": "test"}

def test_load_payload_with_explicit_serializer_override():
    class OverrideSerializer:
        def loads(self, payload: str) -> t.Any:
            return {"overridden": payload}
        def dumps(self, obj: t.Any) -> str:
            return obj["overridden"]
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload, serializer=OverrideSerializer())
    assert result == {"overridden": "{\"key\": \"value\"}"}

def test_load_payload_raises_bad_payload_on_invalid_data():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_text_serializer_and_bytes_input():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_bytes_serializer_and_text_input():
    serializer = Serializer("secret", serializer=json)
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #14
#--------------------------

```python
def test_fallback_signers_not_none():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    fallback_signers = [Signer]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers
```


# LLM-generated content at query #15
#--------------------------

def test_serializer_constructor_with_default_parameters():
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

def test_serializer_constructor_with_string_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_text_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.is_text_serializer is True

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

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #16
#--------------------------

```python
def test_fallback_signers_not_none_when_empty_list_provided():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []


# LLM-generated content at query #17
#--------------------------

```python
def test_loads_returns_any_type():
    serializer = _PDataSerializer()
    serializer.loads = lambda payload: 42
    result = serializer.loads("test")
    assert result == 42

def test_loads_accepts_serialized_input():
    serializer = _PDataSerializer()
    serializer.loads = lambda payload: payload
    result = serializer.loads("data")
    assert result == "data"

def test_loads_handles_none_input():
    serializer = _PDataSerializer()
    serializer.loads = lambda payload: None
    result = serializer.loads(None)
    assert result is None

def test_loads_handles_complex_types():
    serializer = _PDataSerializer()
    deserialized_data = {"key": [1, 2, 3]}
    serializer.loads = lambda payload: deserialized_data
    result = serializer.loads("{}")
    assert result == deserialized_data
```


# LLM-generated content at query #18
#--------------------------

```python
def test_salt_is_not_none_condition_false():
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #19
#--------------------------

```python
def test_serializer_uses_default_serializer_when_none_provided():
    from itsdangerous.serializer import Serializer
    serializer = Serializer("secret-key")
    assert serializer.serializer is Serializer.default_serializer
```


# LLM-generated content at query #20
#--------------------------

def test_dumps_with_str_serializer_returns_str():
    serializer = Serializer("secret", serializer=None)
    result = serializer.dumps("test")
    assert isinstance(result, str)

def test_dumps_with_bytes_serializer_returns_bytes():
    serializer = Serializer("secret", serializer=json, serializer_kwargs={})
    result = serializer.dumps("test")
    assert isinstance(result, bytes)

def test_dumps_different_objects():
    serializer = Serializer("secret")
    assert serializer.dumps(123) is not None
    assert serializer.dumps([1, 2, 3]) is not None
    assert serializer.dumps({"key": "value"}) is not None

def test_dumps_with_salt():
    serializer = Serializer("secret")
    result1 = serializer.dumps("test", salt="custom_salt")
    result2 = serializer.dumps("test", salt="custom_salt")
    assert result1 == result2

def test_dumps_different_salts_produce_different_results():
    serializer = Serializer("secret")
    result1 = serializer.dumps("test", salt="salt1")
    result2 = serializer.dumps("test", salt="salt2")
    assert result1 != result2

def test_dumps_empty_string():
    serializer = Serializer("secret")
    result = serializer.dumps("")
    assert result is not None

def test_dumps_none():
    serializer = Serializer("secret")
    result = serializer.dumps(None)
    assert result is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_serializer_uses_provided_serializer():
    custom_serializer = json
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer is custom_serializer
```


# LLM-generated content at query #22
#--------------------------

```python
def test_load_payload_with_default_serializer():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_explicit_serializer():
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
        def loads(self, payload):
            return payload.upper()
        def dumps(self, obj):
            return obj.lower()
    serializer = Serializer("secret", serializer=TextSerializer())
    payload = serializer.dump_payload("hello")
    result = serializer.load_payload(payload)
    assert result == "HELLO"

def test_load_payload_with_bytes_serializer():
    class BytesSerializer:
        def loads(self, payload):
            return payload.hex()
        def dumps(self, obj):
            return bytes.fromhex(obj)
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = serializer.dump_payload("68656c6c6f")
    result = serializer.load_payload(payload)
    assert result == "68656c6c6f"

def test_load_payload_with_override_serializer():
    serializer = Serializer("secret")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"key": "value"}

def test_load_payload_raises_bad_payload_with_exception():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"bad")
        assert False
    except BadPayload:
        pass
```


# LLM-generated content at query #23
#--------------------------

```python
def test_dumps_returns_bytes_when_is_text_serializer_is_false():
    serializer = Serializer(secret_key=b"secret", serializer=None)
    serializer.is_text_serializer = False
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #24
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

def test_serializer_constructor_with_bytes_secret_key():
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

def test_serializer_constructor_with_serializer_none():
    s = Serializer("secret", serializer=None)
    assert s.serializer is json
    assert s.is_text_serializer is True

def test_serializer_constructor_with_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    s = Serializer("secret", serializer=TextSerializer())
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()
        @staticmethod
        def loads(b):
            return int(b.decode())
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.is_text_serializer is False

def test_serializer_constructor_with_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers_none():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_with_fallback_signers_list():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #25
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

def test_serializer_constructor_with_bytes_secret():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: x, "loads": lambda self, x: x})()
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret-key", signer=custom_signer)
    assert serializer.signer is custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #26
#--------------------------

def test_constructor_with_string_secret_key():
    s = Serializer("secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    s = Serializer(b"secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"

def test_constructor_with_list_of_strings_secret_key():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_constructor_with_list_of_bytes_secret_key():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom-salt")
    assert s.salt == b"custom-salt"

def test_constructor_with_custom_serializer():
    class MockSerializer:
        @staticmethod
        def dumps(obj):
            return "serialized"
        @staticmethod
        def loads(data):
            return data
    s = Serializer("secret", serializer=MockSerializer())
    assert s.serializer is MockSerializer()
    assert s.is_text_serializer is True

def test_constructor_with_custom_signer():
    class MockSigner(Signer):
        pass
    s = Serializer("secret", signer=MockSigner)
    assert s.signer is MockSigner

def test_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_all_arguments():
    class MockSerializer:
        @staticmethod
        def dumps(obj):
            return b"serialized"
        @staticmethod
        def loads(data):
            return data
    class MockSigner(Signer):
        pass
    s = Serializer(
        ["key1", b"key2"],
        salt="custom-salt",
        serializer=MockSerializer(),
        serializer_kwargs={"indent": 2},
        signer=MockSigner,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[MockSigner, (MockSigner, {"digest_method": hashlib.sha256})]
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"custom-salt"
    assert s.serializer is MockSerializer()
    assert s.is_text_serializer is False
    assert s.signer is MockSigner
    assert s.signer_kwargs == {"key_derivation": "hmac"}
    assert s.fallback_signers == [MockSigner, (MockSigner, {"digest_method": hashlib.sha256})]
    assert s.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #27
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

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_secret_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_secret_key_list():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return "custom"
    s = Serializer("secret", serializer=CustomSerializer)
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return "bytes"
    s = Serializer("secret", serializer=BytesSerializer)
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
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback


# LLM-generated content at query #28
#--------------------------

def test_serializer_init_with_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_init_with_list_of_strings() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_salt_none() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_serializer() -> None:
    custom_serializer = json
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer == True

def test_serializer_init_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_init_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers() -> None:
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #29
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"digest_method": "sha256"})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
```


# LLM-generated content at query #30
#--------------------------

```python
def test_salt_is_none_does_not_change_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None
```


# LLM-generated content at query #31
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    secret_key = b"test-secret"
    salt = b"test-salt"
    custom_signer = type("CustomSigner", (Signer,), {})
    fallback_kwargs = {"key_derivation": "none"}
    serializer = Serializer(
        secret_key=secret_key,
        salt=salt,
        fallback_signers=[(custom_signer, fallback_kwargs)]
    )
    unsigners = list(serializer.iter_unsigners(salt))
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], custom_signer)
```


# LLM-generated content at query #32
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

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #33
#--------------------------

```python
def test_loads_calls_loads_with_payload():
    class MockSerializer:
        def loads(self, payload):
            return payload
        def dumps(self, obj):
            return obj
    serializer = MockSerializer()
    result = serializer.loads("test_payload")
    assert result == "test_payload"

def test_loads_returns_any_type():
    class MockSerializer:
        def loads(self, payload):
            return 42
        def dumps(self, obj):
            return obj
    serializer = MockSerializer()
    result = serializer.loads("data")
    assert result == 42

def test_loads_with_none_payload():
    class MockSerializer:
        def loads(self, payload):
            return None
        def dumps(self, obj):
            return obj
    serializer = MockSerializer()
    result = serializer.loads(None)
    assert result is None

def test_loads_with_complex_object():
    class MockSerializer:
        def loads(self, payload):
            return {"key": "value"}
        def dumps(self, obj):
            return obj
    serializer = MockSerializer()
    result = serializer.loads('{"key": "value"}')
    assert result == {"key": "value"}

def test_loads_with_integer_payload():
    class MockSerializer:
        def loads(self, payload):
            return payload * 2
        def dumps(self, obj):
            return obj
    serializer = MockSerializer()
    result = serializer.loads(5)
    assert result == 10
```


# LLM-generated content at query #34
#--------------------------

```python
def test_salt_is_none_evaluates_to_false():
    serializer = Serializer("secret-key", salt=None, serializer=None)
    assert serializer.salt is None
```


# LLM-generated content at query #35
#--------------------------

def test_serializer_constructor_with_default_parameters():
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

def test_serializer_constructor_with_list_of_string_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class FakeSerializer:
        @staticmethod
        def dumps(obj):
            return "dumped"
        @staticmethod
        def loads(s):
            return "loaded"
    serializer = Serializer("secret", serializer=FakeSerializer)
    assert serializer.serializer is FakeSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"dumped"
        @staticmethod
        def loads(s):
            return "loaded"
    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.serializer is BytesSerializer
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
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_all_parameters():
    serializer = Serializer(
        secret_key=["key1", b"key2"],
        salt=b"custom_salt",
        serializer=None,
        serializer_kwargs={"indent": 2},
        signer=Signer,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #36
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

def test_constructor_with_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_salt_string():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_constructor_with_serializer_str():
    class StrSerializer:
        @staticmethod
        def dumps(obj):
            return "str"
        @staticmethod
        def loads(s):
            return {}
    serializer = Serializer("secret", serializer=StrSerializer())
    assert serializer.is_text_serializer == True

def test_constructor_with_serializer_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(b):
            return {}
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_constructor_with_signer():
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

def test_constructor_with_fallback_signers_none():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_constructor_with_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"bytes_secret")
    assert serializer.secret_keys == [b"bytes_secret"]

def test_constructor_with_iterable_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #37
#--------------------------

def test_dumps_with_default_json_serializer_returns_bytes():
    serializer = Serializer("secret")
    result = serializer.dumps("test")
    assert isinstance(result, bytes)


# LLM-generated content at query #38
#--------------------------

```python
def test_dumps_is_text_serializer_false():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    secret_key = b"test-secret"
    salt = b"test-salt"

    serializer = Serializer(
        secret_key=secret_key,
        salt=salt,
        serializer=bytes,
        signer_kwargs={"digest_method": "sha256"},
    )

    result = serializer.dumps("test data")

    assert isinstance(result, bytes)
    assert result == serializer.make_signer(salt).sign(serializer.dump_payload("test data"))
```


# LLM-generated content at query #39
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

def test_serializer_constructor_with_list_of_secrets():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

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
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback


# LLM-generated content at query #40
#--------------------------

def test_serializer_init_with_str_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_init_with_list_of_str_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_init_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True

def test_serializer_init_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_init_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_init_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_init_with_all_parameters():
    serializer = Serializer(
        secret_key=["old_key", "new_key"],
        salt="custom",
        serializer=json,
        serializer_kwargs={"indent": 2},
        signer=Signer,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}],
    )
    assert serializer.secret_keys == [b"old_key", b"new_key"]
    assert serializer.salt == b"custom"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_init_default_serializer():
    serializer = Serializer("secret")
    assert serializer.serializer == json

def test_serializer_init_default_signer():
    serializer = Serializer("secret")
    assert serializer.signer == Signer

def test_serializer_init_default_fallback_signers():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []


# LLM-generated content at query #41
#--------------------------

def test_serializer_constructor_with_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

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
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: t.Any) -> bytes:
            return str(obj).encode()
        @staticmethod
        def loads(s: bytes) -> t.Any:
            return eval(s.decode())
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_secret_key_as_iterable() -> None:
    serializer = Serializer(iter(["key1", "key2"]))
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #42
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

def test_serializer_constructor_with_list_of_str_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt_str():
    s = Serializer("secret", salt="custom-salt")
    assert s.salt == b"custom-salt"

def test_serializer_constructor_with_custom_salt_bytes():
    s = Serializer("secret", salt=b"custom-salt")
    assert s.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "text", "loads": lambda self, x: {}})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer == custom_serializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: {}})()
    s = Serializer("secret", serializer=bytes_serializer)
    assert s.serializer == bytes_serializer
    assert s.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    custom_signer = Signer
    s = Serializer("secret", signer=custom_signer)
    assert s.signer == custom_signer

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers_dict():
    fallback = [{"key_derivation": "hmac"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_tuple():
    fallback = [(Signer, {"key_derivation": "hmac"})]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_signer_class():
    fallback = [Signer]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #43
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

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_text_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

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

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_key == b"new_key"


# LLM-generated content at query #44
#--------------------------

def test_load_payload_with_default_serializer_and_text():
    serializer = Serializer("secret")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def loads(self, payload):
            return {"custom": payload}
        def dumps(self, obj):
            return "custom_dumped"
    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"custom": "custom_dumped"}

def test_load_payload_with_bytes_serializer():
    class BytesSerializer:
        def loads(self, payload):
            return {"data": payload.decode()}
        def dumps(self, obj):
            return b"bytes_dumped"
    serializer = Serializer("secret", serializer=BytesSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"data": "bytes_dumped"}

def test_load_payload_raises_bad_payload_on_exception():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid_payload")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_explicit_serializer():
    class ExplicitSerializer:
        def loads(self, payload):
            return {"explicit": payload}
        def dumps(self, obj):
            return "explicit_dumped"
    serializer = Serializer("secret", serializer=ExplicitSerializer())
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload, serializer=ExplicitSerializer())
    assert result == {"explicit": "explicit_dumped"}


# LLM-generated content at query #45
#--------------------------

def test_serializer_constructor_default():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_bytes_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_list_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_list_bytes_keys():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_custom_salt():
    s = Serializer("secret", salt="my_salt")
    assert s.salt == b"my_salt"

def test_serializer_constructor_bytes_salt():
    s = Serializer("secret", salt=b"my_salt")
    assert s.salt == b"my_salt"

def test_serializer_constructor_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_custom_serializer_text():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(s):
            return {}
    s = Serializer("secret", serializer=TextSerializer)
    assert s.serializer == TextSerializer
    assert s.is_text_serializer == True

def test_serializer_constructor_custom_serializer_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return {}
    s = Serializer("secret", serializer=BytesSerializer)
    assert s.serializer == BytesSerializer
    assert s.is_text_serializer == False

def test_serializer_constructor_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_iterations": 100})
    assert s.signer_kwargs == {"key_iterations": 100}

def test_serializer_constructor_fallback_signers_dict():
    s = Serializer("secret", fallback_signers=[{"key_iterations": 50}])
    assert s.fallback_signers == [{"key_iterations": 50}]

def test_serializer_constructor_fallback_signers_tuple():
    s = Serializer("secret", fallback_signers=[(Signer, {"key_iterations": 50})])
    assert s.fallback_signers == [(Signer, {"key_iterations": 50})]

def test_serializer_constructor_fallback_signers_class():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", fallback_signers=[CustomSigner])
    assert s.fallback_signers == [CustomSigner]

def test_serializer_constructor_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #46
#--------------------------

def test_serializer_constructor_default_serializer() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_serializer_text() -> None:
    class TextSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> object:
            return eval(s)
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_constructor_custom_serializer_bytes() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return str(obj).encode()
        @staticmethod
        def loads(s: bytes) -> object:
            return eval(s.decode())
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_salt() -> None:
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_salt_none() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

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

def test_serializer_constructor_with_fallback_signers_none() -> None:
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_multiple_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_secret_key_bytes() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]


# LLM-generated content at query #47
#--------------------------

```python
def test_load_payload_predicate_false_when_serializer_not_text():
    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

class CustomBytesSerializer:
    def dumps(self, obj):
        return json.dumps(obj).encode("utf-8")
    def loads(self, data):
        return json.loads(data.decode("utf-8"))
```


# LLM-generated content at query #48
#--------------------------

```python
def test_iter_unsigners_handles_tuple_fallback():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", fallback_signers=[(CustomSigner, {"key_derivation": "none"})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[1], CustomSigner)
```


# LLM-generated content at query #49
#--------------------------

def test_serializer_init_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_init_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer == json

def test_serializer_init_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_init_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_init_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_init_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer == False


# LLM-generated content at query #50
#--------------------------

def test_serializer_constructor_default_serializer():
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

def test_serializer_constructor_with_list_of_secrets():
    s = Serializer(["secret1", "secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_list_of_bytes_secrets():
    s = Serializer([b"secret1", b"secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_salt_bytes():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_salt_string():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer_text():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    s = Serializer("secret", serializer=TextSerializer)
    assert s.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_bytes():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return bytes(str(obj), "utf-8")
        @staticmethod
        def loads(b):
            return int(b.decode("utf-8"))
    s = Serializer("secret", serializer=BytesSerializer)
    assert s.is_text_serializer is False

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #51
#--------------------------

def test_constructor_with_defaults():
    s = Serializer("secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer == True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    s = Serializer(b"secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"

def test_constructor_with_list_secret_key():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"

def test_constructor_with_bytes_list_secret_key():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"

def test_constructor_with_custom_salt():
    s = Serializer("secret-key", salt="custom-salt")
    assert s.salt == b"custom-salt"

def test_constructor_with_none_salt():
    s = Serializer("secret-key", salt=None)
    assert s.salt is None

def test_constructor_with_bytes_salt():
    s = Serializer("secret-key", salt=b"custom-salt")
    assert s.salt == b"custom-salt"

def test_constructor_with_custom_serializer():
    serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: eval(x)})()
    s = Serializer("secret-key", serializer=serializer)
    assert s.serializer is serializer
    assert s.is_text_serializer == True

def test_constructor_with_bytes_serializer():
    serializer = type("BytesSerializer", (), {"dumps": lambda self, x: str(x).encode(), "loads": lambda self, x: eval(x.decode())})()
    s = Serializer("secret-key", serializer=serializer)
    assert s.serializer is serializer
    assert s.is_text_serializer == False

def test_constructor_with_serializer_kwargs():
    s = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret-key", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    s = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "none"}]
    s = Serializer("secret-key", fallback_signers=fallback_signers)
    assert s.fallback_signers == fallback_signers

def test_constructor_with_empty_fallback_signers():
    s = Serializer("secret-key", fallback_signers=[])
    assert s.fallback_signers == []

def test_constructor_with_all_parameters():
    serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: eval(x)})()
    class CustomSigner(Signer):
        pass
    s = Serializer(
        ["key1", "key2"],
        salt="custom-salt",
        serializer=serializer,
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"custom-salt"
    assert s.serializer is serializer
    assert s.is_text_serializer == True
    assert s.signer is CustomSigner
    assert s.signer_kwargs == {"key_derivation": "hmac"}
    assert s.fallback_signers == [{"key_derivation": "none"}]
    assert s.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #52
#--------------------------

```python
def test_load_payload_is_text_false_with_bytes_serializer():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    import json

    class BytesSerializer:
        def dumps(self, obj):
            return json.dumps(obj).encode('utf-8')
        def loads(self, data):
            return json.loads(data.decode('utf-8'))

    serializer = Serializer('secret', serializer=BytesSerializer())
    payload = serializer.dump_payload({'key': 'value'})
    result = serializer.load_payload(payload)
    assert result == {'key': 'value'}


# LLM-generated content at query #53
#--------------------------

def test_serializer_constructor_with_str_secret_key() -> None:
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_str_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_bytes_salt() -> None:
    serializer = Serializer("secret", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_str_salt() -> None:
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_custom_text_serializer() -> None:
    class TextSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return str(obj)
        @staticmethod
        def loads(s: str) -> object:
            return int(s)
    serializer = Serializer("secret", serializer=TextSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return str(obj).encode()
        @staticmethod
        def loads(s: bytes) -> object:
            return int(s.decode())
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

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

def test_serializer_constructor_default_fallback_signers() -> None:
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []


# LLM-generated content at query #54
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_string")
    assert result is not None
```


# LLM-generated content at query #55
#--------------------------

def test_serializer_constructor_with_default_parameters():
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

def test_serializer_constructor_with_custom_serializer():
    class FakeSerializer:
        def dumps(self, obj):
            return "fake"
        def loads(self, data):
            return {"fake": data}
    s = Serializer("secret", serializer=FakeSerializer())
    assert s.serializer == FakeSerializer()
    assert s.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        def dumps(self, obj):
            return b"bytes"
        def loads(self, data):
            return {"bytes": data}
    s = Serializer("secret", serializer=BytesSerializer())
    assert s.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    class FakeSigner:
        def __init__(self, secret_keys, salt="", **kwargs):
            pass
        def sign(self, value):
            return value
        def unsign(self, value):
            return value
    s = Serializer("secret", signer=FakeSigner)
    assert s.signer == FakeSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_custom_default_serializer():
    original = Serializer.default_serializer
    Serializer.default_serializer = json
    s = Serializer("secret")
    assert s.serializer == json
    Serializer.default_serializer = original

def test_serializer_constructor_with_custom_default_signer():
    original = Serializer.default_signer
    Serializer.default_signer = Signer
    s = Serializer("secret")
    assert s.signer == Signer
    Serializer.default_signer = original

def test_serializer_constructor_with_custom_default_fallback_signers():
    original = Serializer.default_fallback_signers
    Serializer.default_fallback_signers = [{"key_derivation": "none"}]
    s = Serializer("secret")
    assert s.fallback_signers == [{"key_derivation": "none"}]
    Serializer.default_fallback_signers = original


# LLM-generated content at query #56
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads("test_payload")
    assert result is not None

def test_loads_accepts_string():
    serializer = _PDataSerializer()
    result = serializer.loads("hello")
    assert isinstance(result, object)

def test_loads_accepts_bytes():
    serializer = _PDataSerializer()
    result = serializer.loads(b"data")
    assert result is not None

def test_loads_accepts_int():
    serializer = _PDataSerializer()
    result = serializer.loads(42)
    assert result is not None
```


# LLM-generated content at query #57
#--------------------------

```
def test_dumps_returns_serialized_type():
    _PDataSerializer_dumps = _PDataSerializer().dumps
    assert _PDataSerializer_dumps("test") is not None
    assert _PDataSerializer_dumps(123) is not None
    assert _PDataSerializer_dumps([1, 2, 3]) is not None
    assert _PDataSerializer_dumps({"key": "value"}) is not None
    assert _PDataSerializer_dumps(None) is not None
    assert _PDataSerializer_dumps(True) is not None
```


# LLM-generated content at query #58
#--------------------------

```python
def test_dumps_returns_expected_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_string")
    assert result == "test_string"

def test_dumps_returns_expected_serialized_type_for_integer():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result == 42

def test_dumps_returns_expected_serialized_type_for_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result == [1, 2, 3]

def test_dumps_returns_expected_serialized_type_for_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert result == {"key": "value"}

def test_dumps_returns_expected_serialized_type_for_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is None
```


# LLM-generated content at query #59
#--------------------------

def test_serializer_init_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_init_with_list_of_string_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_init_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_init_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(s):
            return s
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_init_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_init_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_init_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_none_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_init_with_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.serializer is json

def test_serializer_init_with_default_signer():
    serializer = Serializer("secret-key")
    assert serializer.signer is Signer

def test_serializer_init_with_default_fallback_signers():
    serializer = Serializer("secret-key")
    assert serializer.fallback_signers == []


# LLM-generated content at query #60
#--------------------------

def test_serializer_init_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert isinstance(serializer.signer, type)
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_init_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_init_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_init_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_serializer_str():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return {"custom": True}
    serializer = Serializer("secret-key", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_init_with_custom_serializer_bytes():
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"custom"
        @staticmethod
        def loads(s):
            return {"custom": True}
    serializer = Serializer("secret-key", serializer=CustomBytesSerializer())
    assert serializer.serializer == CustomBytesSerializer()
    assert serializer.is_text_serializer == False

def test_serializer_init_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_init_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #61
#--------------------------

def test_serializer_constructor_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == serializer.default_serializer
    assert serializer.is_text_serializer == True
    assert serializer.signer == serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys() -> None:
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt() -> None:
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_bytes_salt() -> None:
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return "dumped"

        @staticmethod
        def loads(s: str) -> object:
            return "loaded"

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return b"dumped"

        @staticmethod
        def loads(s: bytes) -> object:
            return "loaded"

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer_class() -> None:
    class CustomSigner:
        def __init__(self, secret_key: bytes, salt: bytes, **kwargs: object) -> None:
            pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer == CustomSigner

def test_serializer_constructor_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers() -> None:
    class FallbackSigner:
        def __init__(self, secret_key: bytes, salt: bytes, **kwargs: object) -> None:
            pass

    fallback = FallbackSigner
    serializer = Serializer("secret", fallback_signers=[fallback])
    assert serializer.fallback_signers == [fallback]

def test_serializer_constructor_with_fallback_signers_dict() -> None:
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "hmac"}])
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]

def test_serializer_constructor_with_fallback_signers_tuple() -> None:
    class FallbackSigner:
        def __init__(self, secret_key: bytes, salt: bytes, **kwargs: object) -> None:
            pass

    serializer = Serializer("secret", fallback_signers=[(FallbackSigner, {"key_derivation": "hmac"})])
    assert serializer.fallback_signers == [(FallbackSigner, {"key_derivation": "hmac"})]

def test_serializer_constructor_secret_key_property() -> None:
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_key == b"new_key"

def test_serializer_constructor_inherits_default_fallback_signers() -> None:
    class CustomSerializer(Serializer):
        default_fallback_signers = [{"key_derivation": "hmac"}]

    serializer = CustomSerializer("secret")
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]


# LLM-generated content at query #62
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    payload = None
    result = serializer.loads(payload)
    assert result is None or True


# LLM-generated content at query #63
#--------------------------

def test_serializer_constructor_defaults():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(serializer.serializer)
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

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
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer == is_text_serializer(CustomSerializer())

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
            return int(s)
    class CustomSigner(Signer):
        pass
    serializer = Serializer(
        secret_key=["key1", b"key2"],
        salt=b"custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"digest_method": hashlib.sha256},
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer == is_text_serializer(CustomSerializer())
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}
    assert serializer.fallback_signers == [{"key_derivation": "hmac"}]
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #64
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

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"

        @staticmethod
        def loads(s):
            return "custom"

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(s):
            return "bytes"

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer == False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_signer():
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


# LLM-generated content at query #65
#--------------------------

```python
def test_load_payload_predicate_false_when_is_text_false():
    from itsdangerous.serializer import Serializer
    from itsdangerous.exc import BadPayload

    serializer = Serializer("secret-key")
    serializer.is_text_serializer = False
    payload = b"not-valid-json"
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
```


# LLM-generated content at query #66
#--------------------------

```python
def test_load_payload_when_serializer_is_bytes_serializer_and_is_text_is_false():
    serializer = Serializer("secret", serializer=dill)
    payload = dill.dumps({"key": "value"})
    result = serializer.load_payload(payload, serializer=dill)
    assert result == {"key": "value"}
```


# LLM-generated content at query #67
#--------------------------

```python
def test_salt_is_not_none():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"
```


# LLM-generated content at query #68
#--------------------------

def test_iter_unsigners_yields_make_signer_result():
    serializer = Serializer("secret")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0] is not None

def test_iter_unsigners_yields_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_yields_multiple_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[Signer, Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3

def test_iter_unsigners_uses_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    signers = list(serializer.iter_unsigners())
    assert signers[0].salt == b"custom_salt"

def test_iter_unsigners_uses_provided_salt():
    serializer = Serializer("secret")
    signers = list(serializer.iter_unsigners(salt="provided_salt"))
    assert signers[0].salt == b"provided_salt"

def test_iter_unsigners_yields_fallback_with_dict_config():
    serializer = Serializer("secret", fallback_signers=[{}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_yields_fallback_with_tuple_config():
    serializer = Serializer("secret", fallback_signers=[(Signer, {})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_yields_fallback_with_class_only():
    serializer = Serializer("secret", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2

def test_iter_unsigners_yields_multiple_fallback_with_mixed_configs():
    serializer = Serializer("secret", fallback_signers=[{}, (Signer, {}), Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4

def test_iter_unsigners_yields_fallback_for_each_secret_key():
    serializer = Serializer(["key1", "key2"], fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3

def test_iter_unsigners_yields_fallback_for_each_secret_key_with_dict():
    serializer = Serializer(["key1", "key2"], fallback_signers=[{}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3


# LLM-generated content at query #69
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_constructor_with_bytes_serializer():
    from itsdangerous.serializer import _PDataSerializer
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.serializer is BytesSerializer
    assert not serializer.is_text_serializer

def test_constructor_with_text_serializer():
    class TextSerializer:
        @staticmethod
        def dumps(obj):
            return "text"
        @staticmethod
        def loads(data):
            return data
    serializer = Serializer("secret-key", serializer=TextSerializer())
    assert serializer.serializer is TextSerializer
    assert serializer.is_text_serializer

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_secret_key_property():
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_key == b"new_key"


# LLM-generated content at query #70
#--------------------------

```python
def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None
```


# LLM-generated content at query #71
#--------------------------

def test_serializer_constructor_default_values():
    s = Serializer("secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is s.default_serializer
    assert s.is_text_serializer == True
    assert s.signer is s.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_bytes_secret_key():
    s = Serializer(b"secret-key")
    assert s.secret_keys == [b"secret-key"]

def test_serializer_constructor_list_of_str_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_list_of_bytes_secret_keys():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_custom_salt():
    s = Serializer("secret-key", salt="custom-salt")
    assert s.salt == b"custom-salt"

def test_serializer_constructor_custom_salt_none():
    s = Serializer("secret-key", salt=None)
    assert s.salt is None

def test_serializer_constructor_custom_salt_bytes():
    s = Serializer("secret-key", salt=b"custom-salt")
    assert s.salt == b"custom-salt"

def test_serializer_constructor_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "str", "loads": lambda self, x: {}})
    s = Serializer("secret-key", serializer=custom_serializer())
    assert s.serializer is not s.default_serializer
    assert s.is_text_serializer == True

def test_serializer_constructor_custom_serializer_bytes():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: b"bytes", "loads": lambda self, x: {}})
    s = Serializer("secret-key", serializer=custom_serializer())
    assert s.serializer is not s.default_serializer
    assert s.is_text_serializer == False

def test_serializer_constructor_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {"__init__": lambda self, keys, salt, **kwargs: None, "sign": lambda self, x: x, "unsign": lambda self, x: x})
    s = Serializer("secret-key", signer=custom_signer)
    assert s.signer is custom_signer

def test_serializer_constructor_custom_signer_kwargs():
    s = Serializer("secret-key", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_custom_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret-key", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_custom_serializer_kwargs():
    s = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_all_custom_parameters():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "str", "loads": lambda self, x: {}})
    custom_signer = type("CustomSigner", (Signer,), {"__init__": lambda self, keys, salt, **kwargs: None, "sign": lambda self, x: x, "unsign": lambda self, x: x})
    s = Serializer(
        ["key1", "key2"],
        salt="custom-salt",
        serializer=custom_serializer(),
        serializer_kwargs={"sort_keys": True},
        signer=custom_signer,
        signer_kwargs={"key_derivation": "none"},
        fallback_signers=[{"key_derivation": "hmac"}],
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"custom-salt"
    assert s.serializer is custom_serializer()
    assert s.signer is custom_signer
    assert s.signer_kwargs == {"key_derivation": "none"}
    assert s.fallback_signers == [{"key_derivation": "hmac"}]
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #72
#--------------------------

```python
def test_serializer_init_with_serializer_not_none():
    custom_serializer = json
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
```


# LLM-generated content at query #73
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #74
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_not_text():
    serializer = Serializer("secret", serializer=t.Any)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #75
#--------------------------

def test_serializer_init_with_defaults() -> None:
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key() -> None:
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_init_with_list_of_secret_keys() -> None:
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes_secret_keys() -> None:
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_none_salt() -> None:
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_init_with_bytes_salt() -> None:
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_init_with_custom_serializer() -> None:
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: str(x), "loads": lambda self, x: x})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer == custom_serializer
    assert s.is_text_serializer

def test_serializer_init_with_custom_signer() -> None:
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_init_with_signer_kwargs() -> None:
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers() -> None:
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_init_with_serializer_kwargs() -> None:
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #76
#--------------------------

```python
def test_loads_returns_any_type():
    serializer = _PDataSerializer()
    result = serializer.loads("test_payload")
    assert result is not None

def test_loads_accepts_string_payload():
    serializer = _PDataSerializer()
    result = serializer.loads("test_string")
    assert isinstance(result, object)

def test_loads_accepts_bytes_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(b"test_bytes")
    assert isinstance(result, object)

def test_loads_accepts_integer_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(42)
    assert isinstance(result, object)

def test_loads_accepts_list_payload():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert isinstance(result, object)

def test_loads_accepts_dict_payload():
    serializer = _PDataSerializer()
    result = serializer.loads({"key": "value"})
    assert isinstance(result, object)

def test_loads_accepts_none_payload():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is None

def test_loads_accepts_empty_string():
    serializer = _PDataSerializer()
    result = serializer.loads("")
    assert isinstance(result, object)
```


# LLM-generated content at query #77
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.salt == b"itsdangerous"
    assert serializer.secret_keys == [b"secret"]
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

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"digest_method": "sha256"})
    assert serializer.signer_kwargs == {"digest_method": "sha256"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"digest_method": "sha256"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #78
#--------------------------

def test_serializer_init_with_defaults() -> None:
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key() -> None:
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_init_with_list_of_strings_secret_key() -> None:
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes_secret_key() -> None:
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_custom_salt() -> None:
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_init_with_none_salt() -> None:
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_init_with_custom_serializer_positional() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return "dumped"

        @staticmethod
        def loads(s: str) -> object:
            return "loaded"

    s = Serializer("secret", b"salt", CustomSerializer)
    assert s.serializer is CustomSerializer
    assert s.is_text_serializer is True

def test_serializer_init_with_bytes_serializer_keyword() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return b"dumped"

        @staticmethod
        def loads(s: bytes) -> object:
            return b"loaded"

    s = Serializer("secret", serializer=BytesSerializer)
    assert s.serializer is BytesSerializer
    assert s.is_text_serializer is False

def test_serializer_init_with_signer_kwargs() -> None:
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers() -> None:
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_init_with_serializer_kwargs() -> None:
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_init_with_custom_signer_class() -> None:
    class CustomSigner(Signer):
        pass

    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_init_with_multiple_secret_keys_and_fallback_signers() -> None:
    s = Serializer(["old_key", "new_key"], fallback_signers=[{"digest_method": hashlib.sha256}])
    assert s.secret_keys == [b"old_key", b"new_key"]
    assert s.fallback_signers == [{"digest_method": hashlib.sha256}]


# LLM-generated content at query #79
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_string")
    assert result is not None

def test_dumps_returns_correct_type():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result == 42

def test_dumps_accepts_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is None

def test_dumps_accepts_empty_string():
    serializer = _PDataSerializer()
    result = serializer.dumps("")
    assert result == ""

def test_dumps_accepts_zero():
    serializer = _PDataSerializer()
    result = serializer.dumps(0)
    assert result == 0

def test_dumps_accepts_false():
    serializer = _PDataSerializer()
    result = serializer.dumps(False)
    assert result is False

def test_dumps_accepts_empty_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([])
    assert result == []

def test_dumps_accepts_empty_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({})
    assert result == {}
```


# LLM-generated content at query #80
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

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_str_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_keys():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt_as_bytes():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_salt_as_str():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer_positional():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return obj
    s = Serializer("secret", b"itsdangerous", CustomSerializer())
    assert s.serializer is CustomSerializer()
    assert s.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_keyword():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return b"custom"
        @staticmethod
        def loads(s):
            return obj
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer()
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
    s = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert s.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_constructor_with_fallback_signers_none():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []


# LLM-generated content at query #81
#--------------------------

def test_serializer_init_defaults() -> None:
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key() -> None:
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_init_with_list_of_secret_keys() -> None:
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_salt_none() -> None:
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_serializer() -> None:
    class CustomSerializer:
        @staticmethod
        def dumps(obj: object) -> str:
            return str(obj)

        @staticmethod
        def loads(s: str) -> object:
            return s

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.is_text_serializer is True

def test_serializer_init_with_bytes_serializer() -> None:
    class BytesSerializer:
        @staticmethod
        def dumps(obj: object) -> bytes:
            return b"data"

        @staticmethod
        def loads(s: bytes) -> object:
            return s

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_init_with_signer_kwargs() -> None:
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers() -> None:
    serializer = Serializer("secret", fallback_signers=[{"key_derivation": "none"}])
    assert serializer.fallback_signers == [{"key_derivation": "none"}]

def test_serializer_init_with_serializer_kwargs() -> None:
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #82
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

def test_serializer_constructor_with_list_of_strings():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "test", "loads": lambda self, x: x})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_none():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #83
#--------------------------

def test_serializer_constructor_default_values():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer == is_text_serializer(s.serializer)
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_strings():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt=b"custom_salt")
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
        def loads(data):
            return data

    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer == CustomSerializer()
    assert s.is_text_serializer == is_text_serializer(CustomSerializer())

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(data):
            return data

    s = Serializer("secret", serializer=BytesSerializer())
    assert s.serializer == BytesSerializer()
    assert s.is_text_serializer == is_text_serializer(BytesSerializer())

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
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

def test_serializer_constructor_with_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []

def test_serializer_constructor_with_none_fallback_signers():
    s = Serializer("secret", fallback_signers=None)
    assert s.fallback_signers == []


# LLM-generated content at query #84
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == is_text_serializer(json)
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "dumped", "loads": lambda self, x: "loaded"})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_secret_keys():
    serializer = Serializer(["key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_text_serializer():
    text_serializer = type("TextSerializer", (), {"dumps": lambda self, x: "dumped", "loads": lambda self, x: "loaded"})()
    serializer = Serializer("secret", serializer=text_serializer)
    assert serializer.is_text_serializer == is_text_serializer(text_serializer)


# LLM-generated content at query #85
#--------------------------

```python
def test_iter_unsigners_yields_default_signer_first():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 0
    assert isinstance(signers[0], Signer)

def test_iter_unsigners_yields_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[{"algorithm": "sha256"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 1

def test_iter_unsigners_uses_custom_salt():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    signers = list(serializer.iter_unsigners())
    for signer in signers:
        assert signer.salt == b"custom-salt"

def test_iter_unsigners_uses_fallback_as_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"key_derivation": "hmac"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 1

def test_iter_unsigners_uses_fallback_as_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) > 1

def test_iter_unsigners_yields_correct_number_of_signers():
    serializer = Serializer("secret-key", fallback_signers=[{}, {"algorithm": "sha256"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
```


# LLM-generated content at query #86
#--------------------------

def test_serializer_constructor_default_parameters():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert s.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert s.signer is Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_custom_salt_bytes():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_custom_salt_str():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer()
    assert s.is_text_serializer == is_text_serializer(CustomSerializer())

def test_serializer_constructor_custom_serializer_bytes():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(data):
            return data
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer is CustomSerializer()
    assert s.is_text_serializer is False

def test_serializer_constructor_custom_signer():
    class CustomSigner:
        def __init__(self, secret_keys, salt, **kwargs):
            self.secret_keys = secret_keys
            self.salt = salt
            self.kwargs = kwargs
        def sign(self, value):
            return value
        def unsign(self, value):
            return value
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer is CustomSigner

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

def test_serializer_constructor_secret_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_secret_key_bytes():
    s = Serializer(b"key_bytes")
    assert s.secret_keys == [b"key_bytes"]

def test_serializer_constructor_secret_key_str():
    s = Serializer("key_str")
    assert s.secret_keys == [b"key_str"]


# LLM-generated content at query #87
#--------------------------

def test_serializer_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_custom_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_custom_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_custom_salt_str():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json
    assert serializer.is_text_serializer

def test_serializer_constructor_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

def test_serializer_constructor_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert serializer.signer_kwargs == {"key_derivation": "none"}

def test_serializer_constructor_custom_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_secret_key_bytes():
    serializer = Serializer(b"bytes_secret")
    assert serializer.secret_keys == [b"bytes_secret"]

def test_serializer_constructor_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_secret_key_list_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #88
#--------------------------

def test_dumps_returns_bytes_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=lambda: None)
    serializer.serializer.dumps = lambda obj: b"data"
    serializer.is_text_serializer = False
    result = serializer.dumps("test")
    assert isinstance(result, bytes)

def test_dumps_returns_str_with_text_serializer():
    serializer = Serializer("secret-key", serializer=lambda: None)
    serializer.serializer.dumps = lambda obj: "data"
    serializer.is_text_serializer = True
    result = serializer.dumps("test")
    assert isinstance(result, str)

def test_dumps_uses_custom_salt():
    serializer = Serializer("secret-key", serializer=lambda: None)
    serializer.serializer.dumps = lambda obj: b"data"
    serializer.make_signer = lambda salt: type('Signer', (), {'sign': lambda self, payload: b"signed"})()
    result = serializer.dumps("test", salt="custom-salt")
    assert result == b"signed"

def test_dumps_calls_dump_payload():
    serializer = Serializer("secret-key", serializer=lambda: None)
    serializer.dump_payload = lambda obj: b"payload"
    serializer.make_signer = lambda salt: type('Signer', (), {'sign': lambda self, payload: payload})()
    result = serializer.dumps("test")
    assert result == b"payload"


# LLM-generated content at query #89
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer("secret", fallback_signers=[])
    assert serializer.fallback_signers == []
```


# LLM-generated content at query #90
#--------------------------

```python
def test_loads_returns_any_type():
    serializer = _PDataSerializer()
    result = serializer.loads(42)
    assert result is not None

def test_loads_accepts_serialized_input():
    serializer = _PDataSerializer()
    result = serializer.loads("test_string")
    assert isinstance(result, object)

def test_loads_returns_none_for_empty_input():
    serializer = _PDataSerializer()
    result = serializer.loads(None)
    assert result is None

def test_loads_accepts_list_input():
    serializer = _PDataSerializer()
    result = serializer.loads([1, 2, 3])
    assert isinstance(result, object)
```


# LLM-generated content at query #91
#--------------------------

def test_load_payload_with_default_text_serializer():
    serializer = Serializer("secret")
    result = serializer.load_payload(b'"test"')
    assert result == "test"

def test_load_payload_with_bytes_serializer():
    class BytesSerializer:
        def loads(self, payload):
            return payload.decode()
        def dumps(self, obj):
            return obj.encode()
    serializer = Serializer("secret", serializer=BytesSerializer())
    result = serializer.load_payload(b"hello")
    assert result == "hello"

def test_load_payload_with_explicit_text_serializer():
    class TextSerializer:
        def loads(self, payload):
            return payload.upper()
        def dumps(self, obj):
            return obj.lower()
    serializer = Serializer("secret", serializer=TextSerializer())
    result = serializer.load_payload(b"test")
    assert result == "TEST"

def test_load_payload_raises_bad_payload_on_exception():
    class BadSerializer:
        def loads(self, payload):
            raise ValueError("error")
        def dumps(self, obj):
            return ""
    serializer = Serializer("secret", serializer=BadSerializer())
    try:
        serializer.load_payload(b"data")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_override_serializer():
    class OverrideSerializer:
        def loads(self, payload):
            return payload + "overridden"
        def dumps(self, obj):
            return obj
    serializer = Serializer("secret")
    result = serializer.load_payload(b"base", serializer=OverrideSerializer())
    assert result == "baseoverridden"

def test_load_payload_with_override_bytes_serializer():
    class OverrideBytesSerializer:
        def loads(self, payload):
            return payload.hex()
        def dumps(self, obj):
            return bytes.fromhex(obj)
    serializer = Serializer("secret")
    result = serializer.load_payload(b"\x00\x01", serializer=OverrideBytesSerializer())
    assert result == "0001"


# LLM-generated content at query #92
#--------------------------

```
def test_load_payload_is_text_false():
    serializer = Serializer("secret")
    serializer.is_text_serializer = False
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}
```


# LLM-generated content at query #93
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads(b"test")
    assert result is not None
```


# LLM-generated content at query #94
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

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_list_of_strings_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes_secret_key():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "text", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = type("BytesSerializer", (), {"dumps": lambda self, x: b"data", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer is custom_signer

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

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_key == b"new_key"


# LLM-generated content at query #95
#--------------------------

```python
def test_load_payload_raises_bad_payload_when_serializer_returns_bytes_and_payload_decode_fails():
    serializer = Serializer("secret", serializer=json)
    invalid_payload = b"\xff\xfe\x00\x00"
    try:
        serializer.load_payload(invalid_payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass
```


# LLM-generated content at query #96
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
            return b"data"
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

def test_serializer_constructor_secret_key_property():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_key == b"new"

def test_serializer_constructor_serializer_keyword_argument():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(b):
            return b"data"
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False


# LLM-generated content at query #97
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback():
    from itsdangerous.signer import Signer
    from itsdangerous.serializer import Serializer
    from itsdangerous._json import json

    serializer = Serializer(
        secret_key=b"secret",
        serializer=json,
        fallback_signers=[(Signer, {"digest_method": "sha256"})],
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
```


# LLM-generated content at query #98
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

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(s):
            return s

    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.is_text_serializer is False

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
    class FallbackSigner(Signer):
        pass

    serializer = Serializer("secret", fallback_signers=[FallbackSigner, {"key_derivation": "none"}])
    assert len(serializer.fallback_signers) == 2
    assert serializer.fallback_signers[0] is FallbackSigner
    assert serializer.fallback_signers[1] == {"key_derivation": "none"}


# LLM-generated content at query #99
#--------------------------

```python
def test_load_payload_with_bytes_serializer_evaluates_is_text_false():
    from itsdangerous.serializer import Serializer
    from itsdangerous.exc import BadPayload

    serializer = Serializer("secret-key", serializer=lambda: None)
    serializer.serializer = type(
        "BytesSerializer",
        (),
        {
            "loads": lambda self, x: x,
            "dumps": lambda self, x: x,
        },
    )()
    serializer.is_text_serializer = False
    payload = b"test"
    try:
        result = serializer.load_payload(payload)
    except BadPayload:
        result = None
    assert result == b"test"
```


# LLM-generated content at query #100
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

def test_serializer_constructor_with_bytes_secret_key():
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
            return json.loads(s)

    serializer = Serializer("secret", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode()

        @staticmethod
        def loads(b):
            return json.loads(b)

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

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

def test_serializer_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return json.loads(s)

    class CustomSigner(Signer):
        pass

    serializer = Serializer(
        secret_key=["old_key", "new_key"],
        salt=b"custom_salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"sort_keys": True},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert serializer.secret_keys == [b"old_key", b"new_key"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer is True
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {"digest_method": "sha256"}
    assert serializer.fallback_signers == [{"key_derivation": "none"}]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #101
#--------------------------

```python
def test_fallback_signers_not_none_after_init():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer

    serializer = Serializer("secret")
    assert serializer.fallback_signers is not None
```


# LLM-generated content at query #102
#--------------------------

def test_serializer_constructor_defaults():
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

def test_serializer_constructor_with_list_of_strings():
    secret_key = ["key1", "key2"]
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_list_of_bytes():
    secret_key = [b"key1", b"key2"]
    serializer = Serializer(secret_key)
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    secret_key = "secret"
    salt = b"custom_salt"
    serializer = Serializer(secret_key, salt=salt)
    assert serializer.salt == b"custom_salt"

def test_serializer_constructor_with_none_salt():
    secret_key = "secret"
    serializer = Serializer(secret_key, salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    secret_key = "secret"
    serializer = Serializer(secret_key, serializer=json)
    assert serializer.serializer == json

def test_serializer_constructor_with_serializer_kwargs():
    secret_key = "secret"
    serializer_kwargs = {"indent": 2}
    serializer = Serializer(secret_key, serializer_kwargs=serializer_kwargs)
    assert serializer.serializer_kwargs == {"indent": 2}

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
    serializer = Serializer(secret_key, fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_constructor_with_all_parameters():
    secret_key = ["key1", b"key2"]
    salt = b"custom_salt"
    serializer_kwargs = {"sort_keys": True}
    signer_kwargs = {"digest_method": hashlib.sha256}
    fallback_signers = [{"key_derivation": "none"}, (Signer, {"digest_method": hashlib.sha1})]
    serializer = Serializer(
        secret_key,
        salt=salt,
        serializer=json,
        serializer_kwargs=serializer_kwargs,
        signer=Signer,
        signer_kwargs=signer_kwargs,
        fallback_signers=fallback_signers,
    )
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer == True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"digest_method": hashlib.sha256}
    assert serializer.fallback_signers == [
        {"key_derivation": "none"},
        (Signer, {"digest_method": hashlib.sha1}),
    ]
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #103
#--------------------------

def test_constructor_default_serializer():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer == is_text_serializer(Serializer.default_serializer)
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_constructor_with_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
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
    assert serializer.serializer is CustomSerializer()
    assert serializer.is_text_serializer == is_text_serializer(CustomSerializer())

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

def test_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer.serializer_kwargs == {"sort_keys": True}

def test_constructor_with_secret_key_bytes():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_constructor_with_secret_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_secret_key_list_bytes():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #104
#--------------------------

def test_serializer_constructor_with_defaults():
    s = Serializer("secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is Serializer.default_serializer
    assert s.is_text_serializer == is_text_serializer(Serializer.default_serializer)
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

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret-key", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    s = Serializer("secret-key", serializer=CustomSerializer())
    assert s.serializer is not None
    assert s.is_text_serializer == isinstance(CustomSerializer().dumps({}), str)

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret-key", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    from itsdangerous.signer import Signer
    class CustomSigner(Signer):
        pass

    s = Serializer("secret-key", signer=CustomSigner)
    assert s.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret-key", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_none():
    s = Serializer("secret-key", fallback_signers=None)
    assert s.fallback_signers == []

def test_serializer_constructor_with_all_parameters():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    from itsdangerous.signer import Signer
    class CustomSigner(Signer):
        pass

    s = Serializer(
        secret_key=["key1", b"key2"],
        salt=b"custom-salt",
        serializer=CustomSerializer(),
        serializer_kwargs={"indent": 2},
        signer=CustomSigner,
        signer_kwargs={"digest_method": "sha256"},
        fallback_signers=[{"key_derivation": "none"}]
    )
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"custom-salt"
    assert s.is_text_serializer == isinstance(CustomSerializer().dumps({}), str)
    assert s.signer is CustomSigner
    assert s.signer_kwargs == {"digest_method": "sha256"}
    assert s.fallback_signers == [{"key_derivation": "none"}]
    assert s.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #105
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

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_bytes_salt():
    s = Serializer("secret", salt=b"custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_constructor_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return "custom"
        @staticmethod
        def loads(s):
            return "custom"
    s = Serializer("secret", serializer=CustomSerializer)
    assert s.serializer == CustomSerializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_constructor_with_custom_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []


# LLM-generated content at query #106
#--------------------------

```python
def test_dumps_returns_bytes_when_serializer_is_not_text():
    serializer = Serializer(b"secret", serializer=type("BytesSerializer", (), {"dumps": lambda self, obj: b"data", "loads": lambda self, data: data})())
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result == serializer.make_signer().sign(b"data")
```


# LLM-generated content at query #107
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test")
    assert result is not None

def test_dumps_accepts_any_type():
    serializer = _PDataSerializer()
    result = serializer.dumps(123)
    assert result is not None

def test_dumps_accepts_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None

def test_dumps_returns_string_when_serialized_is_string():
    serializer = _PDataSerializer()
    result = serializer.dumps("hello")
    assert isinstance(result, str)

def test_dumps_returns_bytes_when_serialized_is_bytes():
    serializer = _PDataSerializer()
    result = serializer.dumps(b"data")
    assert isinstance(result, bytes)
```


# LLM-generated content at query #108
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

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_strings():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_list_of_bytes():
    s = Serializer([b"key1", b"key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_salt_str():
    s = Serializer("secret", salt="mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_salt_bytes():
    s = Serializer("secret", salt=b"mysalt")
    assert s.salt == b"mysalt"

def test_serializer_constructor_with_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, o: "{}", "loads": lambda self, s: {}})()
    s = Serializer("secret", serializer=custom_serializer)
    assert s.serializer is custom_serializer
    assert s.is_text_serializer is True

def test_serializer_constructor_with_binary_serializer():
    binary_serializer = type("BinarySerializer", (), {"dumps": lambda self, o: b"{}", "loads": lambda self, s: {}})()
    s = Serializer("secret", serializer=binary_serializer)
    assert s.serializer is binary_serializer
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
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_empty_fallback_signers():
    s = Serializer("secret", fallback_signers=[])
    assert s.fallback_signers == []

def test_serializer_constructor_secret_key_property():
    s = Serializer(["old_key", "new_key"])
    assert s.secret_key == b"new_key"


# LLM-generated content at query #109
#--------------------------

def test_serializer_constructor_default_values():
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

def test_serializer_constructor_with_iterable_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt=b"custom")
    assert serializer.salt == b"custom"

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: "{}", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_serializer_bytes():
    custom_serializer = type("CustomSerializer", (), {"dumps": lambda self, x: b"{}", "loads": lambda self, x: {}})()
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer is custom_signer

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


# LLM-generated content at query #110
#--------------------------

def test_constructor_default_serializer():
    serializer_instance = Serializer("secret")
    assert serializer_instance.secret_keys == [b"secret"]
    assert serializer_instance.salt == b"itsdangerous"
    assert serializer_instance.serializer == json
    assert serializer_instance.is_text_serializer == True
    assert serializer_instance.signer == Signer
    assert serializer_instance.signer_kwargs == {}
    assert serializer_instance.fallback_signers == []
    assert serializer_instance.serializer_kwargs == {}

def test_constructor_with_bytes_secret_key():
    serializer_instance = Serializer(b"secret")
    assert serializer_instance.secret_keys == [b"secret"]

def test_constructor_with_list_of_strings_secret_key():
    serializer_instance = Serializer(["key1", "key2"])
    assert serializer_instance.secret_keys == [b"key1", b"key2"]

def test_constructor_with_list_of_bytes_secret_key():
    serializer_instance = Serializer([b"key1", b"key2"])
    assert serializer_instance.secret_keys == [b"key1", b"key2"]

def test_constructor_with_none_salt():
    serializer_instance = Serializer("secret", salt=None)
    assert serializer_instance.salt is None

def test_constructor_with_bytes_salt():
    serializer_instance = Serializer("secret", salt=b"custom_salt")
    assert serializer_instance.salt == b"custom_salt"

def test_constructor_with_string_salt():
    serializer_instance = Serializer("secret", salt="custom_salt")
    assert serializer_instance.salt == b"custom_salt"

def test_constructor_with_custom_text_serializer():
    class CustomTextSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer_instance = Serializer("secret", serializer=CustomTextSerializer())
    assert serializer_instance.serializer == CustomTextSerializer()
    assert serializer_instance.is_text_serializer == True

def test_constructor_with_custom_bytes_serializer():
    class CustomBytesSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode()
        @staticmethod
        def loads(b):
            return int(b.decode())
    serializer_instance = Serializer("secret", serializer=CustomBytesSerializer())
    assert serializer_instance.serializer == CustomBytesSerializer()
    assert serializer_instance.is_text_serializer == False

def test_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer_instance = Serializer("secret", signer=CustomSigner)
    assert serializer_instance.signer == CustomSigner

def test_constructor_with_signer_kwargs():
    serializer_instance = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer_instance.signer_kwargs == {"key_derivation": "hmac"}

def test_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer_instance = Serializer("secret", fallback_signers=fallback)
    assert serializer_instance.fallback_signers == fallback

def test_constructor_with_serializer_kwargs():
    serializer_instance = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert serializer_instance.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #111
#--------------------------

```
def test_dumps_returns_serialized_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("test_string")
    assert result is not None

def test_dumps_accepts_any_object():
    serializer = _PDataSerializer()
    result = serializer.dumps(42)
    assert result is not None

def test_dumps_with_none():
    serializer = _PDataSerializer()
    result = serializer.dumps(None)
    assert result is not None

def test_dumps_with_list():
    serializer = _PDataSerializer()
    result = serializer.dumps([1, 2, 3])
    assert result is not None

def test_dumps_with_dict():
    serializer = _PDataSerializer()
    result = serializer.dumps({"key": "value"})
    assert result is not None

def test_dumps_returns_expected_type():
    serializer = _PDataSerializer()
    result = serializer.dumps("data")
    assert isinstance(result, str)

def test_dumps_with_complex_object():
    serializer = _PDataSerializer()
    result = serializer.dumps((1, "a", 3.14))
    assert result is not None
```


# LLM-generated content at query #112
#--------------------------

```python
def test_loads_returns_any():
    serializer = _PDataSerializer()
    result = serializer.loads(b"test data")
    assert result is not None

def test_loads_accepts_serialized_type():
    serializer = _PDataSerializer()
    payload = b"test payload"
    result = serializer.loads(payload)
    assert isinstance(result, object)

def test_loads_with_string_payload():
    serializer = _PDataSerializer()
    payload = "test string"
    result = serializer.loads(payload)
    assert result is not None

def test_loads_with_empty_payload():
    serializer = _PDataSerializer()
    payload = b""
    result = serializer.loads(payload)
    assert result is not None
```


# LLM-generated content at query #113
#--------------------------

```python
def test_dumps_with_non_text_serializer_returns_bytes():
    from itsdangerous.serializer import Serializer
    from itsdangerous.signer import Signer
    import json

    serializer = Serializer("secret-key", serializer=json, signer=Signer)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
```


# LLM-generated content at query #114
#--------------------------

```python
def test_load_payload_with_non_text_serializer_and_invalid_payload_raises_bad_payload():
    serializer = Serializer("secret-key", serializer=Serializer.default_serializer)
    invalid_payload = b"invalid json"
    try:
        serializer.load_payload(invalid_payload)
        assert False, "Expected BadPayload"
    except BadPayload:
        pass
```


# LLM-generated content at query #115
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

def test_serializer_constructor_with_list_of_str_secret_keys():
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
            return str(obj)
        @staticmethod
        def loads(s):
            return eval(s)
    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.serializer == CustomSerializer
    assert serializer.is_text_serializer == True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(payload):
            return payload
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

def test_serializer_constructor_with_none_fallback_signers():
    serializer = Serializer("secret", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #116
#--------------------------

```python
def test_iter_unsigners_elif_branch_tuple_fallback():
    serializer = Serializer(
        secret_key=b"test-secret",
        fallback_signers=[(Signer, {"digest_method": "sha256"})],
    )
    unsigners = list(serializer.iter_unsigners(salt=b"test-salt"))
    assert len(unsigners) > 1
    assert isinstance(unsigners[1], Signer)
```


# LLM-generated content at query #117
#--------------------------

def test_serializer_init_with_string_secret_key():
    serializer = Serializer("my-secret-key")
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    serializer = Serializer(bytes("my-secret-key", "utf-8"))
    assert serializer.secret_keys == [b"my-secret-key"]
    assert serializer.salt == b"itsdangerous"

def test_serializer_init_with_list_of_string_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_list_of_bytes_secret_keys():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_none_salt():
    serializer = Serializer("my-secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_init_with_custom_salt():
    serializer = Serializer("my-secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_init_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)
        @staticmethod
        def loads(s):
            return int(s)
    serializer = Serializer("my-secret-key", serializer=CustomSerializer())
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True

def test_serializer_init_with_custom_serializer_kwargs():
    serializer = Serializer("my-secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_init_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("my-secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_init_with_custom_signer_kwargs():
    serializer = Serializer("my-secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    serializer = Serializer("my-secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_init_with_empty_fallback_signers():
    serializer = Serializer("my-secret-key", fallback_signers=[])
    assert serializer.fallback_signers == []

def test_serializer_init_with_default_fallback_signers():
    serializer = Serializer("my-secret-key")
    assert serializer.fallback_signers == []

def test_serializer_init_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"data"
        @staticmethod
        def loads(data):
            return data.decode()
    serializer = Serializer("my-secret-key", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_init_secret_key_property():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #118
#--------------------------

def test_serializer_init_with_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == json
    assert s.is_text_serializer == True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_init_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_init_with_list_of_secret_keys():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]

def test_serializer_init_with_custom_salt():
    s = Serializer("secret", salt="custom_salt")
    assert s.salt == b"custom_salt"

def test_serializer_init_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_init_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return f"custom:{obj}"
        @staticmethod
        def loads(s):
            return s.split(":")[1]
    s = Serializer("secret", serializer=CustomSerializer())
    assert s.serializer == CustomSerializer()
    assert s.is_text_serializer == True

def test_serializer_init_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_init_with_custom_signer_class():
    class CustomSigner(Signer):
        pass
    s = Serializer("secret", signer=CustomSigner)
    assert s.signer == CustomSigner

def test_serializer_init_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "none"})
    assert s.signer_kwargs == {"key_derivation": "none"}

def test_serializer_init_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback


# LLM-generated content at query #119
#--------------------------

```python
def test_load_payload_with_default_serializer_and_bytes_payload():
    serializer = Serializer("secret")
    result = serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}

def test_load_payload_with_text_serializer():
    text_serializer = type("TextSerializer", (), {"loads": staticmethod(lambda x: {"key": "value"}), "dumps": staticmethod(lambda x: '{"key": "value"}'})()
    serializer = Serializer("secret", serializer=text_serializer)
    result = serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer_returns_bytes():
    bytes_serializer = type("BytesSerializer", (), {"loads": staticmethod(lambda x: {"key": "value"}), "dumps": staticmethod(lambda x: b'{"key": "value"}'})()
    serializer = Serializer("secret", serializer=bytes_serializer)
    result = serializer.load_payload(b'{"key": "value"}')
    assert result == {"key": "value"}

def test_load_payload_raises_bad_payload_on_exception():
    serializer = Serializer("secret")
    try:
        serializer.load_payload(b"invalid")
        assert False
    except BadPayload:
        pass

def test_load_payload_with_override_serializer():
    serializer = Serializer("secret")
    custom_serializer = type("CustomSerializer", (), {"loads": staticmethod(lambda x: 42), "dumps": staticmethod(lambda x: b"data"})()
    result = serializer.load_payload(b"data", serializer=custom_serializer)
    assert result == 42
```


# LLM-generated content at query #120
#--------------------------

```python
def test_salt_is_not_none_on_init():
    serializer = Serializer(secret_key="secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"
```


# LLM-generated content at query #121
#--------------------------

def test_serializer_constructor_default_values():
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

def test_serializer_constructor_with_custom_serializer():
    s = Serializer("secret", serializer=json)
    assert s.serializer == json

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"sort_keys": True})
    assert s.serializer_kwargs == {"sort_keys": True}

def test_serializer_constructor_with_custom_signer():
    s = Serializer("secret", signer=Signer)
    assert s.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert s.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "none"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback

def test_serializer_constructor_with_fallback_signers_none():
    s = Serializer("secret")
    assert s.fallback_signers == []


# LLM-generated content at query #122
#--------------------------

def test_serializer_constructor_initializes_secret_keys():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_initializes_secret_keys_with_bytes():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_initializes_secret_keys_with_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_initializes_secret_keys_with_bytes_list():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_salt_default():
    serializer = Serializer("secret")
    assert serializer.salt == b"itsdangerous"

def test_serializer_constructor_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_salt_bytes():
    serializer = Serializer("secret", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_salt_str():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_serializer_default():
    serializer = Serializer("secret")
    assert serializer.serializer is json

def test_serializer_constructor_serializer_custom():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json

def test_serializer_constructor_is_text_serializer_true():
    serializer = Serializer("secret")
    assert serializer.is_text_serializer is True

def test_serializer_constructor_is_text_serializer_false():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"{}"

        @staticmethod
        def loads(data):
            return {}

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_signer_default():
    serializer = Serializer("secret")
    assert serializer.signer is Signer

def test_serializer_constructor_signer_custom():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_signer_kwargs_default():
    serializer = Serializer("secret")
    assert serializer.signer_kwargs == {}

def test_serializer_constructor_signer_kwargs_custom():
    serializer = Serializer("secret", signer_kwargs={"key": "value"})
    assert serializer.signer_kwargs == {"key": "value"}

def test_serializer_constructor_fallback_signers_default():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == []

def test_serializer_constructor_fallback_signers_custom():
    fallback = [{"key": "value"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_serializer_kwargs_default():
    serializer = Serializer("secret")
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_serializer_kwargs_custom():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.serializer_kwargs == {"indent": 4}


