####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serializer_constructor_with_single_string_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_single_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_multiple_string_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_multiple_bytes_keys():
    serializer = Serializer([b"secret1", b"secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return b"bytes"

        @staticmethod
        def loads(s):
            return {"data": s}

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #2
#--------------------------

```python
def test_load_payload_with_text_serializer():
    serializer = Serializer("secret-key", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return pickle.dumps(obj)

        def loads(self, payload):
            return pickle.loads(payload)

    serializer = Serializer("secret-key", serializer=BinarySerializer())
    payload = pickle.dumps({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj).encode()

        def loads(self, payload):
            return int(payload.decode())

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = b"42"
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret-key", serializer=json)
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #3
#--------------------------

```python
def test_iter_unsigners_default():
    serializer = Serializer("secret-key")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"secret-key"]
    assert unsigners[0].salt == b"itsdangerous"

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert unsigners[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    fallback = {"digest_method": SHA256}
    serializer = Serializer("secret-key", fallback_signers=[fallback])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].digest_method.name == "sha256"

def test_iter_unsigners_with_fallback_signers_tuple():
    fallback = (Signer, {"digest_method": SHA256})
    serializer = Serializer("secret-key", fallback_signers=[fallback])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].digest_method.name == "sha256"

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)

def test_iter_unsigners_with_multiple_fallbacks():
    fallback1 = {"digest_method": SHA256}
    fallback2 = (Signer, {"digest_method": SHA512})
    serializer = Serializer("secret-key", fallback_signers=[fallback1, fallback2])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3
    assert unsigners[1].digest_method.name == "sha256"
    assert unsigners[2].digest_method.name == "sha512"

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert unsigners[0].secret_keys == [b"old-key", b"new-key"]

def test_iter_unsigners_with_key_rotation_and_fallback():
    fallback = {"digest_method": SHA256}
    serializer = Serializer(["old-key", "new-key"], fallback_signers=[fallback])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert unsigners[0].secret_keys == [b"old-key", b"new-key"]
    assert unsigners[1].secret_keys == [b"old-key"]
    assert unsigners[1].digest_method.name == "sha256"


# LLM-generated content at query #4
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
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
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #5
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
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
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer={"dumps": lambda x: "custom", "loads": lambda x: {}})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: "custom", "loads": lambda x: {}}
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=TimedSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == TimedSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "|"}, (TimedSigner, {"sep": "|"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #6
#--------------------------

```python
def test_serializer_constructor_defaults():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_str_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_iterable_keys():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_keys == [b"old", b"new"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer(b"secret", salt="custom")
    assert serializer.salt == b"custom"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer(b"secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x)}
    serializer = Serializer(b"secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer(b"secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer(b"secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer(b"secret", signer_kwargs={"sep": "|"})
    assert serializer.signer_kwargs == {"sep": "|"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"digest_method": SHA256}]
    serializer = Serializer(b"secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_secret_key_property():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_key == b"new"


# LLM-generated content at query #7
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key", serializer=json)
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert serializer.loads(result) == obj

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"ensure_ascii": False})
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, bytes)
    assert serializer.loads(result) == obj

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert serializer.loads(result, salt="custom-salt") == obj

def test_dumps_with_different_salt():
    serializer = Serializer("secret-key", salt="salt1")
    obj = {"key": "value"}
    result = serializer.dumps(obj, salt="salt2")
    assert serializer.loads(result, salt="salt2") == obj

def test_dumps_with_list_secret_key():
    serializer = Serializer(["secret-key1", "secret-key2"])
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert serializer.loads(result) == obj

def test_dumps_with_fallback_signers():
    fallback_signer = {"key_derivation": "hmac", "digest_method": "sha256"}
    serializer = Serializer("secret-key", fallback_signers=[fallback_signer])
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert serializer.loads(result) == obj

def test_dumps_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert serializer.loads(result) == obj

def test_dumps_with_empty_object():
    serializer = Serializer("secret-key")
    obj = {}
    result = serializer.dumps(obj)
    assert serializer.loads(result) == obj

def test_dumps_with_none_object():
    serializer = Serializer("secret-key")
    obj = None
    result = serializer.dumps(obj)
    assert serializer.loads(result) == obj

def test_dumps_with_complex_object():
    serializer = Serializer("secret-key")
    obj = {"key": "value", "list": [1, 2, 3], "nested": {"a": "b"}}
    result = serializer.dumps(obj)
    assert serializer.loads(result) == obj


# LLM-generated content at query #8
#--------------------------

```python
def test_serializer_constructor_defaults():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_str_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_iterable_keys():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_keys == [b"old", b"new"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer(b"secret", salt="custom")
    assert serializer.salt == b"custom"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer(b"secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer(b"secret", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer(b"secret", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer(b"secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer(b"secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer(b"secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_secret_key_property():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_key == b"new"


# LLM-generated content at query #9
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
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
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x)}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #10
#--------------------------

```python
def test_serializer_constructor_with_string_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_string_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_bytes_keys():
    serializer = Serializer([b"secret1", b"secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: "custom"}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer():
    custom_signer = Signer
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: "custom"}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #11
#--------------------------

```python
def test_serializer_constructor_defaults():
    s = Serializer("secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    s = Serializer(b"secret-key")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_key_list():
    s = Serializer(["key1", "key2"])
    assert s.secret_keys == [b"key1", b"key2"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret-key", salt="custom-salt")
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"custom-salt"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    s = Serializer("secret-key", serializer=serializer)
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is serializer
    assert s.is_text_serializer is False
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    s = Serializer("secret-key", signer=TimedSerializer)
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is TimedSerializer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret-key", signer_kwargs={"sep": "|"})
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {"sep": "|"}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"sep": "|"}, (TimedSerializer, {"sep": "|"}), TimedSerializer]
    s = Serializer("secret-key", fallback_signers=fallback)
    assert s.secret_keys == [b"secret-key"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == fallback
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret-key", salt=None)
    assert s.secret_keys == [b"secret-key"]
    assert s.salt is None
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_secret_key_property():
    s = Serializer(["key1", "key2"])
    assert s.secret_key == b"key2"


# LLM-generated content at query #12
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
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
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"sep": "?"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": "?"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "?"}, (Signer, {"sep": "!"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #13
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_iterable_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key1": "value1"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key1": "value1"}]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key1": "value1"}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #14
#--------------------------

```python
def test_salt_none_predicate():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #15
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_iterable_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    custom_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key1": "value1"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key1": "value1"}, (Signer, {"key2": "value2"})]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key1": "value1"}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #16
#--------------------------

```python
def test_serializer_defaults():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_with_str_secret_key():
    serializer = Serializer("secret-key", salt="salt", serializer=json, signer=Signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_with_bytes_secret_key():
    serializer = Serializer(b"secret-key", salt=b"salt", serializer=json, signer=Signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_with_list_secret_keys():
    serializer = Serializer(["key1", "key2"], salt="salt", serializer=json, signer=Signer)
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_with_none_salt():
    serializer = Serializer("secret-key", salt=None, serializer=json, signer=Signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"sep": "|"})
    assert serializer.signer_kwargs == {"sep": "|"}

def test_serializer_with_fallback_signers():
    fallback = [{"sep": "|"}, (Signer, {"digest_method": hashlib.sha256})]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(data):
            return json.dumps(data).encode("utf-8")

        @staticmethod
        def loads(data):
            return json.loads(data.decode("utf-8"))

    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False


# LLM-generated content at query #17
#--------------------------

```python
def test_dumps_serializes_object():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)


# LLM-generated content at query #18
#--------------------------

```python
def test_serializer_default_serializer():
    serializer = Serializer(b"secret")
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_custom_text_serializer():
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: x}
    serializer = Serializer(b"secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_custom_bytes_serializer():
    custom_serializer = {"dumps": lambda x: bytes(x, "utf-8"), "loads": lambda x: x}
    serializer = Serializer(b"secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_secret_key_bytes():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.secret_key == b"secret"

def test_serializer_secret_key_str():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.secret_key == b"secret"

def test_serializer_secret_key_list():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.secret_key == b"secret2"

def test_serializer_salt_bytes():
    serializer = Serializer(b"secret", salt=b"salt")
    assert serializer.salt == b"salt"

def test_serializer_salt_str():
    serializer = Serializer(b"secret", salt="salt")
    assert serializer.salt == b"salt"

def test_serializer_salt_none():
    serializer = Serializer(b"secret", salt=None)
    assert serializer.salt is None

def test_serializer_default_signer():
    serializer = Serializer(b"secret")
    assert serializer.signer is Signer

def test_serializer_custom_signer():
    serializer = Serializer(b"secret", signer=TimedSerializer)
    assert serializer.signer is TimedSerializer

def test_serializer_signer_kwargs():
    serializer = Serializer(b"secret", signer_kwargs={"sep": "?"})
    assert serializer.signer_kwargs == {"sep": "?"}

def test_serializer_default_fallback_signers():
    serializer = Serializer(b"secret")
    assert serializer.fallback_signers == []

def test_serializer_custom_fallback_signers():
    fallback = [{"digest_method": SHA256}]
    serializer = Serializer(b"secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_serializer_kwargs():
    serializer = Serializer(b"secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #19
#--------------------------

```python
def test_serializer_constructor_defaults():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_str_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_iterable_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer(b"secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer(b"secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {}}
    serializer = Serializer(b"secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    custom_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {}}
    serializer = Serializer(b"secret", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_signer():
    serializer = Serializer(b"secret", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer(b"secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer(b"secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer(b"secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_secret_key_property():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_key == b"secret2"


# LLM-generated content at query #20
#--------------------------

```python
def test_load_payload_with_non_text_serializer():
    serializer = Serializer(b"secret-key")
    payload = b"some-payload"
    result = serializer.load_payload(payload, serializer=json)
    assert result is not None


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: "custom"}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key": "value"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key": "value"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key": "value"}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #2
#--------------------------

```python
def test_is_text_serializer_false():
    serializer = Serializer(b"secret", serializer=json)
    assert serializer.is_text_serializer is False


# LLM-generated content at query #3
#--------------------------

```python
def test_salt_is_none_when_not_provided():
    serializer = Serializer(b"secret-key")
    assert serializer.salt is None


# LLM-generated content at query #4
#--------------------------

```python
def test_load_payload_with_text_serializer():
    serializer = Serializer("secret-key", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return pickle.dumps(obj)

        def loads(self, payload):
            return pickle.loads(payload)

    serializer = Serializer("secret-key", serializer=BinarySerializer())
    payload = pickle.dumps({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj).encode()

        def loads(self, payload):
            return int(payload.decode())

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = b"42"
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret-key", serializer=json)
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #5
#--------------------------

```python
def test_salt_is_none():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #6
#--------------------------

```python
def test_dumps_with_text_serializer():
    serializer = Serializer("secret-key", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.startswith('{"key": "value"}')

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"ensure_ascii": False})
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result.startswith(b'{"key": "value"}')

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_with_different_secret_key():
    serializer = Serializer("different-secret", salt="salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_with_list_secret_key():
    serializer = Serializer(["key1", "key2"], salt="salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_with_bytes_input():
    serializer = Serializer("secret-key")
    result = serializer.dumps(b"bytes input")
    assert isinstance(result, str)

def test_dumps_with_unicode_input():
    serializer = Serializer("secret-key")
    result = serializer.dumps("unicode input")
    assert isinstance(result, str)

def test_dumps_with_empty_input():
    serializer = Serializer("secret-key")
    result = serializer.dumps("")
    assert isinstance(result, str)

def test_dumps_with_none_input():
    serializer = Serializer("secret-key")
    result = serializer.dumps(None)
    assert isinstance(result, str)

def test_dumps_with_complex_input():
    serializer = Serializer("secret-key")
    data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
    result = serializer.dumps(data)
    assert isinstance(result, str)


# LLM-generated content at query #7
#--------------------------

```python
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
    assert s.salt == b"itsdangerous"

def test_serializer_constructor_with_key_list():
    s = Serializer(["secret1", "secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_bytes_key_list():
    s = Serializer([b"secret1", b"secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom")
    assert s.salt == b"custom"

def test_serializer_constructor_with_bytes_salt():
    s = Serializer("secret", salt=b"custom")
    assert s.salt == b"custom"

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_constructor_with_custom_serializer():
    s = Serializer("secret", serializer=json)
    assert s.serializer is json
    assert s.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    s = Serializer("secret", serializer=BytesSerializer)
    assert s.serializer is BytesSerializer
    assert s.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    s = Serializer("secret", signer=Signer)
    assert s.signer is Signer

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"sep": "|"})
    assert s.signer_kwargs == {"sep": "|"}

def test_serializer_constructor_with_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"sep": "|"}])
    assert s.fallback_signers == [{"sep": "|"}]

def test_serializer_secret_key_property():
    s = Serializer(["secret1", "secret2"])
    assert s.secret_key == b"secret2"


# LLM-generated content at query #8
#--------------------------

```python
def test_dumps_serializes_object_correctly():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)  # Assuming _TSerialized is str for this test


# LLM-generated content at query #9
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key", salt="test-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_key == b"secret-key"
    assert signers[0].salt == b"test-salt"

def test_iter_unsigners_with_fallback_dict():
    fallback_signers = [{"digest_method": SHA256}]
    serializer = Serializer("secret-key", salt="test-salt", fallback_signers=fallback_signers)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_key == b"secret-key"
    assert signers[0].salt == b"test-salt"
    assert isinstance(signers[1], Signer)
    assert signers[1].secret_key == b"secret-key"
    assert signers[1].salt == b"test-salt"
    assert signers[1].digest_method.name == "sha256"

def test_iter_unsigners_with_fallback_tuple():
    fallback_signers = [(Signer, {"digest_method": SHA256})]
    serializer = Serializer("secret-key", salt="test-salt", fallback_signers=fallback_signers)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_key == b"secret-key"
    assert signers[0].salt == b"test-salt"
    assert isinstance(signers[1], Signer)
    assert signers[1].secret_key == b"secret-key"
    assert signers[1].salt == b"test-salt"
    assert signers[1].digest_method.name == "sha256"

def test_iter_unsigners_with_fallback_class():
    fallback_signers = [Signer]
    serializer = Serializer("secret-key", salt="test-salt", fallback_signers=fallback_signers)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_key == b"secret-key"
    assert signers[0].salt == b"test-salt"
    assert isinstance(signers[1], Signer)
    assert signers[1].secret_key == b"secret-key"
    assert signers[1].salt == b"test-salt"

def test_iter_unsigners_with_multiple_fallbacks():
    fallback_signers = [
        {"digest_method": SHA256},
        (Signer, {"digest_method": SHA512}),
        Signer
    ]
    serializer = Serializer("secret-key", salt="test-salt", fallback_signers=fallback_signers)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    assert all(isinstance(signer, Signer) for signer in signers)
    assert all(signer.secret_key == b"secret-key" for signer in signers)
    assert all(signer.salt == b"test-salt" for signer in signers)
    assert signers[1].digest_method.name == "sha256"
    assert signers[2].digest_method.name == "sha512"

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"], salt="test-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[0].salt == b"test-salt"

def test_iter_unsigners_with_key_rotation_and_fallbacks():
    fallback_signers = [{"digest_method": SHA256}]
    serializer = Serializer(["old-key", "new-key"], salt="test-salt", fallback_signers=fallback_signers)
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert all(isinstance(signer, Signer) for signer in signers)
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[0].salt == b"test-salt"
    assert signers[1].secret_key == b"old-key"
    assert signers[1].salt == b"test-salt"
    assert signers[1].digest_method.name == "sha256"
    assert signers[2].secret_key == b"new-key"
    assert signers[2].salt == b"test-salt"
    assert signers[2].digest_method.name == "sha256"


# LLM-generated content at query #10
#--------------------------

```python
def test_serializer_not_none():
    serializer = Serializer(b"secret", serializer="custom")
    assert serializer.serializer == "custom"


# LLM-generated content at query #11
#--------------------------

```python
def test_dumps_returns_bytes_when_not_text_serializer():
    serializer = Serializer(b"secret", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #12
#--------------------------

```python
def test_isinstance_dict_predicate():
    serializer = Serializer(b"secret", fallback_signers=[{"key1": "value1"}])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[1], Signer)


# LLM-generated content at query #13
#--------------------------

```python
def test_load_payload_with_non_text_serializer_raises_bad_payload():
    serializer = Serializer(b"secret", serializer=json)
    payload = b"invalid-json"
    try:
        serializer.load_payload(payload)
    except BadPayload as e:
        assert e.original_error is not None
    else:
        assert False, "Expected BadPayload to be raised"


# LLM-generated content at query #14
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert isinstance(serializer.secret_keys, list)
    assert len(serializer.secret_keys) == 1
    assert serializer.secret_keys[0] == b"secret-key"
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert isinstance(serializer.secret_keys, list)
    assert len(serializer.secret_keys) == 1
    assert serializer.secret_keys[0] == b"secret-key"
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["old-key", "new-key"])
    assert isinstance(serializer.secret_keys, list)
    assert len(serializer.secret_keys) == 2
    assert serializer.secret_keys[0] == b"old-key"
    assert serializer.secret_keys[1] == b"new-key"
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret-key", serializer=BytesSerializer)
    assert serializer.serializer == BytesSerializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret-key", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}, (Signer, {"key_derivation": "hmac"})]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #15
#--------------------------

```python
def test_iter_unsigners_with_empty_secret_keys():
    serializer = Serializer(secret_key="test_key")
    serializer.secret_keys = []
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1


# LLM-generated content at query #16
#--------------------------

```python
def test_loads_deserializes_payload():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert isinstance(result, dict)
    assert result == {"key": "value"}


# LLM-generated content at query #17
#--------------------------

```python
def test_serializer_constructor_with_string_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == Serializer.default_fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == Serializer.default_fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == Serializer.default_fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == Serializer.default_fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == Serializer.default_fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == Serializer.default_fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == Serializer.default_fallback_signers
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    custom_signer = Signer
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == Serializer.default_fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"sep": "?"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": "?"}
    assert serializer.fallback_signers == Serializer.default_fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "?"}, (Signer, {"sep": "!"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_secret_key_property():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_key == b"secret2"


# LLM-generated content at query #18
#--------------------------

```python
def test_load_payload_with_non_text_serializer():
    serializer = Serializer(b"secret", serializer=json)
    payload = b"invalid_json"
    try:
        serializer.load_payload(payload)
    except BadPayload as e:
        assert isinstance(e.original_error, json.JSONDecodeError)


# LLM-generated content at query #19
#--------------------------

```python
def test_salt_is_none_predicate():
    serializer = Serializer(secret_key="test")
    assert serializer.salt is None


# LLM-generated content at query #20
#--------------------------

```python
def test_salt_none_when_none_provided():
    serializer = Serializer(secret_key="test")
    assert serializer.salt is None


# LLM-generated content at query #21
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_key == b"secret-key"
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_with_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_key == b"secret-key"
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": SHA256}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[0].digest_name == "md5"
    assert signers[1].digest_name == "sha256"

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(Signer, {"digest_method": SHA256})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[0].digest_name == "md5"
    assert signers[1].digest_name == "sha256"

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[Signer]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[0].digest_name == "md5"
    assert signers[1].digest_name == "md5"

def test_iter_unsigners_with_multiple_fallback_signers():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"digest_method": SHA256},
            (Signer, {"digest_method": SHA512}),
            Signer
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert isinstance(signers[2], Signer)
    assert isinstance(signers[3], Signer)
    assert signers[0].digest_name == "md5"
    assert signers[1].digest_name == "sha256"
    assert signers[2].digest_name == "sha512"
    assert signers[3].digest_name == "md5"

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_key == b"new-key"
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_with_key_rotation_and_fallback():
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": SHA256}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert isinstance(signers[2], Signer)
    assert signers[0].secret_key == b"new-key"
    assert signers[1].secret_key == b"old-key"
    assert signers[2].secret_key == b"new-key"
    assert signers[0].digest_name == "md5"
    assert signers[1].digest_name == "sha256"
    assert signers[2].digest_name == "sha256"


# LLM-generated content at query #22
#--------------------------

```python
def test_dumps_with_default_serializer():
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.startswith('{"key": "value"}')

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, salt=b"salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result, str)

def test_dumps_with_different_object_types():
    serializer = Serializer("secret-key")
    assert isinstance(serializer.dumps(123), str)
    assert isinstance(serializer.dumps([1, 2, 3]), str)
    assert isinstance(serializer.dumps("string"), str)

def test_dumps_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    result = serializer.dumps({"key": "value"})
    assert "\n" in result


# LLM-generated content at query #23
#--------------------------

```python
def test_iter_unsigners_default():
    serializer = Serializer("secret-key")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"secret-key"]
    assert unsigners[0].salt == b"itsdangerous"

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"secret-key"]
    assert unsigners[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    fallback = {"signer_kwargs": {"sep": "|"}}
    serializer = Serializer("secret-key", fallback_signers=[fallback])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"secret-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == [b"secret-key"]
    assert unsigners[1].salt == b"itsdangerous"
    assert unsigners[1].sep == "|"

def test_iter_unsigners_with_fallback_signers_tuple():
    fallback = (Signer, {"sep": "|"})
    serializer = Serializer("secret-key", fallback_signers=[fallback])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"secret-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == [b"secret-key"]
    assert unsigners[1].salt == b"itsdangerous"
    assert unsigners[1].sep == "|"

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"secret-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == [b"secret-key"]
    assert unsigners[1].salt == b"itsdangerous"

def test_iter_unsigners_with_multiple_fallback_signers():
    fallback1 = {"signer_kwargs": {"sep": "|"}}
    fallback2 = (Signer, {"sep": "."})
    serializer = Serializer("secret-key", fallback_signers=[fallback1, fallback2])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"secret-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == [b"secret-key"]
    assert unsigners[1].salt == b"itsdangerous"
    assert unsigners[1].sep == "|"
    assert isinstance(unsigners[2], Signer)
    assert unsigners[2].secret_keys == [b"secret-key"]
    assert unsigners[2].salt == b"itsdangerous"
    assert unsigners[2].sep == "."

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"old-key", b"new-key"]
    assert unsigners[0].salt == b"itsdangerous"

def test_iter_unsigners_with_key_rotation_and_fallback():
    fallback = {"signer_kwargs": {"sep": "|"}}
    serializer = Serializer(["old-key", "new-key"], fallback_signers=[fallback])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"old-key", b"new-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == [b"old-key"]
    assert unsigners[1].salt == b"itsdangerous"
    assert unsigners[1].sep == "|"
    assert isinstance(unsigners[2], Signer)
    assert unsigners[2].secret_keys == [b"new-key"]
    assert unsigners[2].salt == b"itsdangerous"
    assert unsigners[2].sep == "|"


# LLM-generated content at query #24
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_iterable_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = type('CustomSerializer', (), {'dumps': lambda self, obj: str(obj), 'loads': lambda self, s: s})
    serializer = Serializer("secret", serializer=custom_serializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer()
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type('CustomSigner', (), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key1": "value1"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key1": "value1"}, (type('CustomSigner', (), {}), {"key2": "value2"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key1": "value1"}

def test_serializer_secret_key_property():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_key == b"secret2"


# LLM-generated content at query #25
#--------------------------

```python
def test_load_payload_with_invalid_payload():
    serializer = Serializer(b"secret-key")
    payload = b"invalid-payload"
    try:
        serializer.load_payload(payload)
    except BadPayload as e:
        assert e.original_error is not None


# LLM-generated content at query #26
#--------------------------

```python
def test_salt_is_none():
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #27
#--------------------------

```python
def test_serializer_constructor_with_string_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key1": "value1"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key1": "value1"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key1": "value1"}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #28
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: x, "loads": lambda x: x}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret-key", signer=Signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key": "value"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key": "value"}]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key": "value"}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #29
#--------------------------

```python
def test_serializer_constructor_with_string_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_keys():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner:
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key": "value"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key": "value"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key": "value"}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #30
#--------------------------

```python
def test_serializer_constructor_with_string_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_keys():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #31
#--------------------------

```python
def test_salt_is_none():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #32
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #33
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.startswith('{"key": "value"}.')

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"ensure_ascii": False})
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert result.startswith(b'{"key": "value"}.')

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.startswith('{"key": "value"}.')

def test_dumps_with_different_secret_key():
    serializer = Serializer("different-secret", salt="custom-salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.startswith('{"key": "value"}.')

def test_dumps_with_list_secret_key():
    serializer = Serializer(["key1", "key2"], salt="custom-salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.startswith('{"key": "value"}.')

def test_dumps_with_bytes_input():
    serializer = Serializer("secret-key")
    result = serializer.dumps(b"bytes input")
    assert isinstance(result, str)
    assert result.startswith('"bytes input".')

def test_dumps_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.startswith('{"key": "value"}.')

def test_dumps_with_custom_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return f"custom-{obj}"

    serializer = Serializer("secret-key", serializer=CustomSerializer)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.startswith("custom-{")

def test_dumps_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"indent": 2})
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert result.startswith('{\n  "key": "value"\n}.')

def test_dumps_with_empty_object():
    serializer = Serializer("secret-key")
    result = serializer.dumps({})
    assert isinstance(result, str)
    assert result.startswith('{}.')


# LLM-generated content at query #34
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret-key-1", "secret-key-2"])
    assert serializer.secret_keys == [b"secret-key-1", b"secret-key-2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: x}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    custom_signer = Signer
    serializer = Serializer("secret-key", signer=custom_signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #35
#--------------------------

```python
def test_load_payload_with_non_text_serializer():
    serializer = Serializer(b"secret", serializer=json)
    assert not serializer.is_text_serializer
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #36
#--------------------------

```python
def test_dumps_serializes_object():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    serialized = serializer.dumps(obj)
    assert isinstance(serialized, type(serializer.dumps(obj)))


# LLM-generated content at query #37
#--------------------------

```python
def test_serializer_default_serializer():
    serializer = Serializer("secret-key")
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True

def test_serializer_custom_text_serializer():
    custom_serializer = lambda obj: str(obj)
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_custom_bytes_serializer():
    custom_serializer = lambda obj: bytes(str(obj), "utf-8")
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_default_signer():
    serializer = Serializer("secret-key")
    assert serializer.signer == Signer

def test_serializer_custom_signer():
    custom_signer = Signer
    serializer = Serializer("secret-key", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_default_fallback_signers():
    serializer = Serializer("secret-key")
    assert serializer.fallback_signers == []

def test_serializer_custom_fallback_signers():
    fallback_signers = [{"key": "value"}]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_secret_keys_single():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_secret_keys_multiple():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_salt_default():
    serializer = Serializer("secret-key")
    assert serializer.salt == b"itsdangerous"

def test_serializer_salt_custom():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_salt_none():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_serializer_kwargs():
    serializer_kwargs = {"indent": 2}
    serializer = Serializer("secret-key", serializer_kwargs=serializer_kwargs)
    assert serializer.serializer_kwargs == serializer_kwargs

def test_serializer_signer_kwargs():
    signer_kwargs = {"sep": ";"}
    serializer = Serializer("secret-key", signer_kwargs=signer_kwargs)
    assert serializer.signer_kwargs == signer_kwargs


# LLM-generated content at query #38
#--------------------------

```python
def test_load_payload_with_non_text_serializer():
    serializer = Serializer(b"secret-key", serializer=json)
    payload = b"invalid-json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #39
#--------------------------

```python
def test_serializer_default_serializer():
    serializer = Serializer(b"secret-key")
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_custom_serializer():
    custom_serializer = {"dumps": lambda x: x, "loads": lambda x: x}
    serializer = Serializer(b"secret-key", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_default_signer():
    serializer = Serializer(b"secret-key")
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}

def test_serializer_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer(b"secret-key", signer=custom_signer)
    assert serializer.signer is custom_signer
    assert serializer.signer_kwargs == {}

def test_serializer_signer_kwargs():
    serializer = Serializer(b"secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_default_fallback_signers():
    serializer = Serializer(b"secret-key")
    assert serializer.fallback_signers == []

def test_serializer_custom_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer(b"secret-key", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_secret_keys_single():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.secret_key == b"secret-key"

def test_serializer_secret_keys_multiple():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.secret_key == b"key2"

def test_serializer_salt_bytes():
    serializer = Serializer(b"secret-key", salt=b"custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_salt_str():
    serializer = Serializer(b"secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_salt_none():
    serializer = Serializer(b"secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_serializer_kwargs():
    serializer = Serializer(b"secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #40
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
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
    assert serializer.serializer == CustomSerializer()
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}, (Signer, {"key_derivation": "hmac"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #41
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    custom_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key1": "value1"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key1": "value1"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key1": "value1"}

def test_serializer_secret_key_property():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_key == b"secret2"


# LLM-generated content at query #42
#--------------------------

```python
def test_dumps_serializes_object():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)  # Assuming _TSerialized is str for this test


# LLM-generated content at query #43
#--------------------------

```python
def test_load_payload_with_non_text_serializer():
    serializer = Serializer(b"secret-key")
    payload = b"some-payload"
    result = serializer.load_payload(payload, serializer=json)
    assert result is None


# LLM-generated content at query #44
#--------------------------

```python
def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    serialized = serializer.dumps(obj)
    assert isinstance(serialized, _TSerialized)


# LLM-generated content at query #45
#--------------------------

```python
def test_serializer_constructor_with_string_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_string_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_bytes_keys():
    serializer = Serializer([b"secret1", b"secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"sep": "?"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": "?"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "?"}, (Signer, {"sep": "!"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #46
#--------------------------

```python
def test_serializer_not_none():
    serializer = Serializer(b"secret-key", serializer="custom_serializer")
    assert serializer.serializer == "custom_serializer"


# LLM-generated content at query #47
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret-key", serializer=json)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret-key", signer=Signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key": "value"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key": "value"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key": "value"}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret-key", salt=b"custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_secret_key_property():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #48
#--------------------------

```python
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

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_bytes_key_list():
    serializer = Serializer([b"key1", b"key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret-key", serializer=bytes_serializer)
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret-key", signer=Signer)
    assert serializer.signer is Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key1": "value1"})
    assert serializer.signer_kwargs == {"key1": "value1"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key1": "value1"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"key1": "value1"})
    assert serializer.serializer_kwargs == {"key1": "value1"}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_secret_key_property():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #49
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
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
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #50
#--------------------------

```python
def test_load_payload_with_default_serializer():
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def loads(self, payload):
            return payload.decode("utf-8").upper()

        def dumps(self, obj):
            return obj.encode("utf-8")

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = b"hello"
    result = serializer.load_payload(payload)
    assert result == "HELLO"

def test_load_payload_with_text_serializer():
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload, serializer=json)
    assert result == {"key": "value"}

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret-key")
    payload = b"invalid-json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #51
#--------------------------

```python
def test_serializer_not_none():
    serializer = Serializer(b"secret-key", serializer="custom_serializer")
    assert serializer.serializer == "custom_serializer"


# LLM-generated content at query #52
#--------------------------

```python
def test_iter_unsigners_empty_fallback_signers():
    serializer = Serializer(b"secret", fallback_signers=[])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1


# LLM-generated content at query #53
#--------------------------

```python
def test_serializer_constructor_with_single_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_multiple_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #54
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac", "digest_method": "sha256"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #55
#--------------------------

```python
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
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_bytes_key_list():
    serializer = Serializer([b"secret1", b"secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom")
    assert serializer.salt == b"custom"

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom")
    assert serializer.salt == b"custom"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer=json)
    assert serializer.serializer is json

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(data):
            return json.dumps(data).encode("utf-8")

        @staticmethod
        def loads(data):
            return json.loads(data.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.signer is Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"sep": "|"})
    assert serializer.signer_kwargs == {"sep": "|"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"digest_method": SHA256}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_secret_key_property():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_key == b"secret2"


# LLM-generated content at query #56
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer("secret_key")
    assert serializer.secret_keys == [b"secret_key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(b"secret_key")
    assert serializer.secret_keys == [b"secret_key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_iterable_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret_key", salt="custom_salt")
    assert serializer.secret_keys == [b"secret_key"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret_key", serializer=json)
    assert serializer.secret_keys == [b"secret_key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret_key", signer=Signer)
    assert serializer.secret_keys == [b"secret_key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret_key", signer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret_key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key": "value"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_fallback_signers():
    serializer = Serializer("secret_key", fallback_signers=[Signer])
    assert serializer.secret_keys == [b"secret_key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == [Signer]
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret_key", serializer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret_key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key": "value"}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret_key", salt=None)
    assert serializer.secret_keys == [b"secret_key"]
    assert serializer.salt is None
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret_key", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret_key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #57
#--------------------------

```python
def test_load_payload_with_bytes_serializer():
    serializer = Serializer(b"secret", serializer=json)
    assert not serializer.is_text_serializer


# LLM-generated content at query #58
#--------------------------

```python
def test_serializer_default_serializer():
    serializer = Serializer(b"secret")
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_custom_serializer():
    custom_serializer = {"dumps": lambda x: x, "loads": lambda x: x}
    serializer = Serializer(b"secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_default_signer():
    serializer = Serializer(b"secret")
    assert serializer.signer is Signer

def test_serializer_custom_signer():
    class CustomSigner(Signer):
        pass

    serializer = Serializer(b"secret", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_default_fallback_signers():
    serializer = Serializer(b"secret")
    assert serializer.fallback_signers == []

def test_serializer_custom_fallback_signers():
    fallback_signers = [{"key": "value"}]
    serializer = Serializer(b"secret", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_secret_keys():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_secret_key_property():
    serializer = Serializer(b"secret")
    assert serializer.secret_key == b"secret"

def test_serializer_salt():
    serializer = Serializer(b"secret", salt="salt")
    assert serializer.salt == b"salt"

def test_serializer_salt_none():
    serializer = Serializer(b"secret", salt=None)
    assert serializer.salt is None

def test_serializer_serializer_kwargs():
    serializer_kwargs = {"key": "value"}
    serializer = Serializer(b"secret", serializer_kwargs=serializer_kwargs)
    assert serializer.serializer_kwargs == serializer_kwargs

def test_serializer_signer_kwargs():
    signer_kwargs = {"key": "value"}
    serializer = Serializer(b"secret", signer_kwargs=signer_kwargs)
    assert serializer.signer_kwargs == signer_kwargs

def test_serializer_key_rotation():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.secret_key == b"secret2"


# LLM-generated content at query #59
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
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
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_iterable_secret_key():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = Serializer("secret", serializer={"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    assert serializer.is_text_serializer is False
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=type("CustomSigner", (), {}))
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == type("CustomSigner", (), {})
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac", "digest_method": "sha256"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac", "digest_method": "sha256"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac", "digest_method": "sha256"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    serializer = Serializer("secret", serializer={"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x)})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x)}
    assert serializer.is_text_serializer is False
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #60
#--------------------------

```python
def test_serializer_not_none():
    serializer = Serializer(b"secret-key", serializer="custom_serializer")
    assert serializer.serializer == "custom_serializer"


