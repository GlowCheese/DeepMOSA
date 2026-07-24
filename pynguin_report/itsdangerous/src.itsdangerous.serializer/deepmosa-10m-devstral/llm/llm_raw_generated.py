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
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: x}
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
    serializer = Serializer("secret", signer_kwargs={"sep": ";"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": ";"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": ";"}, (type("CustomSigner", (), {}), {"key": "value"})]
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


# LLM-generated content at query #2
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": SHA256}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].digest_method == SHA512
    assert signers[1].digest_method == SHA256

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(Signer, {"digest_method": SHA256})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[0].digest_method == SHA512
    assert signers[1].digest_method == SHA256

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[Signer]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert all(isinstance(signer, Signer) for signer in signers)

def test_iter_unsigners_with_multiple_fallback_signers():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"digest_method": SHA256},
            (Signer, {"digest_method": SHA384}),
            Signer
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    assert signers[0].digest_method == SHA512
    assert signers[1].digest_method == SHA256
    assert signers[2].digest_method == SHA384
    assert signers[3].digest_method == SHA512

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

def test_iter_unsigners_with_key_rotation_and_fallback():
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": SHA256}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]
    assert signers[2].secret_keys == [b"new-key"]


# LLM-generated content at query #3
#--------------------------

```python
def test_serializer_constructor_with_string_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

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

def test_serializer_constructor_with_list_of_string_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_bytes_keys():
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


# LLM-generated content at query #4
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

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["key1", "key2", b"key3"])
    assert serializer.secret_keys == [b"key1", b"key2", b"key3"]
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

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key": "value"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_fallback_signers():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == [Signer]
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_kwargs():
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

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret-key", serializer=BytesSerializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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
    fallback = [{"sep": "|"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_secret_key_property():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_key == b"secret2"


# LLM-generated content at query #6
#--------------------------

```python
def test_load_payload_with_text_serializer():
    serializer = Serializer("secret", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return pickle.dumps(obj)

        def loads(self, payload):
            return pickle.loads(payload)

    serializer = Serializer("secret", serializer=BinarySerializer())
    payload = pickle.dumps({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj).encode()

        def loads(self, payload):
            return int(payload.decode())

    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = b"42"
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_raises_bad_payload_on_error():
    serializer = Serializer("secret", serializer=json)
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer(b"secret-key", fallback_signers=[{"digest_method": "sha256"}])
    assert serializer.fallback_signers is not None


# LLM-generated content at query #8
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    serializer = Serializer(secret_key="secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    serializer = Serializer(secret_key=b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(secret_key=["old_secret", "new_secret"])
    assert serializer.secret_keys == [b"old_secret", b"new_secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer(secret_key="secret", salt="custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    serializer = Serializer(secret_key="secret", salt=None)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt is None
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: x}
    serializer = Serializer(secret_key="secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer(secret_key="secret", serializer_kwargs={"indent": 2})
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
    serializer = Serializer(secret_key="secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer(secret_key="secret", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
    serializer = Serializer(secret_key="secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: x.encode(), "loads": lambda x: x.decode()}
    serializer = Serializer(secret_key="secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #9
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
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
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
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

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


# LLM-generated content at query #10
#--------------------------

```python
def test_loads_deserializes_payload():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #11
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: x}
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
    serializer = Serializer("secret", signer_kwargs={"sep": ":"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": ":"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": ":"}]
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
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj).encode("utf-8")

        @staticmethod
        def loads(data):
            return data.decode("utf-8")

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #12
#--------------------------

```python
def test_load_payload_raises_bad_payload_on_exception():
    serializer = Serializer(b"secret-key")
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid-payload", serializer={"loads": lambda x: 1/0})
    assert "Could not load the payload because an exception occurred on unserializing the data." in str(exc_info.value)


# LLM-generated content at query #13
#--------------------------

```python
def test_serializer_not_none():
    serializer = Serializer(b"secret-key", serializer="custom_serializer")
    assert serializer.serializer == "custom_serializer"


# LLM-generated content at query #14
#--------------------------

```python
def test_serializer_default_serializer():
    s = Serializer("secret")
    assert s.serializer == json
    assert s.is_text_serializer is True

def test_serializer_default_signer():
    s = Serializer("secret")
    assert s.signer == Signer

def test_serializer_default_fallback_signers():
    s = Serializer("secret")
    assert s.fallback_signers == []

def test_serializer_secret_keys_single():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]

def test_serializer_secret_keys_multiple():
    s = Serializer(["secret1", "secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]

def test_serializer_secret_key_property():
    s = Serializer("secret")
    assert s.secret_key == b"secret"

def test_serializer_salt_default():
    s = Serializer("secret")
    assert s.salt == b"itsdangerous"

def test_serializer_salt_custom():
    s = Serializer("secret", salt="custom")
    assert s.salt == b"custom"

def test_serializer_salt_none():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"sep": "|"})
    assert s.signer_kwargs == {"sep": "|"}

def test_serializer_custom_serializer_text():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return str(obj)

        @staticmethod
        def loads(s):
            return int(s)

    s = Serializer("secret", serializer=CustomSerializer())
    assert s.is_text_serializer is True

def test_serializer_custom_serializer_bytes():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return bytes(str(obj), "utf-8")

        @staticmethod
        def loads(s):
            return int(s.decode("utf-8"))

    s = Serializer("secret", serializer=CustomSerializer())
    assert s.is_text_serializer is False

def test_serializer_custom_signer():
    s = Serializer("secret", signer=TimedSerializer)
    assert s.signer == TimedSerializer

def test_serializer_custom_fallback_signers():
    s = Serializer("secret", fallback_signers=[{"key": "value"}])
    assert s.fallback_signers == [{"key": "value"}]


# LLM-generated content at query #15
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    fallback = {"digest_method": SHA256}
    serializer = Serializer("secret-key", fallback_signers=[fallback])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].digest_method.name == "sha256"

def test_iter_unsigners_with_fallback_signers_tuple():
    fallback = (Signer, {"digest_method": SHA256})
    serializer = Serializer("secret-key", fallback_signers=[fallback])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].digest_method.name == "sha256"

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_multiple_fallback_signers():
    fallback1 = {"digest_method": SHA256}
    fallback2 = (Signer, {"digest_method": SHA512})
    serializer = Serializer("secret-key", fallback_signers=[fallback1, fallback2])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert isinstance(signers[2], Signer)
    assert signers[1].digest_method.name == "sha256"
    assert signers[2].digest_method.name == "sha512"

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

def test_iter_unsigners_with_key_rotation_and_fallback():
    fallback = {"digest_method": SHA256}
    serializer = Serializer(["old-key", "new-key"], fallback_signers=[fallback])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]
    assert signers[2].secret_keys == [b"new-key"]
    assert signers[1].digest_method.name == "sha256"
    assert signers[2].digest_method.name == "sha256"


# LLM-generated content at query #16
#--------------------------

```python
def test_load_payload_with_text_serializer():
    serializer = Serializer("secret", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return pickle.dumps(obj)

        def loads(self, payload):
            return pickle.loads(payload)

    serializer = Serializer("secret", serializer=BinarySerializer())
    payload = pickle.dumps({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj).encode()

        def loads(self, payload):
            return int(payload.decode())

    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = b"42"
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret", serializer=json)
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #17
#--------------------------

```python
def test_salt_is_none_when_initialized_with_none():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #18
#--------------------------

```python
def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert result is not None


# LLM-generated content at query #19
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
            return payload.decode()

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = b"test"
    result = serializer.load_payload(payload)
    assert result == "test"

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret-key", serializer=json)
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #20
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
    serializer = Serializer("secret", serializer={"dumps": lambda x: str(x), "loads": lambda x: x})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: str(x), "loads": lambda x: x}
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
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

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
    serializer = Serializer("secret", serializer={"dumps": lambda x: x.encode(), "loads": lambda x: x.decode()})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: x.encode(), "loads": lambda x: x.decode()}
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #21
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key", serializer=json)
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert serializer.loads(result) == data

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"ensure_ascii": False})
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, bytes)
    assert serializer.loads(result) == data

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert serializer.loads(result, salt="custom-salt") == data

def test_dumps_with_different_salt_fails():
    serializer = Serializer("secret-key", salt="salt1")
    data = {"key": "value"}
    result = serializer.dumps(data)
    try:
        serializer.loads(result, salt="salt2")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

def test_dumps_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert serializer.loads(result, salt=None) == data

def test_dumps_with_list_secret_key():
    serializer = Serializer(["secret-key-1", "secret-key-2"])
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert serializer.loads(result) == data

def test_dumps_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"indent": 2})
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert serializer.loads(result) == data

def test_dumps_with_bytes_data():
    serializer = Serializer("secret-key")
    data = b"binary data"
    result = serializer.dumps(data)
    assert serializer.loads(result) == data

def test_dumps_with_unicode_data():
    serializer = Serializer("secret-key")
    data = "unicode data: \u2603"
    result = serializer.dumps(data)
    assert serializer.loads(result) == data


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

```python
def test_serializer_default_serializer():
    serializer = Serializer("secret")
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True

def test_serializer_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_single_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.secret_key == b"secret"

def test_serializer_multiple_keys():
    serializer = Serializer(["old_key", "new_key"])
    assert serializer.secret_keys == [b"old_key", b"new_key"]
    assert serializer.secret_key == b"new_key"

def test_serializer_salt():
    serializer = Serializer("secret", salt="my_salt")
    assert serializer.salt == b"my_salt"

def test_serializer_no_salt():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_custom_signer():
    custom_signer = Signer
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.signer is custom_signer

def test_serializer_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_fallback_signers():
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #24
#--------------------------

```python
def test_load_payload_raises_badpayload_on_exception():
    serializer = Serializer(b"secret", serializer=json)
    payload = b"invalid json"

    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload to be raised"
    except BadPayload as e:
        assert str(e) == "Could not load the payload because an exception occurred on unserializing the data."
        assert isinstance(e.original_error, json.JSONDecodeError)


# LLM-generated content at query #25
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
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
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
    from itsdangerous.signer import Signer
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
    from itsdangerous.signer import Signer
    fallback_signers = [Signer, {"sep": ":"}, (Signer, {"sep": ";"})]
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

def test_serializer_constructor_with_bytes_serializer():
    custom_serializer = {"dumps": lambda x: x, "loads": lambda x: x}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #26
#--------------------------

```python
def test_fallback_signers_none_check():
    serializer = Serializer(b"secret", fallback_signers=None)
    assert serializer.fallback_signers == []


# LLM-generated content at query #27
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


# LLM-generated content at query #28
#--------------------------

```python
def test_iter_unsigners_default_signer():
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

def test_iter_unsigners_with_fallback_signers():
    fallback_signer = {"digest_method": SHA256}
    serializer = Serializer("secret-key", fallback_signers=[fallback_signer])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"secret-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == [b"secret-key"]
    assert unsigners[1].salt == b"itsdangerous"
    assert unsigners[1].digest_method.name == "sha256"

def test_iter_unsigners_with_multiple_fallback_signers():
    fallback_signer1 = {"digest_method": SHA256}
    fallback_signer2 = {"digest_method": SHA512}
    serializer = Serializer("secret-key", fallback_signers=[fallback_signer1, fallback_signer2])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"secret-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == [b"secret-key"]
    assert unsigners[1].salt == b"itsdangerous"
    assert unsigners[1].digest_method.name == "sha256"
    assert isinstance(unsigners[2], Signer)
    assert unsigners[2].secret_keys == [b"secret-key"]
    assert unsigners[2].salt == b"itsdangerous"
    assert unsigners[2].digest_method.name == "sha512"

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"old-key", b"new-key"]
    assert unsigners[0].salt == b"itsdangerous"

def test_iter_unsigners_with_fallback_and_key_rotation():
    fallback_signer = {"digest_method": SHA256}
    serializer = Serializer(["old-key", "new-key"], fallback_signers=[fallback_signer])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == [b"old-key", b"new-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == [b"old-key"]
    assert unsigners[1].salt == b"itsdangerous"
    assert unsigners[1].digest_method.name == "sha256"


# LLM-generated content at query #29
#--------------------------

```python
def test_serializer_default_serializer():
    serializer = Serializer("secret")
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True

def test_serializer_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_default_signer():
    serializer = Serializer("secret")
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}

def test_serializer_custom_signer():
    custom_signer = Signer
    custom_kwargs = {"sep": "?"}
    serializer = Serializer("secret", signer=custom_signer, signer_kwargs=custom_kwargs)
    assert serializer.signer is custom_signer
    assert serializer.signer_kwargs == custom_kwargs

def test_serializer_default_fallback_signers():
    serializer = Serializer("secret")
    assert serializer.fallback_signers == Serializer.default_fallback_signers

def test_serializer_custom_fallback_signers():
    custom_fallback = [{"sep": "?"}]
    serializer = Serializer("secret", fallback_signers=custom_fallback)
    assert serializer.fallback_signers == custom_fallback

def test_serializer_secret_keys_single():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_secret_keys_multiple():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_salt_bytes():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_salt_str():
    serializer = Serializer("secret", salt="custom_salt")
    assert serializer.salt == b"custom_salt"

def test_serializer_salt_none():
    serializer = Serializer("secret", salt=None)
    assert serializer.salt is None

def test_serializer_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #30
#--------------------------

```python
def test_load_payload_raises_bad_payload_on_exception():
    serializer = Serializer(b"secret-key")
    payload = b"invalid-payload"

    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(payload)

    assert "Could not load the payload because an exception occurred on unserializing the data." in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, json.JSONDecodeError)


# LLM-generated content at query #31
#--------------------------

```python
def test_salt_is_none_when_not_provided():
    serializer = Serializer(b"secret-key")
    assert serializer.salt is None


# LLM-generated content at query #32
#--------------------------

```python
def test_dumps_with_bytes_serializer():
    serializer = Serializer(b"secret", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #33
#--------------------------

```python
def test_serializer_constructor_with_single_key():
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

def test_serializer_constructor_with_key_list():
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
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    import pickle
    serializer = Serializer("secret", serializer=pickle)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == pickle
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
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
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
    fallback = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_all_parameters():
    import pickle
    fallback = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
    serializer = Serializer(
        ["old", "new"],
        salt="custom",
        serializer=pickle,
        serializer_kwargs={"protocol": 2},
        signer=Signer,
        signer_kwargs={"sep": "|"},
        fallback_signers=fallback,
    )
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.salt == b"custom"
    assert serializer.serializer == pickle
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {"protocol": 2}


# LLM-generated content at query #34
#--------------------------

```python
def test_iter_unsigners_with_dict_fallback():
    serializer = Serializer("secret", fallback_signers=[{"key1": "value1"}])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)


# LLM-generated content at query #35
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

    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == CustomSerializer
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
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

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
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return bytes(str(obj), "utf-8")

        @staticmethod
        def loads(s):
            return int(s.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #36
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
    serializer = Serializer(["old_secret", "new_secret"])
    assert serializer.secret_keys == [b"old_secret", b"new_secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_list_of_bytes_keys():
    serializer = Serializer([b"old_secret", b"new_secret"])
    assert serializer.secret_keys == [b"old_secret", b"new_secret"]
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
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: x}
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
    custom_serializer = {"dumps": lambda x: bytes(x, "utf-8"), "loads": lambda x: x}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
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

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (), {})
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
    fallback_signers = [{"sep": "|"}, type("CustomSigner", (), {})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #37
#--------------------------

```python
def test_loads_deserializes_payload_correctly():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #38
#--------------------------

```python
def test_dumps_returns_bytes_when_not_text_serializer():
    serializer = Serializer(b"secret", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #39
#--------------------------

```python
def test_salt_is_none_when_not_provided():
    serializer = Serializer(b"secret-key")
    assert serializer.salt is None


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
    serializer = Serializer("secret", serializer={"dumps": lambda x: "custom", "loads": lambda x: "custom"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: "custom", "loads": lambda x: "custom"}
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
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac", "digest_method": "SHA256"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac", "digest_method": "SHA256"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac", "digest_method": "SHA256"}]
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
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

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
    serializer = Serializer("secret", serializer={"dumps": lambda x: b"custom", "loads": lambda x: "custom"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: b"custom", "loads": lambda x: "custom"}
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #2
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


# LLM-generated content at query #3
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
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
    fallback_signers = [{"key_derivation": "hmac"}, (Signer, {"key_derivation": "hmac"}), Signer]
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
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #4
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

def test_serializer_constructor_custom_serializer():
    custom_serializer = {"dumps": lambda x: x, "loads": lambda x: x}
    serializer = Serializer(b"secret", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_constructor_custom_signer():
    serializer = Serializer(b"secret", signer=TimedSigner)
    assert serializer.signer is TimedSigner

def test_serializer_constructor_key_rotation():
    serializer = Serializer([b"old", b"new"])
    assert serializer.secret_keys == [b"old", b"new"]
    assert serializer.secret_key == b"new"

def test_serializer_constructor_salt_none():
    serializer = Serializer(b"secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_fallback_signers():
    fallback = [{"key_derivation": "hmac", "digest_method": "sha256"}]
    serializer = Serializer(b"secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_serializer_kwargs():
    serializer = Serializer(b"secret", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_signer_kwargs():
    serializer = Serializer(b"secret", signer_kwargs={"sep": "|"})
    assert serializer.signer_kwargs == {"sep": "|"}


# LLM-generated content at query #5
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
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
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


# LLM-generated content at query #6
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

def test_serializer_constructor_with_iterable_secret_key():
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


# LLM-generated content at query #7
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret"]
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_with_salt():
    serializer = Serializer("secret", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer("secret", fallback_signers=[{"digest_method": SHA256}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].digest_method.name == "sha256"

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer("secret", fallback_signers=[(Signer, {"digest_method": SHA256})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].digest_method.name == "sha256"

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer("secret", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_multiple_fallback_signers():
    serializer = Serializer("secret", fallback_signers=[
        {"digest_method": SHA256},
        (Signer, {"digest_method": SHA512}),
        Signer
    ])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    assert all(isinstance(signer, Signer) for signer in signers)

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-secret", "new-secret"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-secret", b"new-secret"]

def test_iter_unsigners_with_key_rotation_and_fallback():
    serializer = Serializer(["old-secret", "new-secret"], fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == [b"old-secret", b"new-secret"]
    assert signers[1].secret_keys == [b"old-secret"]
    assert signers[2].secret_keys == [b"new-secret"]


# LLM-generated content at query #8
#--------------------------

```python
def test_load_payload_with_text_serializer():
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return obj.encode('utf-8')

        def loads(self, payload):
            return payload.decode('utf-8')

    serializer = Serializer("secret-key", serializer=BinarySerializer())
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == '{"key": "value"}'

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj).encode('utf-8')

        def loads(self, payload):
            return int(payload.decode('utf-8'))

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = b'42'
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_bad_payload():
    serializer = Serializer("secret-key")
    payload = b'invalid json'
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #9
#--------------------------

```python
def test_isinstance_fallback_dict():
    serializer = Serializer(b"secret", fallback_signers=[{"key": "value"}])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)


# LLM-generated content at query #10
#--------------------------

```python
def test_loads_deserializes_payload():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #11
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, is_text_serializer=False)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert b"key" in result

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_different_secret_key():
    serializer = Serializer("different-secret")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_list_secret_key():
    serializer = Serializer(["key1", "key2"])
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result


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
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
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
    fallback_signers = [Signer]
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

def test_serializer_constructor_with_bytes_serializer():
    serializer = Serializer("secret", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #13
#--------------------------

```python
def test_dumps_serializes_object():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)  # Assuming _TSerialized is str for this test


# LLM-generated content at query #14
#--------------------------

```python
def test_is_text_serializer_false():
    serializer = Serializer(b"secret", serializer=bytes)
    assert not serializer.is_text_serializer


# LLM-generated content at query #15
#--------------------------

```python
def test_dumps_with_text_serializer():
    serializer = Serializer("secret-key", serializer=json)
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert result.startswith('{"key": "value"}')

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, salt=b"salt")
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, bytes)
    assert result.startswith(b'{"key": "value"}')

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key")
    obj = {"key": "value"}
    result = serializer.dumps(obj, salt="custom-salt")
    assert isinstance(result, str)
    assert result.startswith('{"key": "value"}')

def test_dumps_with_different_object():
    serializer = Serializer("secret-key")
    obj = ["a", "b", "c"]
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert result.startswith('["a", "b", "c"]')

def test_dumps_empty_object():
    serializer = Serializer("secret-key")
    obj = {}
    result = serializer.dumps(obj)
    assert isinstance(result, str)
    assert result.startswith('{}')


# LLM-generated content at query #16
#--------------------------

```python
def test_load_payload_with_non_text_serializer():
    serializer = Serializer(b"secret", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #17
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer(b"secret-key", fallback_signers=[{}])
    assert serializer.fallback_signers is not None


# LLM-generated content at query #18
#--------------------------

```python
def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)  # Assuming _TSerialized is str for this test


# LLM-generated content at query #19
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
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: int(x)}
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
    fallback_signers = [{"sep": "|"}, (type("CustomSigner", (), {}), {"sep": "|"})]
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
    bytes_serializer = {"dumps": lambda x: bytes(x, "utf-8"), "loads": lambda x: x.decode("utf-8")}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #20
#--------------------------

```python
def test_load_payload_with_text_serializer():
    serializer = Serializer("secret", serializer=json)
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return pickle.dumps(obj)

        def loads(self, payload):
            return pickle.loads(payload)

    serializer = Serializer("secret", serializer=BinarySerializer())
    payload = pickle.dumps({"key": "value"})
    assert serializer.load_payload(payload) == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return str(obj).encode()

        def loads(self, payload):
            return int(payload.decode())

    serializer = Serializer("secret", serializer=CustomSerializer())
    payload = b"42"
    assert serializer.load_payload(payload) == 42

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret", serializer=json)
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_key == b"secret-key"
    assert unsigners[0].salt == b"itsdangerous"

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert unsigners[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    fallback_signers = [{"digest_method": SHA256}]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].digest_method == SHA256

def test_iter_unsigners_with_fallback_signers_tuple():
    fallback_signers = [(CustomSigner, {"digest_method": SHA256})]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], CustomSigner)
    assert unsigners[1].digest_method == SHA256

def test_iter_unsigners_with_fallback_signers_class():
    fallback_signers = [CustomSigner]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], CustomSigner)

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert unsigners[0].secret_key == b"new-key"

def test_iter_unsigners_with_key_rotation_and_fallback():
    fallback_signers = [{"digest_method": SHA256}]
    serializer = Serializer(["old-key", "new-key"], fallback_signers=fallback_signers)
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3
    assert unsigners[0].secret_key == b"new-key"
    assert unsigners[1].secret_key == b"old-key"
    assert unsigners[2].secret_key == b"new-key"
    assert unsigners[1].digest_method == SHA256
    assert unsigners[2].digest_method == SHA256


# LLM-generated content at query #22
#--------------------------

```python
def test_serializer_defaults():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer is json
    assert s.is_text_serializer is True
    assert s.signer is Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_with_bytes_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"

def test_serializer_with_key_list():
    s = Serializer(["old", "new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_with_bytes_key_list():
    s = Serializer([b"old", b"new"])
    assert s.secret_keys == [b"old", b"new"]
    assert s.secret_key == b"new"

def test_serializer_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.salt is None

def test_serializer_with_custom_salt():
    s = Serializer("secret", salt="custom")
    assert s.salt == b"custom"

def test_serializer_with_custom_serializer():
    serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x)}
    s = Serializer("secret", serializer=serializer)
    assert s.serializer is serializer
    assert s.is_text_serializer is False

def test_serializer_with_text_serializer():
    serializer = {"dumps": lambda x: str(x), "loads": lambda x: int(x)}
    s = Serializer("secret", serializer=serializer)
    assert s.serializer is serializer
    assert s.is_text_serializer is True

def test_serializer_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_with_custom_signer():
    s = Serializer("secret", signer=TimedSerializer)
    assert s.signer is TimedSerializer

def test_serializer_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"sep": "|"})
    assert s.signer_kwargs == {"sep": "|"}

def test_serializer_with_fallback_signers():
    fallback = [{"digest_method": SHA256}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.fallback_signers == fallback


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
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
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
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
    fallback = [{"sep": "|"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #2
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
    custom_serializer = json
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
    custom_signer = Signer
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
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"})]
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


# LLM-generated content at query #3
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
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
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
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
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


# LLM-generated content at query #4
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_with_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_dict():
    serializer = Serializer("secret-key", fallback_signers=[{"digest_method": SHA256}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].digest_method == SHA256

def test_iter_unsigners_with_fallback_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"digest_method": SHA256})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].digest_method == SHA256

def test_iter_unsigners_with_fallback_signer_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_multiple_fallbacks():
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
    assert signers[1].digest_method == SHA256
    assert signers[2].digest_method == SHA512
    assert isinstance(signers[3], Signer)

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

def test_iter_unsigners_with_key_rotation_and_fallback():
    serializer = Serializer(["old-key", "new-key"], fallback_signers=[{"digest_method": SHA256}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]
    assert signers[2].secret_keys == [b"new-key"]


# LLM-generated content at query #5
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


# LLM-generated content at query #6
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer(b"secret-key", fallback_signers=[{"key": "value"}])
    assert serializer.fallback_signers is not None


# LLM-generated content at query #7
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"ensure_ascii": False})
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_different_secret_key():
    serializer = Serializer("different-secret", salt="salt")
    result = serializer.dumps({"data": 123})
    assert isinstance(result, str)
    assert "data" in result

def test_dumps_with_list_secret_key():
    serializer = Serializer(["key1", "key2"], salt="salt")
    result = serializer.dumps({"list": "keys"})
    assert isinstance(result, str)
    assert "list" in result

def test_dumps_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    result = serializer.dumps({"salt": None})
    assert isinstance(result, str)
    assert "salt" in result


# LLM-generated content at query #8
#--------------------------

```python
def test_serializer_not_none():
    serializer = Serializer(secret_key="test", serializer="custom_serializer")
    assert serializer.serializer == "custom_serializer"


# LLM-generated content at query #9
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

def test_serializer_constructor_with_list_of_keys():
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
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: int(x)}
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
    custom_serializer = {"dumps": lambda x: bytes(x, "utf-8"), "loads": lambda x: int(x)}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is custom_serializer
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

def test_serializer_secret_key_property():
    serializer = Serializer(["old", "new"])
    assert serializer.secret_key == b"new"


# LLM-generated content at query #10
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
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: x}
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
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
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

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: bytes(str(x), "utf-8"), "loads": lambda x: x.decode("utf-8")}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #11
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
            return f"custom_{payload.decode('utf-8')}"

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = b"test_data"
    result = serializer.load_payload(payload)
    assert result == "custom_test_data"

def test_load_payload_with_text_serializer():
    serializer = Serializer("secret-key", serializer=json)
    payload = b'{"text": "data"}'
    result = serializer.load_payload(payload)
    assert result == {"text": "data"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def loads(self, payload):
            return f"binary_{payload}"

    serializer = Serializer("secret-key", serializer=BinarySerializer())
    payload = b"binary_data"
    result = serializer.load_payload(payload)
    assert result == "binary_binary_data"

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret-key")
    payload = b"invalid_json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_loads_deserializes_payload_correctly():
    class MockSerializer(_PDataSerializer[str]):
        def loads(self, payload: str, /) -> t.Any:
            return payload.upper()

        def dumps(self, obj: t.Any, /) -> str:
            return str(obj).lower()

    serializer = MockSerializer()
    result = serializer.loads("hello")
    assert result == "HELLO"


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
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
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

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #2
#--------------------------

```python
def test_dumps_serializes_object():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)  # Assuming _TSerialized is str for this test


# LLM-generated content at query #3
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
    s = Serializer("secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_secret_key():
    s = Serializer(b"secret")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_list_secret_key():
    s = Serializer(["secret1", "secret2"])
    assert s.secret_keys == [b"secret1", b"secret2"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    s = Serializer("secret", salt="custom-salt")
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"custom-salt"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    serializer = {"dumps": lambda x: str(x), "loads": lambda x: x}
    s = Serializer("secret", serializer=serializer)
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == serializer
    assert s.is_text_serializer is True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    s = Serializer("secret", serializer_kwargs={"indent": 2})
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    s = Serializer("secret", signer=Signer)
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer == Signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    s = Serializer("secret", signer_kwargs={"sep": "|"})
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {"sep": "|"}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"sep": "|"}]
    s = Serializer("secret", fallback_signers=fallback)
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == fallback
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_none_salt():
    s = Serializer("secret", salt=None)
    assert s.secret_keys == [b"secret"]
    assert s.salt is None
    assert s.serializer == Serializer.default_serializer
    assert s.is_text_serializer is True
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    serializer = {"dumps": lambda x: bytes(x, "utf-8"), "loads": lambda x: x}
    s = Serializer("secret", serializer=serializer)
    assert s.secret_keys == [b"secret"]
    assert s.salt == b"itsdangerous"
    assert s.serializer == serializer
    assert s.is_text_serializer is False
    assert s.signer == Serializer.default_signer
    assert s.signer_kwargs == {}
    assert s.fallback_signers == []
    assert s.serializer_kwargs == {}


# LLM-generated content at query #4
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, salt="test-salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    result = serializer.dumps({"data": "test"})
    assert isinstance(result, str)
    assert "data" in result

def test_dumps_with_list_data():
    serializer = Serializer("secret-key")
    result = serializer.dumps([1, 2, 3])
    assert isinstance(result, str)
    assert "1" in result

def test_dumps_with_none_data():
    serializer = Serializer("secret-key")
    result = serializer.dumps(None)
    assert isinstance(result, str)
    assert "null" in result

def test_dumps_with_empty_dict():
    serializer = Serializer("secret-key")
    result = serializer.dumps({})
    assert isinstance(result, str)
    assert result == 'e30.{}'

def test_dumps_with_nested_data():
    serializer = Serializer("secret-key")
    data = {"outer": {"inner": "value"}}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert "outer" in result
    assert "inner" in result

def test_dumps_with_bytes_key():
    serializer = Serializer(b"secret-key")
    result = serializer.dumps({"test": "data"})
    assert isinstance(result, str)
    assert "test" in result

def test_dumps_with_different_serializer():
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return f"custom-{obj}"

        @staticmethod
        def loads(s):
            return s.replace("custom-", "")

    serializer = Serializer("secret-key", serializer=CustomSerializer)
    result = serializer.dumps("test-data")
    assert isinstance(result, str)
    assert "custom-test-data" in result


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

def test_serializer_constructor_with_iterable_secret_key():
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
    serializer = Serializer("secret", serializer=b"custom_serializer")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == b"custom_serializer"
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
    serializer = Serializer("secret", signer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key1": "value1"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key1": "value1"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"key1": "value1"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key1": "value1"}

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


# LLM-generated content at query #6
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"ensure_ascii": False})
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert b"key" in result

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_different_secret_key():
    serializer = Serializer("different-secret", salt="test-salt")
    result = serializer.dumps({"test": "data"})
    assert isinstance(result, str)
    assert "test" in result

def test_dumps_with_list_secret_key():
    serializer = Serializer(["key1", "key2"], salt="list-salt")
    result = serializer.dumps({"list": "test"})
    assert isinstance(result, str)
    assert "list" in result


# LLM-generated content at query #7
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key", serializer=json)
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert "key" in result
    assert "value" in result

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, serializer_kwargs={"ensure_ascii": False})
    data = {"key": "value"}
    result = serializer.dumps(data)
    assert isinstance(result, bytes)
    assert b"key" in result
    assert b"value" in result

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    data = {"key": "value"}
    result = serializer.dumps(data, salt="another-salt")
    assert isinstance(result, str)
    assert "key" in result
    assert "value" in result

def test_dumps_with_empty_data():
    serializer = Serializer("secret-key")
    result = serializer.dumps({})
    assert isinstance(result, str)
    assert result != ""

def test_dumps_with_none_data():
    serializer = Serializer("secret-key")
    result = serializer.dumps(None)
    assert isinstance(result, str)
    assert result != ""

def test_dumps_with_list_data():
    serializer = Serializer("secret-key")
    data = [1, 2, 3]
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert "1" in result
    assert "2" in result
    assert "3" in result

def test_dumps_with_nested_data():
    serializer = Serializer("secret-key")
    data = {"outer": {"inner": "value"}}
    result = serializer.dumps(data)
    assert isinstance(result, str)
    assert "outer" in result
    assert "inner" in result
    assert "value" in result


# LLM-generated content at query #8
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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
    class CustomSerializer:
        @staticmethod
        def dumps(obj):
            return f"custom-{obj}"

        @staticmethod
        def loads(s):
            return s.replace("custom-", "")

    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is CustomSerializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
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


# LLM-generated content at query #9
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
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    from itsdangerous.signer import Signer
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
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac", "digest_method": "SHA256"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac", "digest_method": "SHA256"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    from itsdangerous.signer import Signer
    fallback_signers = [Signer, {"key_derivation": "hmac"}, (Signer, {"digest_method": "SHA256"})]
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

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #10
#--------------------------

```python
def test_salt_none_predicate():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #11
#--------------------------

```python
def test_loads_with_valid_payload():
    serializer = _PDataSerializer()
    payload = b'some_serialized_data'
    result = serializer.loads(payload)
    assert isinstance(result, object)

def test_loads_with_empty_payload():
    serializer = _PDataSerializer()
    payload = b''
    result = serializer.loads(payload)
    assert isinstance(result, object)

def test_loads_with_none_payload():
    serializer = _PDataSerializer()
    payload = None
    result = serializer.loads(payload)
    assert isinstance(result, object)


# LLM-generated content at query #12
#--------------------------

```python
def test_serializer_default_serializer_assignment():
    serializer_instance = Serializer(b"secret-key")
    assert serializer_instance.serializer is Serializer.default_serializer


# LLM-generated content at query #13
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer("secret-key", fallback_signers=[{"key_derivation": "hmac"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].signer_kwargs == {"key_derivation": "hmac"}

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"key_derivation": "hmac"})])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)
    assert signers[1].signer_kwargs == {"key_derivation": "hmac"}

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_multiple_fallback_signers():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"key_derivation": "hmac"},
            (Signer, {"key_derivation": "concat"}),
            Signer
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    assert all(isinstance(signer, Signer) for signer in signers)

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

def test_iter_unsigners_with_key_rotation_and_fallback():
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"key_derivation": "hmac"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3  # 1 default + 2 from fallback (one per key)
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]
    assert signers[2].secret_keys == [b"new-key"]


# LLM-generated content at query #14
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
    serializer = Serializer("secret", serializer={"dumps": lambda x: "custom", "loads": lambda x: {}})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: "custom", "loads": lambda x: {}}
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
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac", "digest_method": "SHA256"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac", "digest_method": "SHA256"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac", "digest_method": "SHA256"}]
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


# LLM-generated content at query #15
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
    fallback_signers = [Signer]
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


# LLM-generated content at query #16
#--------------------------

```python
def test_load_payload_with_default_serializer():
    serializer = Serializer("secret-key")
    payload = b'{"key": "value"}'
    assert serializer.load_payload(payload) == {"key": "value"}

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def loads(self, payload):
            return f"custom_{payload.decode('utf-8')}"

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = b"test_data"
    assert serializer.load_payload(payload) == "custom_test_data"

def test_load_payload_with_text_serializer():
    serializer = Serializer("secret-key")
    payload = b'{"text": "data"}'
    assert serializer.load_payload(payload) == {"text": "data"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def loads(self, payload):
            return f"binary_{payload}"

    serializer = Serializer("secret-key", serializer=BinarySerializer())
    payload = b"\x00\x01\x02"
    assert serializer.load_payload(payload) == "binary_b'\\x00\\x01\\x02'"

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret-key")
    payload = b"invalid_json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #17
#--------------------------

```python
def test_serializer_not_none():
    serializer = Serializer(b"secret-key", serializer="custom_serializer")
    assert serializer.serializer == "custom_serializer"


# LLM-generated content at query #18
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback():
    serializer = Serializer("secret-key")
    fallback_signer = Signer
    fallback_kwargs = {"sep": "?"}
    serializer.fallback_signers = [(fallback_signer, fallback_kwargs)]
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[1], fallback_signer)
    assert unsigners[1].sep == "?"


# LLM-generated content at query #19
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
    fallback = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
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


# LLM-generated content at query #20
#--------------------------

```python
def test_iter_unsigners_default_signer():
    serializer = Serializer("secret-key")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert isinstance(signers[0], Signer)
    assert signers[0].secret_keys == [b"secret-key"]
    assert signers[0].salt == b"itsdangerous"

def test_iter_unsigners_with_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].signer_kwargs["digest_method"] == "sha256"

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[(Signer, {"digest_method": "sha256"})]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert signers[1].signer_kwargs["digest_method"] == "sha256"

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[Signer]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[1], Signer)

def test_iter_unsigners_with_multiple_fallback_signers():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"digest_method": "sha256"},
            (Signer, {"digest_method": "sha512"}),
            Signer
        ]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 4
    assert signers[1].signer_kwargs["digest_method"] == "sha256"
    assert signers[2].signer_kwargs["digest_method"] == "sha512"
    assert isinstance(signers[3], Signer)

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 1
    assert signers[0].secret_keys == [b"old-key", b"new-key"]

def test_iter_unsigners_with_fallback_and_key_rotation():
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": "sha256"}]
    )
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 3
    assert signers[0].secret_keys == [b"old-key", b"new-key"]
    assert signers[1].secret_keys == [b"old-key"]
    assert signers[2].secret_keys == [b"new-key"]


# LLM-generated content at query #21
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

def test_serializer_constructor_with_iterable_secret_key():
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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

def test_serializer_constructor_with_bytes_serializer():
    serializer = Serializer("secret", serializer={"dumps": lambda x: b"custom", "loads": lambda x: {}})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: b"custom", "loads": lambda x: {}}
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=type("CustomSigner", (), {}))
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == type("CustomSigner", (), {})
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac", "digest_method": "sha256"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac", "digest_method": "sha256"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key_derivation": "hmac", "digest_method": "sha256"}]
    serializer = Serializer("secret", fallback_signers=fallback)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"sort_keys": True, "indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"sort_keys": True, "indent": 4}

def test_serializer_constructor_with_all_parameters():
    fallback = [{"key_derivation": "hmac", "digest_method": "sha256"}]
    serializer = Serializer(
        ["secret1", "secret2"],
        salt="custom-salt",
        serializer={"dumps": lambda x: "custom", "loads": lambda x: {}},
        serializer_kwargs={"sort_keys": True},
        signer=type("CustomSigner", (), {}),
        signer_kwargs={"key_derivation": "hmac"},
        fallback_signers=fallback,
    )
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == {"dumps": lambda x: "custom", "loads": lambda x: {}}
    assert serializer.is_text_serializer is True
    assert serializer.signer == type("CustomSigner", (), {})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == fallback
    assert serializer.serializer_kwargs == {"sort_keys": True}


# LLM-generated content at query #22
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

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret-key-1", "secret-key-2"])
    assert serializer.secret_keys == [b"secret-key-1", b"secret-key-2"]
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

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

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


# LLM-generated content at query #23
#--------------------------

```python
def test_loads_deserializes_payload():
    serializer = _PDataSerializer()
    payload = b'some_serialized_data'
    result = serializer.loads(payload)
    assert result is not None


# LLM-generated content at query #24
#--------------------------

```python
def test_dumps_with_bytes_serializer():
    serializer = Serializer(b"secret", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #25
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
            return f"custom_{payload.decode('utf-8')}"

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = b"test_data"
    result = serializer.load_payload(payload)
    assert result == "custom_test_data"

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret-key")
    payload = b"invalid_json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)

def test_load_payload_with_text_serializer():
    serializer = Serializer("secret-key")
    payload = b'{"text": "data"}'
    result = serializer.load_payload(payload)
    assert result == {"text": "data"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def loads(self, payload):
            return f"binary_{payload}"

    serializer = Serializer("secret-key", serializer=BinarySerializer())
    payload = b"binary_data"
    result = serializer.load_payload(payload)
    assert result == "binary_binary_data"


# LLM-generated content at query #26
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

def test_serializer_constructor_with_list_secret_keys():
    serializer = Serializer(["old_secret", "new_secret"])
    assert serializer.secret_keys == [b"old_secret", b"new_secret"]
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
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: int(x)}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    from itsdangerous.signer import Signer
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
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
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
    bytes_serializer = {"dumps": lambda x: x, "loads": lambda x: x}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #27
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

def test_serializer_constructor_with_list_secret_keys():
    serializer = Serializer(["key1", "key2", "key3"])
    assert serializer.secret_keys == [b"key1", b"key2", b"key3"]
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
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
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
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (), {})
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
    serializer = Serializer("secret-key", signer_kwargs={"sep": ";"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": ";"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": ";"}, (type("CustomSigner", (), {}), {"sep": ":"})]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #28
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback():
    serializer = Serializer(b"secret", fallback_signers=[(Signer, {"key_derivation": "hmac"})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)


# LLM-generated content at query #29
#--------------------------

```python
def test_loads_deserializes_payload_correctly():
    class MockSerializer(_PDataSerializer[str]):
        def loads(self, payload: str, /) -> t.Any:
            return f"deserialized_{payload}"

        def dumps(self, obj: t.Any, /) -> str:
            return str(obj)

    serializer = MockSerializer()
    result = serializer.loads("test_payload")
    assert result == "deserialized_test_payload"


# LLM-generated content at query #30
#--------------------------

```python
def test_serializer_default_serializer():
    serializer = Serializer(b"secret-key")
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x), "loads": lambda x: int(x)}
    serializer = Serializer(b"secret-key", serializer=custom_serializer)
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_custom_salt():
    serializer = Serializer(b"secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_custom_signer():
    custom_signer = type("CustomSigner", (), {"__init__": lambda self, *args, **kwargs: None})
    serializer = Serializer(b"secret-key", signer=custom_signer)
    assert serializer.signer == custom_signer

def test_serializer_custom_signer_kwargs():
    serializer = Serializer(b"secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}

def test_serializer_custom_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
    serializer = Serializer(b"secret-key", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_custom_serializer_kwargs():
    serializer = Serializer(b"secret-key", serializer_kwargs={"indent": 2})
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    assert serializer.secret_keys == [b"old-key", b"new-key"]
    assert serializer.secret_key == b"new-key"

def test_serializer_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: bytes(x, "utf-8"), "loads": lambda x: x.decode("utf-8")}
    serializer = Serializer(b"secret-key", serializer=bytes_serializer)
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False

def test_serializer_none_salt():
    serializer = Serializer(b"secret-key", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #31
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
            return f"custom_{payload.decode('utf-8')}"

        def dumps(self, obj):
            return f"{obj}".encode('utf-8')

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = b"test_data"
    result = serializer.load_payload(payload)
    assert result == "custom_test_data"

def test_load_payload_with_text_serializer():
    serializer = Serializer("secret-key", serializer=json)
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def loads(self, payload):
            return f"binary_{payload}"

        def dumps(self, obj):
            return f"{obj}".encode('utf-8')

    serializer = Serializer("secret-key", serializer=BinarySerializer())
    payload = b"test_data"
    result = serializer.load_payload(payload)
    assert result == "binary_b'test_data'"

def test_load_payload_with_invalid_payload():
    serializer = Serializer("secret-key")
    payload = b"invalid_json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #32
#--------------------------

```python
def test_load_payload_raises_bad_payload_on_exception():
    serializer = Serializer(b"secret")
    payload = b"invalid_payload"
    try:
        serializer.load_payload(payload)
    except BadPayload as e:
        assert str(e) == "Could not load the payload because an exception occurred on unserializing the data."
        assert isinstance(e.original_error, Exception)
    else:
        assert False, "Expected BadPayload to be raised"


# LLM-generated content at query #33
#--------------------------

```python
def test_dumps_serializes_object():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)


# LLM-generated content at query #34
#--------------------------

```python
def test_salt_is_none_when_not_provided():
    serializer = Serializer(secret_key=b"secret")
    assert serializer.salt is None


# LLM-generated content at query #35
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

def test_serializer_constructor_with_iterable_secret_keys():
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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


# LLM-generated content at query #36
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

def test_serializer_constructor_with_iterable_secret_key():
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
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac", "digest_method": "sha256"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac", "digest_method": "sha256"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac", "digest_method": "sha256"}]
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
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}


# LLM-generated content at query #37
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

def test_serializer_with_bytes_key():
    serializer = Serializer(b"secret-key")
    assert serializer.secret_keys == [b"secret-key"]

def test_serializer_with_iterable_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]

def test_serializer_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_with_none_salt():
    serializer = Serializer("secret-key", salt=None)
    assert serializer.salt is None

def test_serializer_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is False

def test_serializer_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret-key", signer=CustomSigner)
    assert serializer.signer is CustomSigner

def test_serializer_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key1": "value1"})
    assert serializer.signer_kwargs == {"key1": "value1"}

def test_serializer_with_fallback_signers():
    fallback_signers = [{"key1": "value1"}, (Signer, {"key2": "value2"})]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.fallback_signers == fallback_signers

def test_serializer_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"key1": "value1"})
    assert serializer.serializer_kwargs == {"key1": "value1"}

def test_serializer_secret_key_property():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


# LLM-generated content at query #38
#--------------------------

```python
def test_dumps_serializes_object():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, str)  # Assuming _TSerialized is str for this test


# LLM-generated content at query #39
#--------------------------

```python
def test_serializer_initialization_with_none_serializer():
    serializer = Serializer(secret_key="test", serializer=None)
    assert serializer.serializer == Serializer.default_serializer


# LLM-generated content at query #40
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
    serializer = Serializer(["secret-key-1", "secret-key-2"])
    assert serializer.secret_keys == [b"secret-key-1", b"secret-key-2"]
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
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}, (Signer, {"key_derivation": "hmac"})]
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
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer()
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
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
    serializer = Serializer("secret", signer_kwargs={"sep": "custom"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": "custom"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "custom"}]
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


# LLM-generated content at query #42
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    class CustomSigner(Signer):
        pass
    serializer = Serializer("secret", signer=CustomSigner)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == CustomSigner
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac", "digest_method": "SHA256"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac", "digest_method": "SHA256"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac", "digest_method": "SHA256"}]
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
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #43
#--------------------------

```python
def test_loads_deserializes_payload():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #44
#--------------------------

```python
def test_load_payload_with_binary_serializer():
    serializer = Serializer(b"secret", serializer=json)
    payload = b"invalid_json"
    try:
        serializer.load_payload(payload)
    except BadPayload:
        pass
    assert not serializer.is_text_serializer


# LLM-generated content at query #45
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

def test_serializer_constructor_with_iterable_keys():
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
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
    serializer = Serializer("secret", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

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


# LLM-generated content at query #46
#--------------------------

```python
def test_load_payload_with_non_text_serializer_raises_bad_payload():
    serializer = Serializer(b"secret-key")
    payload = b"invalid-payload"
    assert_raises(BadPayload, serializer.load_payload, payload)


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

def test_serializer_constructor_with_iterable_secret_keys():
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


# LLM-generated content at query #48
#--------------------------

```python
def test_load_payload_with_text_serializer():
    serializer = Serializer("secret-key")
    payload = serializer.dump_payload({"key": "value"})
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def dumps(self, obj):
            return str(obj).encode()

        def loads(self, payload):
            return int(payload.decode())

    serializer = Serializer("secret-key", serializer=BinarySerializer())
    payload = serializer.dump_payload(42)
    result = serializer.load_payload(payload)
    assert result == 42

def test_load_payload_with_custom_serializer():
    class CustomSerializer:
        def dumps(self, obj):
            return f"custom:{obj}".encode()

        def loads(self, payload):
            return payload.decode().split(":", 1)[1]

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = serializer.dump_payload("test")
    result = serializer.load_payload(payload)
    assert result == "test"

def test_load_payload_with_bad_payload():
    serializer = Serializer("secret-key")
    try:
        serializer.load_payload(b"invalid-payload")
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #49
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
            return s

    serializer = Serializer("secret", serializer=CustomSerializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == CustomSerializer
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

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return bytes(str(obj), "utf-8")

        @staticmethod
        def loads(s):
            return s.decode("utf-8")

    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #50
#--------------------------

```python
def test_serializer_constructor_with_string_secret_key():
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

def test_serializer_constructor_with_iterable_secret_key():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
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

def test_serializer_constructor_with_custom_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {"key": "value"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_fallback_signers():
    fallback_signers = [{"key": "value"}]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is json
    assert serializer.is_text_serializer is True
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key": "value"}

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

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {}}
    serializer = Serializer("secret-key", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer is Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #51
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

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
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
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac", "digest_method": "SHA256"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac", "digest_method": "SHA256"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac", "digest_method": "SHA256"}]
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
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

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


# LLM-generated content at query #52
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
    serializer = Serializer("secret", signer_kwargs={"sep": "--"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": "--"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "--"}, (Signer, {"sep": "++"})]
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


# LLM-generated content at query #53
#--------------------------

```python
def test_serializer_constructor_with_single_secret_key():
    serializer = Serializer("secret-key")
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_multiple_secret_keys():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_keys == [b"key1", b"key2"]
    assert serializer.secret_key == b"key2"

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
            return b"bytes"

        @staticmethod
        def loads(s):
            return {}

    serializer = Serializer("secret-key", serializer=BytesSerializer())
    assert serializer.is_text_serializer is False

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret-key", signer=Signer)
    assert serializer.signer == Signer

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key": "value"})
    assert serializer.signer_kwargs == {"key": "value"}

def test_serializer_constructor_with_fallback_signers():
    fallback = [{"key": "value"}]
    serializer = Serializer("secret-key", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"key": "value"})
    assert serializer.serializer_kwargs == {"key": "value"}


# LLM-generated content at query #54
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
    serializer = Serializer("secret", serializer={"dumps": lambda x: "test", "loads": lambda x: x})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == {"dumps": lambda x: "test", "loads": lambda x: x}
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
    serializer = Serializer("secret", signer_kwargs={"key_derivation": "hmac", "digest_method": "SHA256"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac", "digest_method": "SHA256"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac", "digest_method": "SHA256"}]
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
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

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
            return b"test"

        @staticmethod
        def loads(data):
            return data

    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #55
#--------------------------

```python
def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, _TSerialized)


# LLM-generated content at query #56
#--------------------------

```python
def test_salt_is_none():
    serializer = Serializer(secret_key="test")
    assert serializer.salt is None


# LLM-generated content at query #57
#--------------------------

```python
def test_load_payload_with_invalid_payload():
    serializer = Serializer(b"secret-key")
    payload = b"invalid-payload"
    try:
        serializer.load_payload(payload)
    except BadPayload as e:
        assert e.original_error is not None


# LLM-generated content at query #58
#--------------------------

```python
def test_salt_none_predicate():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #59
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

def test_serializer_constructor_with_list_secret_key():
    serializer = Serializer(["secret-key-1", "secret-key-2"])
    assert serializer.secret_keys == [b"secret-key-1", b"secret-key-2"]
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

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret-key", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret-key", signer=custom_signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"key_derivation": "hmac"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"key_derivation": "hmac"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key_derivation": "hmac"}]
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
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}


# LLM-generated content at query #60
#--------------------------

```python
def test_dumps_with_bytes_serializer():
    serializer = Serializer(b"secret", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #61
#--------------------------

```python
def test_iter_unsigners_with_dict_fallback():
    serializer = Serializer(b"secret", fallback_signers=[{"key_derivation": "hmac"}])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)


# LLM-generated content at query #62
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
            return f"loaded-{payload.decode('utf-8')}"
        def dumps(self, obj):
            return f"dumped-{obj}".encode('utf-8')

    serializer = Serializer("secret-key", serializer=CustomSerializer())
    payload = b"test-data"
    result = serializer.load_payload(payload)
    assert result == "loaded-test-data"

def test_load_payload_with_text_serializer():
    serializer = Serializer("secret-key")
    payload = '{"key": "value"}'.encode('utf-8')
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}

def test_load_payload_with_binary_serializer():
    class BinarySerializer:
        def loads(self, payload):
            return f"binary-{payload}"
        def dumps(self, obj):
            return f"binary-{obj}".encode('utf-8')

    serializer = Serializer("secret-key", serializer=BinarySerializer())
    payload = b"test-data"
    result = serializer.load_payload(payload)
    assert result == "binary-test-data"

def test_load_payload_raises_bad_payload_on_error():
    serializer = Serializer("secret-key")
    payload = b"invalid-json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #63
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

def test_serializer_constructor_with_iterable_secret_key():
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


# LLM-generated content at query #64
#--------------------------

```python
def test_salt_is_none():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #65
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
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret-key", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret-key", signer=custom_signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
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
    fallback_signers = [{"key": "value"}]
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


# LLM-generated content at query #66
#--------------------------

```python
def test_dumps_serializes_object():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    serialized = serializer.dumps(obj)
    assert isinstance(serialized, str)


# LLM-generated content at query #67
#--------------------------

```python
def test_fallback_signers_not_none():
    serializer = Serializer(b"secret-key", fallback_signers=[{}])
    assert serializer.fallback_signers is not None


# LLM-generated content at query #68
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

def test_serializer_constructor_with_iterable_keys():
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
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
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

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #69
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
    serializer = Serializer("secret-key")
    payload = b"invalid json"
    try:
        serializer.load_payload(payload)
        assert False, "Expected BadPayload exception"
    except BadPayload as e:
        assert "Could not load the payload" in str(e)


# LLM-generated content at query #70
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (), {})
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
    serializer = Serializer("secret", signer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {"key": "value"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"key": "value"}]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret", serializer_kwargs={"key": "value"})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer is Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer is Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"key": "value"}

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


# LLM-generated content at query #71
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

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_secret_key_property():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_key == b"secret2"


# LLM-generated content at query #72
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
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

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
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
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
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #73
#--------------------------

```python
def test_loads_deserializes_payload():
    serializer = _PDataSerializer()
    payload = b'some_serialized_data'
    result = serializer.loads(payload)
    assert isinstance(result, object)


# LLM-generated content at query #74
#--------------------------

```python
def test_load_payload_with_bytes_serializer():
    serializer = Serializer(b"secret", serializer=json)
    assert not serializer.is_text_serializer


# LLM-generated content at query #75
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
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
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

def test_serializer_constructor_with_bytes_salt():
    serializer = Serializer("secret", salt=b"custom_salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom_salt"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #76
#--------------------------

```python
def test_salt_none_predicate():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #77
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

def test_serializer_constructor_with_iterable_secret_keys():
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
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
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


# LLM-generated content at query #78
#--------------------------

```python
def test_iter_unsigners_with_dict_fallback():
    serializer = Serializer(b"secret", fallback_signers=[{"key1": "value1"}])
    signers = list(serializer.iter_unsigners())
    assert len(signers) == 2
    assert isinstance(signers[0], Signer)
    assert isinstance(signers[1], Signer)


# LLM-generated content at query #79
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

def test_serializer_constructor_with_iterable_secret_key():
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


# LLM-generated content at query #80
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
    serializer = Serializer(["secret-key-1", "secret-key-2"])
    assert serializer.secret_keys == [b"secret-key-1", b"secret-key-2"]
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

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    custom_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
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
    custom_signer = type("CustomSigner", (), {})
    serializer = Serializer("secret-key", signer=custom_signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
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
    fallback_signers = [{"key1": "value1"}, (Signer, {"key2": "value2"}), Signer]
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


# LLM-generated content at query #81
#--------------------------

```python
def test_loads_deserializes_payload():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    assert serializer.loads(payload) == {"key": "value"}


# LLM-generated content at query #82
#--------------------------

```python
def test_dumps_returns_bytes_when_not_text_serializer():
    serializer = Serializer(b"secret", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #83
#--------------------------

```python
def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, _TSerialized)


# LLM-generated content at query #84
#--------------------------

```python
def test_serializer_constructor_with_single_string_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_single_bytes_key():
    serializer = Serializer(b"secret")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_multiple_string_keys():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_multiple_bytes_keys():
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


# LLM-generated content at query #85
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
    serializer = Serializer("secret", serializer=json)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    serializer = Serializer("secret", signer=Signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
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
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

def test_serializer_constructor_with_bytes_serializer():
    class BytesSerializer:
        @staticmethod
        def dumps(obj):
            return json.dumps(obj).encode("utf-8")

        @staticmethod
        def loads(s):
            return json.loads(s.decode("utf-8"))

    serializer = Serializer("secret", serializer=BytesSerializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == BytesSerializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_secret_key_property():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_key == b"secret2"


# LLM-generated content at query #86
#--------------------------

```python
def test_load_payload_raises_bad_payload_on_unserializing_exception():
    serializer = Serializer(b"secret-key")
    payload = b"invalid-payload"

    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(payload)

    assert exc_info.value.original_error is not None


# LLM-generated content at query #87
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": x}}
    serializer = Serializer("secret", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
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

def test_serializer_constructor_with_custom_signer():
    custom_signer = Signer
    serializer = Serializer("secret", signer=custom_signer)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
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
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #88
#--------------------------

```python
def test_serializer_not_none():
    serializer = Serializer(b"secret-key", serializer="custom_serializer")
    assert serializer.serializer == "custom_serializer"


# LLM-generated content at query #89
#--------------------------

```python
def test_dumps_returns_bytes_when_not_text_serializer():
    serializer = Serializer(b"secret", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)


# LLM-generated content at query #90
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
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
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
    custom_signer = type("CustomSigner", (), {})
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
    fallback_signers = [{"sep": "|"}, (type("CustomSigner", (), {}), {"sep": "|"})]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #91
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
    serializer = Serializer("secret", salt="custom-salt")
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"custom-salt"
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
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
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
    serializer = Serializer("secret", serializer_kwargs={"indent": 4})
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 4}

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
    fallback_signers = [{"sep": "?"}, (type("CustomSigner", (), {}), {"sep": "!"})]
    serializer = Serializer("secret", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_all_parameters():
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": x}}
    custom_signer = type("CustomSigner", (), {})
    fallback_signers = [{"sep": "?"}, (type("CustomSigner", (), {}), {"sep": "!"})]
    serializer = Serializer(
        ["secret1", "secret2"],
        salt="custom-salt",
        serializer=custom_serializer,
        serializer_kwargs={"indent": 4},
        signer=custom_signer,
        signer_kwargs={"sep": "?"},
        fallback_signers=fallback_signers,
    )
    assert serializer.secret_keys == [b"secret1", b"secret2"]
    assert serializer.salt == b"custom-salt"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
    assert serializer.signer_kwargs == {"sep": "?"}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {"indent": 4}


# LLM-generated content at query #92
#--------------------------

```python
def test_dumps_returns_serialized_data():
    serializer = _PDataSerializer()
    obj = {"key": "value"}
    result = serializer.dumps(obj)
    assert isinstance(result, _TSerialized)


# LLM-generated content at query #93
#--------------------------

```python
def test_load_payload_raises_bad_payload_on_exception():
    serializer = Serializer(b"secret")
    with pytest.raises(BadPayload) as exc_info:
        serializer.load_payload(b"invalid payload")
    assert "Could not load the payload because an exception occurred on unserializing the data." in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, Exception)


# LLM-generated content at query #94
#--------------------------

```python
def test_salt_is_none_when_not_provided():
    serializer = Serializer(b"secret-key")
    assert serializer.salt is None


# LLM-generated content at query #95
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

def test_serializer_constructor_with_bytes_list_secret_key():
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


# LLM-generated content at query #96
#--------------------------

```python
def test_salt_is_none():
    serializer = Serializer(secret_key="test", salt=None)
    assert serializer.salt is None


# LLM-generated content at query #97
#--------------------------

```python
def test_iter_unsigners_with_tuple_fallback():
    serializer = Serializer(b"secret", fallback_signers=[(Signer, {"digest_method": "sha256"})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].digest_method == "sha256"


# LLM-generated content at query #98
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

def test_serializer_constructor_with_string_key():
    serializer = Serializer("secret")
    assert serializer.secret_keys == [b"secret"]

def test_serializer_constructor_with_key_list():
    serializer = Serializer(["secret1", "secret2"])
    assert serializer.secret_keys == [b"secret1", b"secret2"]

def test_serializer_constructor_with_custom_salt():
    serializer = Serializer(b"secret", salt="custom-salt")
    assert serializer.salt == b"custom-salt"

def test_serializer_constructor_with_none_salt():
    serializer = Serializer(b"secret", salt=None)
    assert serializer.salt is None

def test_serializer_constructor_with_custom_serializer():
    custom_serializer = {"dumps": lambda x: str(x).encode(), "loads": lambda x: int(x.decode())}
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
    fallback = [{"sep": "|"}, (Signer, {"digest_method": "sha256"})]
    serializer = Serializer(b"secret", fallback_signers=fallback)
    assert serializer.fallback_signers == fallback

def test_serializer_secret_key_property():
    serializer = Serializer(["key1", "key2", "key3"])
    assert serializer.secret_key == b"key3"


# LLM-generated content at query #99
#--------------------------

```python
def test_loads_deserializes_payload():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #100
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

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

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
    serializer = Serializer("secret-key", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #101
#--------------------------

```python
def test_load_payload_with_non_text_serializer():
    serializer = Serializer(b"secret-key")
    payload = b'{"key": "value"}'
    result = serializer.load_payload(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #102
#--------------------------

```python
def test_dumps_with_string_serializer():
    serializer = Serializer("secret-key", serializer=json)
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_bytes_serializer():
    serializer = Serializer("secret-key", serializer=json, salt=b"salt")
    result = serializer.dumps({"key": "value"})
    assert isinstance(result, bytes)
    assert b"key" in result

def test_dumps_with_custom_salt():
    serializer = Serializer("secret-key")
    result = serializer.dumps({"key": "value"}, salt="custom-salt")
    assert isinstance(result, str)
    assert "key" in result

def test_dumps_with_bytes_input():
    serializer = Serializer("secret-key")
    result = serializer.dumps(b"bytes-input")
    assert isinstance(result, str)
    assert "bytes-input" in result

def test_dumps_with_empty_object():
    serializer = Serializer("secret-key")
    result = serializer.dumps({})
    assert isinstance(result, str)

def test_dumps_with_list_object():
    serializer = Serializer("secret-key")
    result = serializer.dumps([1, 2, 3])
    assert isinstance(result, str)
    assert "1" in result

def test_dumps_with_none_object():
    serializer = Serializer("secret-key")
    result = serializer.dumps(None)
    assert isinstance(result, str)
    assert "null" in result


# LLM-generated content at query #103
#--------------------------

```python
def test_load_payload_raises_bad_payload():
    serializer = Serializer(b"secret")
    payload = b"invalid_payload"

    try:
        serializer.load_payload(payload)
    except BadPayload as e:
        assert str(e) == "Could not load the payload because an exception occurred on unserializing the data."
        assert isinstance(e.original_error, json.JSONDecodeError)
    else:
        assert False, "Expected BadPayload exception"


# LLM-generated content at query #104
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

def test_serializer_constructor_with_serializer_kwargs():
    serializer = Serializer("secret-key", serializer_kwargs={"indent": 2})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {"indent": 2}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret-key", signer=custom_signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_signer_kwargs():
    serializer = Serializer("secret-key", signer_kwargs={"sep": "|"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {"sep": "|"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": "|"}, (Signer, {"sep": "|"}), Signer]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #105
#--------------------------

```python
def test_iter_unsigners_default():
    serializer = Serializer("secret-key")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == ["secret-key"]
    assert unsigners[0].salt == b"itsdangerous"

def test_iter_unsigners_custom_salt():
    serializer = Serializer("secret-key", salt="custom-salt")
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == ["secret-key"]
    assert unsigners[0].salt == b"custom-salt"

def test_iter_unsigners_with_fallback_signers_dict():
    serializer = Serializer("secret-key", fallback_signers=[{"digest_method": "md5"}])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == ["secret-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == ["secret-key"]
    assert unsigners[1].salt == b"itsdangerous"
    assert unsigners[1].digest_method.name == "md5"

def test_iter_unsigners_with_fallback_signers_tuple():
    serializer = Serializer("secret-key", fallback_signers=[(Signer, {"digest_method": "md5"})])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == ["secret-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == ["secret-key"]
    assert unsigners[1].salt == b"itsdangerous"
    assert unsigners[1].digest_method.name == "md5"

def test_iter_unsigners_with_fallback_signers_class():
    serializer = Serializer("secret-key", fallback_signers=[Signer])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 2
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == ["secret-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == ["secret-key"]
    assert unsigners[1].salt == b"itsdangerous"

def test_iter_unsigners_with_multiple_fallback_signers():
    serializer = Serializer(
        "secret-key",
        fallback_signers=[
            {"digest_method": "md5"},
            (Signer, {"digest_method": "sha256"}),
            Signer
        ]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 4
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == ["secret-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == ["secret-key"]
    assert unsigners[1].salt == b"itsdangerous"
    assert unsigners[1].digest_method.name == "md5"
    assert isinstance(unsigners[2], Signer)
    assert unsigners[2].secret_keys == ["secret-key"]
    assert unsigners[2].salt == b"itsdangerous"
    assert unsigners[2].digest_method.name == "sha256"
    assert isinstance(unsigners[3], Signer)
    assert unsigners[3].secret_keys == ["secret-key"]
    assert unsigners[3].salt == b"itsdangerous"

def test_iter_unsigners_with_key_rotation():
    serializer = Serializer(["old-key", "new-key"])
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 1
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == ["old-key", "new-key"]
    assert unsigners[0].salt == b"itsdangerous"

def test_iter_unsigners_with_key_rotation_and_fallback():
    serializer = Serializer(
        ["old-key", "new-key"],
        fallback_signers=[{"digest_method": "md5"}]
    )
    unsigners = list(serializer.iter_unsigners())
    assert len(unsigners) == 3
    assert isinstance(unsigners[0], Signer)
    assert unsigners[0].secret_keys == ["old-key", "new-key"]
    assert unsigners[0].salt == b"itsdangerous"
    assert isinstance(unsigners[1], Signer)
    assert unsigners[1].secret_keys == ["old-key"]
    assert unsigners[1].salt == b"itsdangerous"
    assert unsigners[1].digest_method.name == "md5"
    assert isinstance(unsigners[2], Signer)
    assert unsigners[2].secret_keys == ["new-key"]
    assert unsigners[2].salt == b"itsdangerous"
    assert unsigners[2].digest_method.name == "md5"


# LLM-generated content at query #106
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

def test_serializer_constructor_with_list_secret_keys():
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
    serializer = Serializer("secret-key", signer_kwargs={"sep": ";"})
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {"sep": ";"}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_fallback_signers():
    fallback_signers = [{"sep": ";"}, (Signer, {"sep": ":"})]
    serializer = Serializer("secret-key", fallback_signers=fallback_signers)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == Serializer.default_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == fallback_signers
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

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"bytes", "loads": lambda x: {"bytes": True}}
    serializer = Serializer("secret-key", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Serializer.default_signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}


# LLM-generated content at query #107
#--------------------------

```python
def test_loads_deserializes_payload():
    serializer = _PDataSerializer()
    payload = b'{"key": "value"}'
    result = serializer.loads(payload)
    assert result == {"key": "value"}


# LLM-generated content at query #108
#--------------------------

```python
def test_serializer_not_none():
    serializer = Serializer(b"secret-key", serializer="custom_serializer")
    assert serializer.serializer == "custom_serializer"


# LLM-generated content at query #109
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

def test_serializer_constructor_with_iterable_secret_keys():
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
    custom_serializer = {"dumps": lambda x: "custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret-key", serializer=custom_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == custom_serializer
    assert serializer.is_text_serializer is True
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_bytes_serializer():
    bytes_serializer = {"dumps": lambda x: b"custom", "loads": lambda x: {"custom": True}}
    serializer = Serializer("secret-key", serializer=bytes_serializer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == bytes_serializer
    assert serializer.is_text_serializer is False
    assert serializer.signer == Signer
    assert serializer.signer_kwargs == {}
    assert serializer.fallback_signers == []
    assert serializer.serializer_kwargs == {}

def test_serializer_constructor_with_custom_signer():
    custom_signer = type("CustomSigner", (Signer,), {})
    serializer = Serializer("secret-key", signer=custom_signer)
    assert serializer.secret_keys == [b"secret-key"]
    assert serializer.salt == b"itsdangerous"
    assert serializer.serializer == json
    assert serializer.is_text_serializer is True
    assert serializer.signer == custom_signer
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

def test_serializer_secret_key_property():
    serializer = Serializer(["key1", "key2"])
    assert serializer.secret_key == b"key2"


