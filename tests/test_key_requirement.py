import base64
import gzip
import json

import pytest

from layered_decoder import LayeredDecoder
from tests.test_cli import run_cli


SALTED = b'Salted__' + bytes(range(8)) + bytes(range(16, 48))
FERNET = b'\x80' + (1700000000).to_bytes(8, 'big') + bytes(range(16)) + bytes(range(16)) + bytes(range(32))


@pytest.mark.parametrize('raw,format_name', [(SALTED, 'OpenSSL salted'), (FERNET, 'Fernet')])
@pytest.mark.parametrize('wrapper', [lambda b: b, base64.b64encode, lambda b: base64.b64encode(b.hex().encode())])
def test_recognized_encryption_stops_with_evidence(raw, format_name, wrapper):
    result = LayeredDecoder().decode(wrapper(raw))
    assert result.status == 'key_required'
    assert result.stopped_reason == 'key_required'
    assert result.key_required is True
    assert result.encryption_format == format_name
    assert result.key_requirement_evidence
    assert result.required_decryption_info
    assert result.final_bytes == raw
    assert not any(step.statistical for step in result.steps)
    assert result.to_dict()['key_required'] is True
    assert 'Decryption key required' in result.summary()


@pytest.mark.parametrize('raw', [b'Hello World', b'Salted__', b'Salted__12345678', FERNET[:-1], bytes(range(256)), gzip.compress(b'Hello World')])
def test_insufficient_evidence_does_not_assert_key_required(raw):
    result = LayeredDecoder(max_depth=1).decode(raw, allow_segmentation=False)
    assert result.key_required is False
    assert result.encryption_format is None


def test_branch_fallback_preserves_key_requirement():
    result = LayeredDecoder().decode_with_branches(base64.b64encode(SALTED))
    assert result.key_required
    assert result.final_bytes == SALTED


def test_cli_reports_key_requirement_and_evidence():
    payload = base64.b64encode(SALTED).decode()
    output = run_cli('-i', payload)
    assert 'Decryption key required' in output
    assert 'Salted__' in output
    assert 'cipher' in output
    data = json.loads(run_cli('--json', '-i', payload))
    assert data['key_required'] is True
    assert data['encryption_format'] == 'OpenSSL salted'


def test_encrypted_payload_inside_compression_keeps_trace():
    result = LayeredDecoder().decode(base64.b64encode(gzip.compress(SALTED)))
    assert result.key_required
    assert result.final_bytes == SALTED
    assert "Gzip/Zlib" in result.layers_detected


def test_depth_limit_still_reports_encryption_exposed_by_last_layer():
    result = LayeredDecoder(max_depth=1).decode(base64.b64encode(SALTED))
    assert result.key_required
    assert len(result.steps) == 1


@pytest.mark.parametrize('raw', [
    b'', b'\x80' * 73, b'\x81' + FERNET[1:],
    b'\x80' + (253402300800).to_bytes(8, 'big') + FERNET[9:],
    b'd41d8cd98f00b204e9800998ecf8427e',
])
def test_headers_and_hash_like_strings_are_not_enough(raw):
    from layered_decoder.key_requirement import identify_key_requirement
    assert identify_key_requirement(raw) is None
