"""Regression checks for transport trust and HTTP dependency security."""

import ssl

import certifi
import pytest
from aiohttp.test_utils import make_mocked_request
from tornado.httputil import HTTPHeaders, HTTPInputError


def test_certifi_bundle_excludes_compromised_emsign_trust_root() -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=certifi.where())
    authorities = context.get_ca_certs()

    # Legitimate clients must still receive a usable CA store.
    assert authorities
    common_names = {
        value
        for authority in authorities
        for relative_name in authority["subject"]
        for key, value in relative_name
        if key == "commonName"
    }
    assert "emSign ECC Root CA - G3" not in common_names


@pytest.mark.parametrize("target", ["/", "http://example.test/", "https://example.test/"])
@pytest.mark.parametrize("encrypted", [False, True])
def test_aiohttp_scheme_comes_from_transport(target: str, encrypted: bool) -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER) if encrypted else None
    request = make_mocked_request("GET", target, sslcontext=context)

    assert request.scheme == ("https" if encrypted else "http")


@pytest.mark.parametrize("continuation", [" folded", "\tfolded"])
def test_tornado_rejects_obsolete_folded_headers(continuation: str) -> None:
    with pytest.raises(HTTPInputError):
        HTTPHeaders.parse(f"X-Test: value\r\n{continuation}\r\n")


def test_tornado_preserves_normal_headers() -> None:
    headers = HTTPHeaders.parse("Content-Type: text/plain\r\nX-Test: value\r\n")

    assert headers["Content-Type"] == "text/plain"
    assert headers["X-Test"] == "value"
