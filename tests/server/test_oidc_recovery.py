"""Regression coverage for OIDC provider-error recovery."""

from __future__ import annotations

import html
import re
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omnigent.server.admin_list import AdminList
from omnigent.server.auth import UnifiedAuthProvider
from omnigent.server.oidc import OIDCConfig, derive_code_challenge
from omnigent.server.routes.auth import create_auth_router

_SECRET = b"oidc-recovery-test-only-secret-32"
_ISSUER = "https://idp.example.test"


class TestOidcRecovery(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.admins = AdminList(Path(directory) / "admins")
        self.stack.enter_context(
            patch(
                "omnigent.server.routes.auth.resolve_allowed_domains_path",
                return_value=Path(directory) / "domains",
            )
        )
        self.forms: list[dict[str, str]] = []
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.claims: dict[str, object] = {}

        async def token_response(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
            self.assertEqual(url, f"{_ISSUER}/token")
            self.forms.append(kwargs["data"])
            token = jwt.encode(
                {
                    "iss": _ISSUER,
                    "aud": "test-client",
                    "sub": "test-subject",
                    "email": "User@example.test",
                    "email_verified": True,
                    "exp": int(time.time()) + 300,
                    **self.claims,
                },
                self.key,
                algorithm="RS256",
            )
            return httpx.Response(200, json={"id_token": token})

        self.stack.enter_context(patch.object(httpx.AsyncClient, "post", new=token_response))
        self.stack.enter_context(
            patch.object(
                jwt.PyJWKClient,
                "get_signing_key_from_jwt",
                return_value=SimpleNamespace(key=self.key.public_key()),
            )
        )
        self.build_client()

    def build_client(self, *, secure: bool = False, base_path: str = "", invites: bool = False):
        self.prefix = f"{base_path}/auth"
        origin = "https://omni.example.test" if secure else "http://localhost:8000"
        self.config = OIDCConfig(
            issuer=_ISSUER,
            client_id="test-client",
            client_secret="test-client-secret",
            redirect_uri=f"{origin}{self.prefix}/callback",
            cookie_secret=_SECRET,
            scopes="openid email profile",
            session_ttl_hours=8,
            logout_redirect_uri=None,
            allowed_domains=None,
            provider_type="oidc",
            authorization_endpoint=f"{_ISSUER}/authorize",
            token_endpoint=f"{_ISSUER}/token",
            jwks_uri=f"{_ISSUER}/jwks",
            userinfo_endpoint=None,
            allow_invites=invites,
        )
        self.app = FastAPI()
        self.app.state.base_path = base_path
        self.app.include_router(
            create_auth_router(
                UnifiedAuthProvider(source="oidc", oidc_config=self.config),
                permission_store=None,
                admin_list=self.admins,
                account_store=MagicMock() if invites else None,
            ),
            prefix=self.prefix,
        )
        self.client = self.stack.enter_context(
            TestClient(self.app, base_url=origin, follow_redirects=False)
        )
        self.cookie = "__Host-ap_auth_state" if secure else "ap_auth_state"

    def login(self, **params: str) -> tuple[httpx.Response, dict]:
        response = self.client.get(f"{self.prefix}/login", params=params)
        self.assertEqual(response.status_code, 302)
        return response, self.state()

    def state(self) -> dict:
        return jwt.decode(self.client.cookies.get(self.cookie), _SECRET, algorithms=["HS256"])

    def replace_state(self, payload: dict, secret: bytes = _SECRET) -> None:
        cookie = next(cookie for cookie in self.client.cookies.jar if cookie.name == self.cookie)
        self.client.cookies.clear()
        self.client.cookies.set(
            self.cookie,
            jwt.encode(payload, secret, algorithm="HS256"),
            domain=cookie.domain,
            path=cookie.path,
        )

    def expired(self, state: str, **extra: str) -> httpx.Response:
        return self.client.get(
            f"{self.prefix}/callback",
            params={
                "state": state,
                "error": "temporarily_unavailable",
                "error_description": "authentication_expired",
                **extra,
            },
        )

    def finish(self, state: str) -> httpx.Response:
        return self.client.get(
            f"{self.prefix}/callback", params={"state": state, "code": "test-code"}
        )

    def recovery_link(self, response: httpx.Response) -> str:
        return html.unescape(re.search(r'href="([^"]+)"', response.text).group(1))

    def assert_manual_restart(self, response: httpx.Response) -> None:
        self.assertEqual(response.status_code, 400)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertEqual(self.recovery_link(response), f"{self.prefix}/login")
        self.assertNotIn("location", response.headers)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")
        self.assertEqual(self.forms, [])

    def test_expired_flow_rotates_state_and_pkce_once(self) -> None:
        original_response, original = self.login(return_to="/sessions/test")
        response = self.expired(original["state"])
        self.assertEqual(response.status_code, 302)
        retried = self.state()
        self.assertNotEqual(retried["state"], original["state"])
        self.assertNotEqual(retried["code_verifier"], original["code_verifier"])
        self.assertTrue(retried["oidc_retry"])
        self.assertEqual(retried["return_to"], "/sessions/test")
        query = parse_qs(urlsplit(response.headers["location"]).query)
        self.assertEqual(query["state"], [retried["state"]])
        self.assertEqual(
            query["code_challenge"], [derive_code_challenge(retried["code_verifier"])]
        )
        for name in (
            "client_id",
            "redirect_uri",
            "scope",
            "response_type",
            "code_challenge_method",
        ):
            self.assertEqual(
                query[name], parse_qs(urlsplit(original_response.headers["location"]).query)[name]
            )
        self.assertEqual(self.forms, [])
        self.assertIsNone(self.client.cookies.get(self.config.session_cookie_name))

    def test_repeated_expiry_stops_and_clears_cookie(self) -> None:
        _, original = self.login()
        self.expired(original["state"])
        retried = self.state()
        response = self.expired(retried["state"], oidc_retry="0")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Your sign-in session expired", response.text)
        self.assertNotIn("location", response.headers)
        self.assertEqual(urlsplit(self.recovery_link(response)).path, "/auth/login")
        self.assertIsNone(self.client.cookies.get(self.cookie))
        self.assertEqual(self.forms, [])

    def test_provider_errors_do_not_retry_or_reflect_details(self) -> None:
        for error, description in (
            ("access_denied", "declined"),
            ("temporarily_unavailable", "another_problem"),
            ("server_error", "authentication_expired"),
            ("<script>secret-value</script>", "<script>private-detail</script>"),
        ):
            with self.subTest(error=error):
                _, original = self.login()
                with self.assertLogs("omnigent.server.routes.auth", level="WARNING") as logs:
                    response = self.client.get(
                        f"{self.prefix}/callback",
                        params={
                            "state": original["state"],
                            "error": error,
                            "error_description": description,
                        },
                    )
                self.assertEqual(response.status_code, 400)
                self.assertIn("Sign-in could not be completed", response.text)
                self.assertNotIn("private-detail", response.text + str(logs.output))
                self.assertNotIn("secret-value", response.text + str(logs.output))
                self.assertNotIn(original["state"], str(logs.output))
                self.assertNotIn("location", response.headers)
                self.assertIsNone(self.client.cookies.get(self.cookie))
        self.assertEqual(self.forms, [])

    def test_missing_state_offers_restart_without_changing_active_cookie(self) -> None:
        self.login(return_to="/sessions/active", ticket="active-ticket")
        cookie = self.client.cookies.get(self.cookie)
        response = self.client.get(
            f"{self.prefix}/callback",
            params={
                "error": "<script>private-error</script>",
                "error_description": "private-description",
                "code": "private-code",
                "return_to": "https://attacker.example/private-path",
                "ticket": "private-ticket",
                "invite": "private-invite",
                "reauth": "1",
            },
        )
        self.assert_manual_restart(response)
        self.assertNotIn("private-", response.text)
        self.assertNotIn("active-ticket", response.text)
        self.assertNotIn("set-cookie", response.headers)
        self.assertEqual(self.client.cookies.get(self.cookie), cookie)

    def test_missing_cookie_offers_manual_restart(self) -> None:
        response = self.expired("private-state", code="private-code", ticket="private-ticket")
        self.assert_manual_restart(response)
        self.assertNotIn("private-", response.text)
        self.assertNotIn("set-cookie", response.headers)
        self.assertIsNone(self.client.cookies.get(self.config.session_cookie_name))

    def test_invalid_expired_or_mismatched_state_cannot_retry(self) -> None:
        for kind in ("signature", "expired", "mismatch"):
            with self.subTest(kind=kind):
                _, original = self.login()
                if kind == "signature":
                    self.replace_state(original, b"different-test-only-signing-key")
                elif kind == "expired":
                    self.replace_state({**original, "exp": int(time.time()) - 30})
                response = self.expired(
                    "another-state" if kind == "mismatch" else original["state"],
                    code="private-code",
                    return_to="https://attacker.example/private-path",
                    ticket="private-ticket",
                    invite="private-invite",
                    reauth="1",
                )
                self.assert_manual_restart(response)
                self.assertNotIn("private-", response.text)
                self.assertIsNone(self.client.cookies.get(self.config.session_cookie_name))
                if kind == "mismatch":
                    self.assertNotIn("set-cookie", response.headers)
                    self.assertEqual(self.state(), original)
                else:
                    self.assertIsNone(self.client.cookies.get(self.cookie))

    def test_old_tab_error_preserves_second_tab_pending_login(self) -> None:
        _, first = self.login(return_to="/sessions/first", ticket="first-ticket")
        _, second = self.login(return_to="/sessions/second")
        cookie = self.client.cookies.get(self.cookie)
        response = self.expired(first["state"])
        self.assert_manual_restart(response)
        self.assertNotIn("set-cookie", response.headers)
        self.assertNotIn("/sessions/", response.text)
        self.assertNotIn("first-ticket", response.text)
        self.assertEqual(self.client.cookies.get(self.cookie), cookie)
        success = self.finish(second["state"])
        self.assertEqual(success.status_code, 302)
        self.assertEqual(success.headers["location"], "/sessions/second")
        self.assertEqual(self.forms[0]["code_verifier"], second["code_verifier"])

    def test_invalid_code_callbacks_keep_json_rejection(self) -> None:
        for kind, message in (
            ("missing_state", "Missing code or state parameter"),
            ("missing_cookie", "Missing auth state cookie"),
            ("signature", "Invalid or expired auth state"),
            ("expired", "Invalid or expired auth state"),
            ("mismatch", "State mismatch (possible CSRF)"),
        ):
            with self.subTest(kind=kind):
                _, original = self.login()
                if kind == "missing_cookie":
                    self.client.cookies.clear()
                elif kind == "signature":
                    self.replace_state(original, b"different-test-only-signing-key")
                elif kind == "expired":
                    self.replace_state({**original, "exp": int(time.time()) - 30})
                params = {"code": "test-code"}
                if kind != "missing_state":
                    params["state"] = "another-state" if kind == "mismatch" else original["state"]
                response = self.client.get(f"{self.prefix}/callback", params=params)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json(), {"error": message})
                self.assertNotIn("set-cookie", response.headers)
                self.assertNotIn("location", response.headers)
                self.assertIsNone(self.client.cookies.get(self.config.session_cookie_name))
        self.assertEqual(self.forms, [])

    def test_old_tab_error_preserves_second_tab_completed_session(self) -> None:
        _, first = self.login()
        _, second = self.login()
        self.assertEqual(self.finish(second["state"]).status_code, 302)
        session = self.client.cookies.get(self.config.session_cookie_name)
        self.forms.clear()
        response = self.expired(first["state"])
        self.assert_manual_restart(response)
        self.assertNotIn("set-cookie", response.headers)
        self.assertEqual(self.client.cookies.get(self.config.session_cookie_name), session)

    def test_expired_cookie_restart_starts_fresh_without_old_context(self) -> None:
        self.build_client(invites=True)
        _, original = self.login(
            return_to="/sessions/old", ticket="old-ticket", invite="old-invite", reauth="1"
        )
        self.replace_state({**original, "exp": int(time.time()) - 30})
        response = self.expired(original["state"])
        self.assert_manual_restart(response)
        self.assertIsNone(self.client.cookies.get(self.cookie))
        restart = self.client.get(self.recovery_link(response))
        self.assertEqual(restart.status_code, 302)
        fresh = self.state()
        self.assertNotEqual(fresh["state"], original["state"])
        self.assertNotEqual(fresh["code_verifier"], original["code_verifier"])
        self.assertEqual(fresh["return_to"], "/")
        for key in ("ticket", "invite", "reauth_at", "oidc_retry"):
            self.assertNotIn(key, fresh)
        self.assertEqual(self.finish(fresh["state"]).status_code, 302)

    def test_unverified_restart_respects_base_path(self) -> None:
        self.build_client(base_path="/omnigent")
        self.assert_manual_restart(self.expired("private-state"))

    def test_invalid_https_cookie_is_cleared_with_matching_attributes(self) -> None:
        self.build_client(secure=True)
        _, original = self.login()
        self.replace_state({**original, "exp": int(time.time()) - 30})
        response = self.expired(original["state"])
        self.assert_manual_restart(response)
        header = response.headers["set-cookie"]
        for attribute in (
            "__Host-ap_auth_state=",
            "Max-Age=0",
            "Path=/",
            "Secure",
            "HttpOnly",
            "SameSite=lax",
        ):
            self.assertIn(attribute, header)
        self.assertIsNone(self.client.cookies.get(self.cookie))

    def test_retry_marker_query_parameter_is_ignored(self) -> None:
        _, original = self.login(oidc_retry="1")
        self.assertNotIn("oidc_retry", original)
        self.assertEqual(self.expired(original["state"]).status_code, 302)

    def test_old_callback_cannot_use_rotated_cookie(self) -> None:
        _, original = self.login()
        self.expired(original["state"])
        self.assertEqual(self.expired(original["state"]).status_code, 400)
        self.assertEqual(self.finish(original["state"]).status_code, 400)
        self.assertEqual(self.forms, [])

    def test_error_with_code_does_not_exchange_or_mint_session(self) -> None:
        _, original = self.login()
        self.assertEqual(self.expired(original["state"], code="test-code").status_code, 302)
        self.assertEqual(self.forms, [])
        self.assertIsNone(self.client.cookies.get(self.config.session_cookie_name))

    def test_missing_code_without_error_remains_invalid(self) -> None:
        _, original = self.login()
        response = self.client.get(f"{self.prefix}/callback", params={"state": original["state"]})
        self.assertEqual(response.json()["error"], "Missing code or state parameter")

    def test_success_after_retry_uses_fresh_verifier_and_real_jwt(self) -> None:
        _, original = self.login(return_to="/sessions/test")
        self.expired(original["state"])
        retried = self.state()
        response = self.finish(retried["state"])
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["location"], "/sessions/test")
        self.assertEqual(self.forms[0]["code_verifier"], retried["code_verifier"])
        session = jwt.decode(
            self.client.cookies.get(self.config.session_cookie_name), _SECRET, algorithms=["HS256"]
        )
        self.assertEqual(session["sub"], "user@example.test")
        self.assertIsNone(self.client.cookies.get(self.cookie))

    def test_native_errors_return_immediately_without_issuing_credentials(self) -> None:
        for redirect in ("http://127.0.0.1:53682/callback", "ai.omnigent.ios:/oauth/callback"):
            for error in ("access_denied", "temporarily_unavailable", ""):
                for code in (None, "test-code"):
                    with self.subTest(redirect=redirect, error=error, code=code):
                        _, original = self.login(
                            native_redirect_uri=redirect,
                            native_state="native-state",
                            code_challenge=derive_code_challenge("v" * 64),
                            code_challenge_method="S256",
                        )
                        params = {
                            "state": original["state"],
                            "error": error,
                            "error_description": "authentication_expired",
                        }
                        if code is not None:
                            params["code"] = code
                        response = self.client.get(f"{self.prefix}/callback", params=params)
                        self.assertEqual(response.status_code, 302)
                        target = urlsplit(response.headers["location"])
                        self.assertEqual(target._replace(query="").geturl(), redirect)
                        query = parse_qs(target.query)
                        self.assertEqual(query["state"], ["native-state"])
                        self.assertEqual(query["error"], ["access_denied"])
                        self.assertNotIn("code", query)
                        self.assertEqual(self.forms, [])
                        self.assertIsNone(self.client.cookies.get(self.cookie))
                        self.assertIsNone(self.client.cookies.get(self.config.session_cookie_name))

    def test_empty_error_with_code_does_not_exchange_or_mint_session(self) -> None:
        _, original = self.login()
        response = self.client.get(
            f"{self.prefix}/callback",
            params={"state": original["state"], "error": "", "code": "test-code"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.forms, [])
        self.assertIsNone(self.client.cookies.get(self.config.session_cookie_name))

    def test_cli_ticket_survives_retry_and_is_single_use(self) -> None:
        ticket = self.client.post(f"{self.prefix}/cli-login").json()["ticket"]
        _, original = self.login(ticket=ticket)
        self.expired(original["state"])
        retried = self.state()
        self.assertEqual(retried["ticket"], ticket)
        self.assertEqual(
            self.client.get(f"{self.prefix}/cli-poll", params={"ticket": ticket}).status_code, 202
        )
        response = self.finish(retried["state"])
        self.assertEqual(response.status_code, 200)
        self.assertIn("Login successful", response.text)
        poll = self.client.get(f"{self.prefix}/cli-poll", params={"ticket": ticket})
        self.assertEqual(poll.status_code, 200)
        self.assertEqual(poll.json()["user_id"], "user@example.test")
        self.assertEqual(
            self.client.get(f"{self.prefix}/cli-poll", params={"ticket": ticket}).status_code, 410
        )

    def test_manual_recovery_preserves_cli_ticket_and_destination(self) -> None:
        ticket = self.client.post(f"{self.prefix}/cli-login").json()["ticket"]
        _, original = self.login(ticket=ticket, return_to="/sessions/test")
        self.expired(original["state"])
        response = self.expired(self.state()["state"])
        link = self.recovery_link(response)
        self.assertEqual(parse_qs(urlsplit(link).query)["ticket"], [ticket])
        self.assertEqual(self.client.get(link).status_code, 302)
        fresh = self.state()
        self.assertNotIn("oidc_retry", fresh)
        self.assertEqual(fresh["return_to"], "/sessions/test")
        self.assertEqual(self.finish(fresh["state"]).status_code, 200)

    def test_reauth_requirement_survives_retry(self) -> None:
        _, original = self.login(reauth="1")
        self.replace_state({**original, "reauth_at": int(time.time()) - 60})
        response = self.expired(original["state"])
        retried = self.state()
        query = parse_qs(urlsplit(response.headers["location"]).query)
        self.assertEqual(query["prompt"], ["login"])
        self.assertEqual(query["max_age"], ["0"])
        self.assertGreater(retried["reauth_at"], int(time.time()) - 60)
        self.claims["auth_time"] = int(time.time()) - 60
        self.assertEqual(self.finish(retried["state"]).status_code, 403)
        self.assertIsNone(self.client.cookies.get(self.config.session_cookie_name))

    def test_https_retry_preserves_host_cookie_attributes(self) -> None:
        self.build_client(secure=True)
        _, original = self.login()
        response = self.expired(original["state"])
        self.assertEqual(response.status_code, 302)
        cookie = response.headers["set-cookie"]
        for part in ("__Host-ap_auth_state=", "HttpOnly", "Secure", "SameSite=lax", "Path=/"):
            self.assertIn(part, cookie)
        response = self.expired(self.state()["state"])
        self.assertIn("Max-Age=0", response.headers["set-cookie"])
        self.assertIn("Secure", response.headers["set-cookie"])

    def test_manual_recovery_keeps_forced_reauthentication(self) -> None:
        _, original = self.login(reauth="1")
        self.expired(original["state"])
        response = self.expired(self.state()["state"])
        link = self.recovery_link(response)
        self.assertEqual(parse_qs(urlsplit(link).query)["reauth"], ["1"])
        response = self.client.get(link)
        self.assertEqual(
            parse_qs(urlsplit(response.headers["location"]).query)["prompt"], ["login"]
        )
        self.claims["auth_time"] = int(time.time())
        self.assertEqual(self.finish(self.state()["state"]).status_code, 302)

    def test_retry_does_not_extend_cli_ticket_lifetime(self) -> None:
        ticket = self.client.post(f"{self.prefix}/cli-login").json()["ticket"]
        _, original = self.login(ticket=ticket)
        self.expired(original["state"])
        with patch("omnigent.server.routes.auth.time.time", return_value=time.time() + 301):
            response = self.client.get(f"{self.prefix}/cli-poll", params={"ticket": ticket})
        self.assertEqual(response.status_code, 410)

    def test_subpath_retry_and_recovery_stay_under_mount(self) -> None:
        self.build_client(base_path="/proxy/42")
        _, original = self.login()
        self.expired(original["state"])
        retried = self.state()
        self.assertEqual(retried["return_to"], "/proxy/42/")
        response = self.expired(retried["state"])
        self.assertEqual(urlsplit(self.recovery_link(response)).path, "/proxy/42/auth/login")

    def test_invite_survives_retry_without_entering_idp_url(self) -> None:
        self.build_client(invites=True)
        _, original = self.login(invite="test-only-invite")
        response = self.expired(original["state"])
        self.assertEqual(self.state()["invite"], "test-only-invite")
        self.assertNotIn("test-only-invite", response.headers["location"])
        response = self.expired(self.state()["state"])
        self.assertEqual(
            parse_qs(urlsplit(self.recovery_link(response)).query)["invite"], ["test-only-invite"]
        )

    def test_unsafe_destinations_are_sanitized_on_retry_and_recovery(self) -> None:
        for destination in (
            "https://evil.example",
            "//evil.example",
            "/\\evil.example",
            '/<script>"',
        ):
            with self.subTest(destination=destination):
                _, original = self.login()
                self.replace_state({**original, "return_to": destination})
                self.expired(original["state"])
                retried = self.state()
                self.assertFalse(retried["return_to"].startswith(("https:", "//", "/\\")))
                response = self.expired(retried["state"])
                self.assertEqual(urlsplit(self.recovery_link(response)).netloc, "")
                self.assertNotIn("<script>", response.text)

    def test_auth_redirects_and_failure_pages_are_not_cached(self) -> None:
        response, original = self.login()
        responses = [
            response,
            self.expired(original["state"]),
            self.expired(self.state()["state"]),
        ]
        for response in responses:
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(response.headers["referrer-policy"], "no-referrer")
