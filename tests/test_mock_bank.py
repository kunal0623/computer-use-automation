"""Contract tests for the mock bank back-office app.

Run from the build repo root with:
    .venv/bin/python -m pytest tests/test_mock_bank.py -q
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from bankgpt_cua.mock_bank.app import create_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def login(client: TestClient, username: str = "operator", password: str = "pw123") -> TestClient:
    resp = client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/search"
    assert "mockbank_session" in resp.cookies
    return client


def test_login_page_fields(client: TestClient) -> None:
    resp = client.get("/login")
    assert resp.status_code == 200
    body = resp.text
    assert "Username" in body
    assert "Password" in body
    assert 'name="username"' in body
    assert 'name="password"' in body
    assert "Log in" in body


def test_login_happy_path_sets_cookie_and_redirects(client: TestClient) -> None:
    resp = client.post(
        "/login",
        data={"username": "operator", "password": "pw123"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/search"
    assert "mockbank_session" in resp.cookies
    session_cookie = resp.cookies["mockbank_session"]
    assert session_cookie


def test_login_empty_credentials_shows_error(client: TestClient) -> None:
    resp = client.post("/login", data={"username": "", "password": ""})
    assert resp.status_code == 200
    assert "Username and password are required" in resp.text


def test_search_requires_login(client: TestClient) -> None:
    resp = client.get("/search", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"


def test_search_page_heading_and_button(client: TestClient) -> None:
    login(client)
    resp = client.get("/search")
    assert resp.status_code == 200
    body = resp.text
    assert "Member Search" in body
    assert "Member ID" in body
    assert 'name="member_id"' in body
    assert "Search member" in body


def test_search_happy_path_for_12345(client: TestClient) -> None:
    login(client)
    resp = client.post("/search", data={"member_id": "12345"}, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/member/12345"


def test_search_unknown_id_shows_not_found(client: TestClient) -> None:
    login(client)
    resp = client.post("/search", data={"member_id": "99999"})
    assert resp.status_code == 200
    assert "No member found for ID 99999" in resp.text


def test_search_malformed_id_shows_validation_error(client: TestClient) -> None:
    login(client)
    resp = client.post("/search", data={"member_id": "abc"})
    assert resp.status_code == 200
    assert "Member ID must be 5 digits" in resp.text


def test_search_blank_id_shows_validation_error(client: TestClient) -> None:
    login(client)
    resp = client.post("/search", data={"member_id": "   "})
    assert resp.status_code == 200
    assert "Member ID must be 5 digits" in resp.text


def test_member_page_contents(client: TestClient) -> None:
    login(client)
    resp = client.get("/member/12345")
    assert resp.status_code == 200
    body = resp.text
    assert "Member 12345" in body
    assert "Savings balance" in body
    assert "$4,321.09" in body
    assert "Close savings account" in body
    assert "Are you sure you want to close this savings account?" in body


def test_member_page_second_member(client: TestClient) -> None:
    login(client)
    resp = client.get("/member/23456")
    assert resp.status_code == 200
    body = resp.text
    assert "Member 23456" in body
    assert "Liam Brooks" in body
    assert "$987.65" in body


def test_member_page_unknown_id(client: TestClient) -> None:
    login(client)
    resp = client.get("/member/99999")
    assert resp.status_code == 200
    assert "No member found for ID 99999" in resp.text


def test_member_requires_login(client: TestClient) -> None:
    resp = client.get("/member/12345", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"


def test_timeout_trigger_clears_session_and_redirects(client: TestClient) -> None:
    login(client)
    # session works before the trigger
    assert client.get("/search").status_code == 200
    resp = client.get("/trigger/timeout", follow_redirects=False)
    assert resp.status_code == 303
    location = resp.headers["location"]
    assert location.startswith("/login")
    # session cookie is cleared
    assert resp.cookies.get("mockbank_session") in (None, "")
    # follow through to the login page and check the expiry text
    page = client.get(location)
    assert page.status_code == 200
    assert "Session expired. Please log in again." in page.text
    # session no longer grants access
    after = client.get("/search", follow_redirects=False)
    assert after.status_code == 303
    assert after.headers["location"] == "/login"


def test_admin_returns_403_permission_denied(client: TestClient) -> None:
    resp = client.get("/admin")
    assert resp.status_code == 403
    assert "Permission denied" in resp.text


def test_slow_member_page_takes_about_three_seconds(client: TestClient) -> None:
    login(client)
    start = time.monotonic()
    resp = client.get("/member/12345?slow=1")
    elapsed = time.monotonic() - start
    assert resp.status_code == 200
    assert "Member 12345" in resp.text
    assert elapsed >= 2.5
