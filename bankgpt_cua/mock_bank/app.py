"""Mock bank back-office app for the BankGPT take-home (computer-use automation target).

Run: python -m bankgpt_cua.mock_bank.app
Port comes from the MOCK_PORT env var (default 8765).

Session model: an in-memory token store with a `mockbank_session` cookie.
There is no TTL on sessions; the /trigger/timeout endpoint is the supported
way to inject a session-expiry event during automation runs.
"""

from __future__ import annotations

import os
import re
import secrets
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

SESSION_COOKIE = "mockbank_session"

MEMBERS = {
    "12345": {"name": "Ava Carter", "savings": "$4,321.09"},
    "23456": {"name": "Liam Brooks", "savings": "$987.65"},
}

MEMBER_ID_RE = re.compile(r"^\d{5}$")

# token -> username. No TTL; /trigger/timeout is the expiry injector.
_sessions: dict[str, str] = {}


def _current_user(request: Request) -> Optional[str]:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        return _sessions.get(token)
    return None


def _require_login(request: Request) -> Optional[Response]:
    if _current_user(request) is None:
        return RedirectResponse(url="/login", status_code=303)
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="MockBank")
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.get("/login", response_class=HTMLResponse)
    def login_get(request: Request, notice: Optional[str] = None) -> HTMLResponse:
        message = (
            "Session expired. Please log in again." if notice == "expired" else None
        )
        return TEMPLATES.TemplateResponse(request, "login.html", {"request": request, "error": None, "message": message},
        )

    @app.post("/login")
    def login_post(
        request: Request,
        username: str = Form(default=""),
        password: str = Form(default=""),
    ) -> Response:
        if not username.strip() or not password:
            return TEMPLATES.TemplateResponse(request, "login.html", {
                    "request": request,
                    "error": "Username and password are required",
                    "message": None,
                },
                status_code=200,
            )
        token = secrets.token_hex(16)
        _sessions[token] = username.strip()
        resp = RedirectResponse(url="/search", status_code=303)
        resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax")
        return resp

    @app.get("/search", response_class=HTMLResponse)
    def search_get(request: Request) -> Response:
        guard = _require_login(request)
        if guard is not None:
            return guard
        return TEMPLATES.TemplateResponse(
                request, "search.html", {"request": request, "error": None, "not_found": None},
        )

    @app.post("/search")
    def search_post(
        request: Request, member_id: str = Form(default="")
    ) -> Response:
        guard = _require_login(request)
        if guard is not None:
            return guard
        cleaned = member_id.strip()
        if not MEMBER_ID_RE.match(cleaned):
            return TEMPLATES.TemplateResponse(
                request, "search.html", {
                    "request": request,
                    "error": "Member ID must be 5 digits",
                    "not_found": None,
                },
                status_code=200,
            )
        if cleaned not in MEMBERS:
            return TEMPLATES.TemplateResponse(
                request, "search.html", {
                    "request": request,
                    "error": None,
                    "not_found": f"No member found for ID {cleaned}",
                },
                status_code=200,
            )
        return RedirectResponse(url=f"/member/{cleaned}", status_code=303)

    @app.get("/member/{member_id}")
    def member_detail(request: Request, member_id: str, slow: Optional[str] = None) -> Response:
        guard = _require_login(request)
        if guard is not None:
            return guard
        if slow == "1":
            time.sleep(3)
        member = MEMBERS.get(member_id)
        if member is None:
            return TEMPLATES.TemplateResponse(
                request, "not_found.html", {
                    "request": request,
                    "not_found": f"No member found for ID {member_id}",
                },
                status_code=200,
            )
        return TEMPLATES.TemplateResponse(
                request, "member.html", {
                "request": request,
                "member_id": member_id,
                "member": member,
            },
        )

    @app.get("/trigger/timeout")
    def trigger_timeout() -> Response:
        resp = RedirectResponse(url="/login?notice=expired", status_code=303)
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    @app.get("/admin", response_class=HTMLResponse)
    def admin(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
                request, "denied.html", {"request": request, "message": "Permission denied"},
            status_code=403,
        )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("MOCK_PORT", "8765"))
    uvicorn.run("bankgpt_cua.mock_bank.app:app", host="127.0.0.1", port=port, reload=False)
