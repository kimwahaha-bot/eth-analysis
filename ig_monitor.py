#!/usr/bin/env python3
"""Instagram follower monitor — tracks who follows/unfollows and sends email."""

import json
import os
import smtplib
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import browser_cookie3
import requests
from dotenv import load_dotenv

load_dotenv()

IG_TARGET: str = os.environ["IG_TARGET"]
GMAIL_ADDRESS: str = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD: str = os.environ["GMAIL_APP_PASSWORD"]
CHECK_INTERVAL: int = int(os.getenv("CHECK_INTERVAL_SECONDS", "900"))

STATE_FILE = Path("ig_monitor_state.json")
EVENTS_FILE = Path("ig_events.jsonl")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "X-IG-App-ID": "936619743392459",
}


def make_session() -> requests.Session:
    session = requests.Session()
    session.cookies = browser_cookie3.brave(domain_name=".instagram.com")
    session.headers.update(HEADERS)
    return session


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def log_event(event: str, username: str, full_name: str, timestamp: str) -> None:
    record = json.dumps(
        {"timestamp": timestamp, "event": event, "username": username, "full_name": full_name},
        ensure_ascii=False,
    )
    with EVENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(record + "\n")


def send_email(subject: str, body: str) -> None:
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = GMAIL_ADDRESS
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        print(f"[email sent] {subject}")
    except Exception as err:
        print(f"[warn] email failed: {err}")


def get_user_id(session: requests.Session, username: str) -> Optional[str]:
    try:
        resp = session.get(
            f"https://www.instagram.com/api/v1/users/web_profile_info/?username={username}",
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["data"]["user"]["id"]
    except Exception as err:
        print(f"[error] failed to get user_id: {err}")
        return None


def get_followers(session: requests.Session, user_id: str) -> Optional[dict[str, str]]:
    """Returns {username: full_name} for all followers."""
    followers: dict[str, str] = {}
    max_id: Optional[str] = None

    while True:
        params: dict = {"count": 200}
        if max_id:
            params["max_id"] = max_id

        try:
            resp = session.get(
                f"https://www.instagram.com/api/v1/friendships/{user_id}/followers/",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as err:
            print(f"[error] failed to fetch followers page: {err}")
            return None

        for user in data.get("users", []):
            followers[user["username"]] = user.get("full_name") or user["username"]

        max_id = data.get("next_max_id")
        if not max_id:
            break

        time.sleep(1)  # avoid rate limiting between pages

    return followers


def format_user(username: str, full_name: str) -> str:
    if full_name and full_name != username:
        return f"@{username} ({full_name})"
    return f"@{username}"


def run_check(session: requests.Session, user_id: str) -> None:
    state = load_state()
    previous: dict[str, str] = state.get("followers", {})

    current = get_followers(session, user_id)
    if current is None:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{now}] @{IG_TARGET} followers: {len(current):,}")

    if previous:
        prev_set = set(previous)
        curr_set = set(current)

        gained = curr_set - prev_set
        lost = prev_set - curr_set

        if gained or lost:
            for u in gained:
                log_event("follow", u, current[u], now)
            for u in lost:
                log_event("unfollow", u, previous[u], now)

            lines = [f"帳號：@{IG_TARGET}\n時間：{now}\n"]

            if gained:
                lines.append(f"新增粉絲 (+{len(gained)})：")
                for u in sorted(gained):
                    lines.append(f"  • {format_user(u, current[u])}")

            if lost:
                lines.append(f"\n取消追蹤 (-{len(lost)})：")
                for u in sorted(lost):
                    lines.append(f"  • {format_user(u, previous[u])}")

            parts = []
            if gained:
                parts.append(f"+{len(gained)}")
            if lost:
                parts.append(f"-{len(lost)}")

            subject = f"IG 粉絲變動 @{IG_TARGET} ({', '.join(parts)})"
            send_email(subject, "\n".join(lines))

    state["followers"] = current
    state["last_checked"] = now
    save_state(state)


def main() -> None:
    session = make_session()

    test = session.get("https://www.instagram.com/accounts/edit/", allow_redirects=False)
    if test.status_code != 200:
        print("[error] Brave 的 IG session 已過期，請在 Brave 重新登入 Instagram 後再執行")
        sys.exit(1)

    user_id = get_user_id(session, IG_TARGET)
    if not user_id:
        sys.exit(1)

    print(f"[info] 監控 @{IG_TARGET} (id={user_id})，每 {CHECK_INTERVAL}s 檢查一次")

    while True:
        try:
            run_check(session, user_id)
        except KeyboardInterrupt:
            print("\n[info] stopped by user")
            sys.exit(0)
        except Exception as err:
            print(f"[error] unexpected error: {err}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
