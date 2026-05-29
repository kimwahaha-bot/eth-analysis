#!/usr/bin/env python3
"""Generate and email a 6-month Instagram follower activity report."""

import json
import os
import smtplib
from collections import defaultdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

IG_TARGET: str = os.environ["IG_TARGET"]
GMAIL_ADDRESS: str = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD: str = os.environ["GMAIL_APP_PASSWORD"]

EVENTS_FILE = Path("ig_events.jsonl")
MONTHS = 6


def load_events(since: datetime) -> list[dict]:
    if not EVENTS_FILE.exists():
        return []
    events = []
    for line in EVENTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        ts = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M")
        if ts >= since:
            events.append(record)
    return events


def build_report(events: list[dict], since: datetime, until: datetime) -> str:
    follows = [e for e in events if e["event"] == "follow"]
    unfollows = [e for e in events if e["event"] == "unfollow"]

    # Who followed then unfollowed within the period
    followed_users = {e["username"] for e in follows}
    unfollowed_users = {e["username"] for e in unfollows}
    churned = followed_users & unfollowed_users  # followed then unfollowed

    # Monthly breakdown
    monthly: dict[str, dict[str, int]] = defaultdict(lambda: {"follow": 0, "unfollow": 0})
    for e in events:
        month = e["timestamp"][:7]  # "YYYY-MM"
        monthly[month][e["event"]] += 1

    net = len(follows) - len(unfollows)
    sign = "+" if net >= 0 else ""

    lines = [
        f"📊 IG 粉絲活動報表 @{IG_TARGET}",
        f"報表期間：{since.strftime('%Y-%m-%d')} ～ {until.strftime('%Y-%m-%d')}",
        "=" * 50,
        "",
        "【總覽】",
        f"  新增粉絲：{len(follows)} 人",
        f"  取消追蹤：{len(unfollows)} 人",
        f"  淨變動：{sign}{net} 人",
        f"  先追蹤後取消：{len(churned)} 人",
        "",
    ]

    # Monthly breakdown
    if monthly:
        lines.append("【每月明細】")
        for month in sorted(monthly):
            m = monthly[month]
            net_m = m["follow"] - m["unfollow"]
            sign_m = "+" if net_m >= 0 else ""
            lines.append(
                f"  {month}　新增 +{m['follow']}　取消 -{m['unfollow']}　淨 {sign_m}{net_m}"
            )
        lines.append("")

    # New followers list
    if follows:
        lines.append(f"【新增粉絲 {len(follows)} 人】")
        for e in follows:
            name = e["full_name"]
            label = f"@{e['username']}" + (f" ({name})" if name and name != e["username"] else "")
            lines.append(f"  {e['timestamp']}  {label}")
        lines.append("")

    # Unfollowers list
    if unfollows:
        lines.append(f"【取消追蹤 {len(unfollows)} 人】")
        for e in unfollows:
            name = e["full_name"]
            label = f"@{e['username']}" + (f" ({name})" if name and name != e["username"] else "")
            tag = "（追蹤後取消）" if e["username"] in churned else ""
            lines.append(f"  {e['timestamp']}  {label}{tag}")
        lines.append("")

    if not events:
        lines.append("這段期間沒有粉絲變動紀錄。")

    return "\n".join(lines)


def send_email(subject: str, body: str) -> None:
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = GMAIL_ADDRESS
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)
    print(f"[email sent] {subject}")


def main() -> None:
    until = datetime.now()
    since = until - timedelta(days=MONTHS * 30)

    events = load_events(since)
    print(f"[info] 載入 {len(events)} 筆事件（{since.strftime('%Y-%m-%d')} 起）")

    report = build_report(events, since, until)
    print(report)

    subject = f"IG 半年報表 @{IG_TARGET}（{since.strftime('%Y-%m')} ～ {until.strftime('%Y-%m')}）"
    send_email(subject, report)


if __name__ == "__main__":
    main()
