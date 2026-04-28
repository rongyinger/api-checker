import os
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
DINGTALK_WEBHOOK = os.environ["DINGTALK_WEBHOOK"]
DINGTALK_SECRET = os.environ["DINGTALK_SECRET"]

MODELS = [
    # OpenAI
    "GPT-4.1", "GPT-4o", "gpt-4o-mini-2024-07-18", "gpt-5",
    # Claude
    "claude-3.7-sonnet-20250219", "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001", "claude-opus-4-20250514",
    # Gemini
    "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro",
    # Kimi
    "kimi-k2", "kimi-k2-thinking",
    # MiMo
    "mimo-v2-flash", "mimo-v2-pro",
    # DeepSeek
    "deepseek-v3", "deepseek-reasoner", "deepseek-v3.1",
    # Qwen
    "qwen-turbo", "qwen-plus", "qwen3-32b", "qwen3-max", "qwq-32b",
    # 智谱
    "glm-4.6", "glm-5",
    # MiniMax
    "minimax-m2.5",
    # 豆包
    "doubao-seed-2-0-lite",
]


def check_model(model: str) -> tuple[bool, float, str]:
    start = time.time()
    try:
        resp = requests.post(
            f"{API_BASE_URL.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "Reply with OK only."}], "max_tokens": 16},
            timeout=30,
        )
        elapsed = time.time() - start
        if resp.status_code == 200:
            return True, elapsed, ""
        return False, elapsed, f"HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return False, time.time() - start, "Timeout"
    except Exception as e:
        return False, time.time() - start, str(e)


def dingtalk_sign() -> tuple[str, str]:
    timestamp = str(int(time.time() * 1000))
    msg = f"{timestamp}\n{DINGTALK_SECRET}"
    sign = base64.b64encode(
        hmac.new(DINGTALK_SECRET.encode(), msg.encode(), digestmod=hashlib.sha256).digest()
    ).decode()
    return timestamp, urllib.parse.quote_plus(sign)


def send_dingtalk(text: str, at_all: bool) -> None:
    timestamp, sign = dingtalk_sign()
    url = f"{DINGTALK_WEBHOOK}&timestamp={timestamp}&sign={sign}"
    payload = {
        "msgtype": "markdown",
        "markdown": {"title": "API Model Check", "text": text},
        "at": {"isAtAll": at_all},
    }
    requests.post(url, json=payload, timeout=10)


def main():
    rows = []
    any_fail = False

    for model in MODELS:
        ok, elapsed, err = check_model(model)
        status = "✅" if ok else "❌"
        note = f"{elapsed:.1f}s" if ok else f"{elapsed:.1f}s ({err})"
        rows.append((status, model, note))
        if not ok:
            any_fail = True
        print(f"{status} {model} {note}")

    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S CST")

    lines = [
        f"### API Model Check — {now}\n",
        "| 状态 | 模型 | 耗时 |",
        "| --- | --- | --- |",
    ]
    for status, model, note in rows:
        lines.append(f"| {status} | `{model}` | {note} |")

    if any_fail:
        lines.append("\n> ⚠️ 部分模型不可用，请检查！")

    send_dingtalk("\n".join(lines), at_all=any_fail)
    print("Report sent to DingTalk.")


if __name__ == "__main__":
    main()
