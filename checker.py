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
    "GPT-4.1",
    "GPT-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "gpt-5",
    "gpt-5-codex",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-oss-120b",
    "mog-5",
    # Claude
    "claude-3.7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-opus-4-5-20251101",
    "claude-opus-4-6",
    "claude-opus-4-7-stable",
    # Gemini
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
    # Kimi
    "kimi-k2",
    "kimi-k2-5-260127",
    "kimi-k2-thinking",
    # MiMo
    "mimo-v2-flash",
    "mimo-v2-pro",
    # DeepSeek
    "deepseek-v3",
    "deepseek-v3-2-251201",
    "deepseek-reasoner",
    "deepseek-v3.1",
    "deepseek-v3.2-speciale",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    # Qwen
    "qwen-flash",
    "qwen-turbo",
    "qwen-long",
    "qwen-plus",
    "qwen3.5-flash",
    "qwen3.5-plus",
    "qwen3.5-35b-a3b",
    "qwen3.5-397b-a17b",
    "qwen3-next",
    "qwen3-32b",
    "qwen3-max",
    "qwen3-235b",
    "qwen3-coder",
    "qwen3-30b-a3b-instruct-2507",
    # 智谱
    "glm-4.6",
    "glm-4.7",
    "glm-5",
    "glm-5.1",
    # MiniMax
    "minimax-m2.5",
    "minimax-m2.5-highspeed",
    "minimax-m2.7",
    # 腾讯混元
    "hunyuan-vision-1.5-instruct",
    "hunyuan-t1-vision-20250916",
    # 豆包
    "doubao-seed-2-0-lite",
]


ERROR_HINTS = {
    400: "请求格式错误，模型可能不支持该参数",
    401: "API Key 无效或已过期",
    403: "无权限访问该模型，可能未购买或已禁用",
    404: "模型名称不存在，请确认模型ID是否正确",
    429: "请求过于频繁，触发限流",
    500: "平台服务器内部错误，上游服务异常",
    502: "网关错误，上游服务无响应",
    503: "服务暂时不可用，上游服务过载或维护中",
    504: "网关超时，上游响应太慢",
}


def _do_request(model: str) -> requests.Response:
    if "codex" in model or model == "mog-5":
        resp = requests.post(
            "https://aiplatform.zjsk.cc/v1/responses",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "sec-ch-ua-platform": '"macOS"',
                "accept": "text/event-stream",
            },
            json={"model": model, "input": "Reply with OK only.", "stream": True},
            timeout=60,
        )
        if resp.status_code == 404 and "Invalid URL (POST /api/v1/responses)" in (resp.text or ""):
            return requests.post(
                f"{API_BASE_URL.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "Reply with OK only."}], "max_tokens": 16},
                timeout=60,
            )
        return resp
    if "gemini" in model:
        return requests.post(
            f"https://aiplatform.zjsk.cc/v1beta/models/{model}:generateContent?alt=sse",
            headers={"Authorization": API_KEY, "Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": [{"text": "Reply with OK only."}]}],
                "generationConfig": {
                    "candidateCount": 1,
                    "temperature": 0.7,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 2048,
                },
            },
            timeout=60,
        )
    body: dict = {"model": model, "messages": [{"role": "user", "content": "Reply with OK only."}], "max_tokens": 16}
    if "hunyuan" in model:
        body["seed"] = 1
    if "qwen3" in model or "qwq" in model:
        body["enable_thinking"] = False
    if "qwq" in model or model == "qwen3-32b":
        body["stream"] = True
    return requests.post(
        f"{API_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )


def check_model(model: str) -> tuple[bool, float, str]:
    start = time.time()
    try:
        resp = _do_request(model)
        if resp.status_code == 429:
            time.sleep(5)
            resp = _do_request(model)
            elapsed = time.time() - start
            if resp.status_code == 200:
                return True, elapsed, ""
            body_preview = (resp.text or "").replace("\n", " ")[:300]
            if body_preview:
                print(f"[DEBUG] {model} retry response body: {body_preview}")
            hint = ERROR_HINTS.get(resp.status_code, "Unknown error")
            req_id = resp.headers.get("x-request-id", "")
            ctype = resp.headers.get("content-type", "")
            detail = f"; body={body_preview}" if body_preview else f"; body=<empty>; content-type={ctype}; request-id={req_id}"
            return False, elapsed, f"HTTP {resp.status_code} - {hint} (retried){detail}"
        elapsed = time.time() - start
        if resp.status_code == 200:
            return True, elapsed, ""
        body_preview = (resp.text or "").replace("\n", " ")[:300]
        if body_preview:
            print(f"[DEBUG] {model} response body: {body_preview}")
        hint = ERROR_HINTS.get(resp.status_code, "Unknown error")
        req_id = resp.headers.get("x-request-id", "")
        ctype = resp.headers.get("content-type", "")
        detail = f"; body={body_preview}" if body_preview else f"; body=<empty>; content-type={ctype}; request-id={req_id}"
        return False, elapsed, f"HTTP {resp.status_code} - {hint}{detail}"
    except requests.exceptions.Timeout:
        return False, time.time() - start, "Request timeout (>60s) - model too slow or service unavailable"
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
        note = note.replace("|", "\\|").replace("\n", " ")
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
