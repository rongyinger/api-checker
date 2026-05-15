import os
import csv
import time
import requests

API_BASE_URL = os.environ.get("API_BASE_URL_B", "https://b.pandatoken.net/v1")
API_KEY      = os.environ.get("API_KEY_B", "sk-IvoVJKFxaEbOUVVRXJZGlemJZriTaHhpIezzZphb2BI7wkPx")

MODELS = [
    # OpenAI
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4-pro",
    "gpt-5.5",
    # Claude
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    # Gemini
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
]

ERROR_HINTS = {
    400: "请求格式错误",
    401: "API Key 无效或已过期",
    403: "无权限访问该模型",
    404: "模型名称不存在",
    429: "请求过于频繁，触发限流",
    500: "平台服务器内部错误",
    502: "网关错误",
    503: "服务暂时不可用",
    504: "网关超时",
}


def check_model(model: str) -> tuple[bool, float, str]:
    start = time.time()
    try:
        resp = requests.post(
            f"{API_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "Reply with OK only."}], "max_tokens": 16},
            timeout=60,
        )
        if resp.status_code == 429:
            time.sleep(5)
            resp = requests.post(
                f"{API_BASE_URL.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "Reply with OK only."}], "max_tokens": 16},
                timeout=60,
            )
        elapsed = time.time() - start
        if resp.status_code == 200:
            return True, elapsed, ""
        body_preview = (resp.text or "").replace("\n", " ")[:300]
        if body_preview:
            print(f"[DEBUG] {model} response body: {body_preview}")
        hint = ERROR_HINTS.get(resp.status_code, "Unknown error")
        return False, elapsed, f"HTTP {resp.status_code} - {hint}; body={body_preview}"
    except requests.exceptions.Timeout:
        return False, time.time() - start, "Request timeout (>60s)"
    except Exception as e:
        return False, time.time() - start, str(e)


def main():
    from datetime import datetime, timezone, timedelta
    now_dt = datetime.now(timezone(timedelta(hours=8)))
    ts = now_dt.strftime("%Y-%m-%d %H:%M")

    csv_path = "docs/data-b.csv"
    os.makedirs("docs", exist_ok=True)
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "model", "latency", "status", "error"])

        for model in MODELS:
            ok, elapsed, err = check_model(model)
            status = "✅" if ok else "❌"
            note = f"{elapsed:.1f}s" if ok else f"{elapsed:.1f}s ({err})"
            print(f"{status} {model} {note}")
            short_err = (err or "").replace("\n", " ")[:200]
            writer.writerow([ts, model, round(elapsed, 3) if ok else -1, "ok" if ok else "fail", short_err])

    print(f"Data appended to {csv_path}.")


if __name__ == "__main__":
    main()
