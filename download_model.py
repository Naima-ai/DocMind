"""
download_model.py — run this ONCE, from a normal terminal, before using app.py.

It downloads the embedding model to a plain local folder (./models/all-MiniLM-L6-v2).
app.py then loads the model from that folder PATH, not by repo name — a local
folder load makes zero network calls, so it can never hit a HuggingFace rate
limit (429) or outage again, no matter how many times the app restarts.

If you hit a 429 "Too Many Requests" here, it means HuggingFace is throttling
your IP right now (common on shared/cloud IPs). This script retries with
backoff automatically. If it still fails after all retries, wait ~15-60
minutes and run it again, or set an HF token (see bottom of this file) to
get a much higher rate limit.

Usage:
    python download_model.py
"""

import os
import sys
import time

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "all-MiniLM-L6-v2")

MAX_RETRIES = 6
BASE_DELAY = 10  # seconds; doubles each retry (10, 20, 40, 80, 160, 320)


def download_with_retry():
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    os.makedirs(LOCAL_DIR, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Attempt {attempt}/{MAX_RETRIES}: downloading {MODEL_NAME} ...")
            snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=LOCAL_DIR,
                # Uses HF_TOKEN env var automatically if you've set one —
                # authenticated requests get a much higher rate limit than
                # anonymous ones, which is usually what fixes repeated 429s.
            )
            print(f"\nDone. Model saved to: {LOCAL_DIR}")
            print("app.py will now load it from this local path with zero network calls.")
            return
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429 and attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** (attempt - 1))
                print(f"Rate limited (429). Waiting {delay}s before retrying...")
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(
        f"Still rate-limited after {MAX_RETRIES} attempts. Wait 15-60 minutes "
        "and re-run this script, or set an HF_TOKEN env var (see below) for a "
        "much higher rate limit."
    )


if __name__ == "__main__":
    try:
        download_with_retry()
    except Exception as e:
        print(f"\nFAILED: {e}")
        print(
            "\nTo raise your rate limit: create a free token at "
            "https://huggingface.co/settings/tokens, then run:\n"
            "    export HF_TOKEN=hf_xxxxxxxxxxxx   (Linux/Mac)\n"
            "    set HF_TOKEN=hf_xxxxxxxxxxxx      (Windows cmd)\n"
            "and re-run this script."
        )
        sys.exit(1)
