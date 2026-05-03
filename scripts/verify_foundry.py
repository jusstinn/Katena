"""Foundry connectivity smoke test.

Reads FOUNDRY_URL and FOUNDRY_TOKEN from environment (or a .env file in
the project root) and makes a single authenticated request to the
Foundry platform to confirm credentials work end-to-end.

Run this as soon as you have a token from Developer Console — it's the
fastest way to verify auth before generating any OSDK clients.

Usage:
    python scripts/verify_foundry.py
"""

import os
import sys
from pathlib import Path

import requests


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    url = os.getenv("FOUNDRY_URL", "").rstrip("/")
    token = os.getenv("FOUNDRY_TOKEN", "")

    if not url:
        print("FAIL: FOUNDRY_URL is not set. Copy .env.example to .env and fill it in.")
        return 1
    if not token or token.startswith("eyJhbGciOi...replace"):
        print("FAIL: FOUNDRY_TOKEN is not set (or still the placeholder).")
        return 1

    endpoint = f"{url}/api/v2/ontologies"
    print(f"GET {endpoint}")
    try:
        resp = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
    except requests.RequestException as exc:
        print(f"FAIL: network error — {exc}")
        return 1

    print(f"  status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        ontologies = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(ontologies, list):
            print(f"OK: token works. Visible ontologies: {len(ontologies)}")
            for ont in ontologies[:5]:
                if isinstance(ont, dict):
                    rid = ont.get("rid", "?")
                    name = ont.get("displayName", ont.get("apiName", "?"))
                    print(f"    - {name}  ({rid})")
        else:
            print("OK: token works (response was not a list — printing raw):")
            print(data)
        return 0
    if resp.status_code in (401, 403):
        print("FAIL: token rejected. Regenerate from Developer Console.")
    elif resp.status_code == 404:
        print("FAIL: 404 — FOUNDRY_URL might be wrong or this stack uses a different API path.")
    else:
        print("FAIL: unexpected status. Body:")
        print(resp.text[:500])
    return 1


if __name__ == "__main__":
    sys.exit(main())
