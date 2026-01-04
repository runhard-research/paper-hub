import os
import re
import json
from pathlib import Path

import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

OPENAI_MODEL = "gpt-4.1-mini"  # コスト・精度バランス良


# -------------------------
# Markdown utilities
# -------------------------

def extract_frontmatter(md_text: str) -> str | None:
    m = re.search(r"^---\n(.*?)\n---", md_text, re.S)
    return m.group(1) if m else None


def has_tags(md_text: str) -> bool:
    for line in md_text.splitlines():
        print("[DEBUG] line:", line)
        if "**Tags**" in line:
            print("[DEBUG] Matched Tags line:", repr(line))
            value = line.split("**Tags**", 1)[1]
            value = value.replace(":", "").strip()
            print("[DEBUG] Parsed value:", repr(value))
            return value != ""
    print("[DEBUG] No Tags line found")
    return False


def extract_notes(md_text: str) -> str:
    m = re.search(r"##\s*Notes\s+(.*?)(\n##|\Z)", md_text, re.S | re.I)
    if not m:
        raise ValueError("Notes section not found")
    return m.group(1).strip()


def insert_tags(md_text: str, tags: list[str]) -> str:
    tag_str = " ".join(tags)

    # **Tags**: の行がある場合 → その行に追記
    if re.search(r"^\-\s+\*\*Tags\*\*:", md_text, re.M):
        return re.sub(
            r"(^\-\s+\*\*Tags\*\*:\s*)$",
            r"\1" + tag_str,
            md_text,
            flags=re.M,
        )

    # 万一 Tags 行がなければ追加（保険）
    return md_text + f"\n- **Tags**: {tag_str}\n"


# -------------------------
# LLM
# -------------------------

import os
import requests

def call_openai_for_tags(text: str) -> list[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract exactly 3 concise research tags. "
                    "Return them as a comma-separated list without explanations."
                ),
            },
            {
                "role": "user",
                "content": text[:4000],
            },
        ],
        "temperature": 0.2,
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    r.raise_for_status()

    content = r.json()["choices"][0]["message"]["content"]
    tags = [f"#{t.strip().lower()}" for t in content.split(",")][:3]
    return tags

def normalize_tags(tags: list[str]) -> list[str]:
    out = []
    for t in tags[:3]:
        t = t.lower().strip()
        t = t.replace(" ", "-")
        if not t.startswith("#"):
            t = "#" + t
        out.append(t)
    return out[:3]


# -------------------------
# Main
# -------------------------

def main():
    print("=== generate_tags.py VERSION 2025-01-CLEAN ===")

    paper_dir = Path("papers")
    print(f"[INFO] Scanning markdowns under: {paper_dir.resolve()}")

    targets = []

    # 1. タグ未設定ファイルを収集
    for md in paper_dir.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        if not has_tags(text):
            targets.append(md)

    if not targets:
        print("No papers need tagging.")
        return

    print(f"[INFO] Tagging {len(targets)} papers...")

    # 2. タグ付与
    for md in targets:
        print(f"- {md.name}")
        text = md.read_text(encoding="utf-8")

        try:
            notes = extract_notes(text)
        except Exception as e:
            print("[WARN] Notes extraction failed:", md.name)
            print("       ", e)
            continue

        try:
            tags = call_openai_for_tags(notes)
        except Exception as e:
            print("[ERROR] Failed tagging:", md.name)
            print("        ", e)
            continue

        new_text = insert_tags(text, tags)
        md.write_text(new_text, encoding="utf-8")

        print(f"  tags: {tags}")



import traceback

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("========== FATAL ERROR ==========")
        traceback.print_exc()
        print("=================================")
        raise

#if __name__ == "__main__":
    #main()
