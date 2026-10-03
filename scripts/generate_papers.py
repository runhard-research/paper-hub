import csv
import re
import sys
import time
from pathlib import Path
from datetime import datetime

import requests
import xml.etree.ElementTree as ET
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from utils.hash_utils import file_hash


# ========= setting =========
CSV_PATH = Path("papers.csv")
HASH_PATH = Path("papers.csv.sha256")
OUTPUT_DIR = Path("papers")
DEFAULT_YEAR = "2025"

HF_PATTERN = re.compile(r"/papers/(\d+\.\d+)")

# arXiv API retry settings
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 10


def load_csv(csv_path):
    print(f"Loading CSV from: {csv_path}")

    df = pd.read_csv(csv_path)

    print("CSV Columns:", df.columns.tolist())

    return df


def extract_key_ideas(abstract: str, max_items: int = 4) -> list[str]:
    if not abstract:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', abstract)

    keywords = [
        "propose",
        "present",
        "introduce",
        "show",
        "demonstrate",
        "find",
        "achieve",
        "outperform",
        "experiment",
        "evaluate",
        "result",
        "method",
        "approach",
        "framework",
    ]

    ideas = []

    for sent in sentences:
        s = sent.strip()

        if not s:
            continue

        lower = s.lower()

        if any(k in lower for k in keywords):
            ideas.append(s)

        if len(ideas) >= max_items:
            break

    if not ideas:
        ideas = sentences[:max_items]

    return ideas


def make_one_liner(abstract: str) -> str:
    if not abstract:
        return "（What problem does this paper solve?）"

    for sep in [". ", "? ", "! "]:
        if sep in abstract:
            return abstract.split(sep)[0] + sep.strip()

    return abstract


def extract_arxiv_id(hf_url: str) -> str:
    match = HF_PATTERN.search(hf_url)

    if not match:
        raise ValueError(
            f"Invalid HuggingFace Papers URL: {hf_url}"
        )

    return match.group(1)


def fetch_arxiv_metadata(arxiv_id: str) -> tuple[str, str] | None:
    """
    Fetch title and abstract from arXiv API.

    Returns:
        (title, abstract) on success
        None on failure

    Important:
        If arXiv API fails, this function does NOT return
        "Unknown Title". This prevents creation of incomplete MD files.
    """

    url = "https://export.arxiv.org/api/query"

    params = {
        "id_list": arxiv_id
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            print(
                f"[arXiv] Fetching {arxiv_id} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            resp = requests.get(
                url,
                params=params,
                timeout=20,
            )

            resp.raise_for_status()

        except requests.RequestException as e:

            print(
                f"[WARN] arXiv API error for {arxiv_id}: {e}"
            )

            if attempt < MAX_RETRIES:

                wait_seconds = RETRY_WAIT_SECONDS * attempt

                print(
                    f"[arXiv] Retrying in "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)

                continue

            print(
                f"[ERROR] Failed to retrieve arXiv metadata "
                f"for {arxiv_id} after {MAX_RETRIES} attempts."
            )

            return None

        try:
            root = ET.fromstring(resp.text)

        except ET.ParseError as e:

            print(
                f"[WARN] Failed to parse arXiv response "
                f"for {arxiv_id}: {e}"
            )

            return None

        ns = {
            "atom": "http://www.w3.org/2005/Atom"
        }

        entry = root.find(
            "atom:entry",
            ns
        )

        if entry is None:

            print(
                f"[WARN] No arXiv entry found for {arxiv_id}"
            )

            return None

        title = entry.findtext(
            "atom:title",
            default="",
            namespaces=ns,
        )

        abstract = entry.findtext(
            "atom:summary",
            default="",
            namespaces=ns,
        )

        title = title.replace("\n", " ").strip()
        abstract = abstract.replace("\n", " ").strip()

        # Do not create an incomplete Markdown file.
        if not title:

            print(
                f"[WARN] Empty title returned for {arxiv_id}"
            )

            return None

        if not abstract:

            print(
                f"[WARN] Empty abstract returned for {arxiv_id}"
            )

            return None

        print(
            f"[arXiv] Successfully retrieved metadata "
            f"for {arxiv_id}"
        )

        return title, abstract

    return None


def is_incomplete_markdown(md_path: Path) -> bool:
    """
    Detect Markdown files that were created by a previous
    failed arXiv API request.

    Current known failure pattern:
        # Unknown Title

    Also checks for an empty Notes section.
    """

    if not md_path.exists():
        return False

    try:
        content = md_path.read_text(
            encoding="utf-8"
        )
    except OSError as e:
        print(
            f"[WARN] Could not read {md_path}: {e}"
        )
        return False

    # Previous version created this when arXiv API failed.
    if content.startswith("# Unknown Title"):
        return True

    # Detect empty Notes section.
    notes_match = re.search(
        r"## Notes\s*\n(.*?)\n\s*---",
        content,
        flags=re.DOTALL,
    )

    if notes_match:
        notes = notes_match.group(1).strip()

        if not notes:
            return True

    return False


def generate_markdown(
    arxiv_id: str,
    hf_url: str,
) -> str | None:

    metadata = fetch_arxiv_metadata(arxiv_id)

    # IMPORTANT:
    # Do not create an MD if arXiv metadata could not be retrieved.
    if metadata is None:

        print(
            f"[SKIP] Markdown not created for "
            f"{arxiv_id} because arXiv metadata "
            f"could not be retrieved."
        )

        return None

    title, abstract = metadata

    arxiv_url = (
        f"https://arxiv.org/abs/{arxiv_id}"
    )

    pdf_url = (
        f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    )

    alphaxiv_url = (
        f"https://www.alphaxiv.org/abs/{arxiv_id}"
    )

    one_liner = make_one_liner(
        abstract
    )

    key_ideas = extract_key_ideas(
        abstract
    )

    key_ideas_md = "\n".join(
        f"- {idea}"
        for idea in key_ideas
    )

    if not key_ideas_md:
        key_ideas_md = "- "

    return f"""# {title}

- **arXiv**: {arxiv_url}
- **alphaXiv**: {alphaxiv_url}
- **PDF**: {pdf_url}
- **HuggingFace Papers**: {hf_url}
- **Tags**:
- **Added**: {datetime.now().strftime("%Y-%m-%d")}

---

## One-liner
{one_liner}

---

## Why I care
- Why I read this paper

---

## Key Ideas
{key_ideas_md}

---

## Notes
{abstract}

---

## alphaXiv discussion memo
- Comments that caught my attention
- A question I have
"""


def main():

    csv_path = CSV_PATH

    if not csv_path.exists():

        print(
            f"[ERROR] {csv_path} not found"
        )

        sys.exit(1)

    df = load_csv(csv_path)

    if "task" not in df.columns:

        raise ValueError(
            "CSV must contain 'task' column but found: "
            + ", ".join(df.columns)
        )

    current_hash = file_hash(
        CSV_PATH
    )

    old_hash = None

    if HASH_PATH.exists():

        old_hash = HASH_PATH.read_text(
            encoding="utf-8"
        ).strip()

    csv_changed = (
        current_hash != old_hash
    )

    if csv_changed:

        print(
            "CSV changed. Checking papers..."
        )

    else:

        print(
            "CSV not changed. "
            "Checking for missing/incomplete Markdown files..."
        )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # We intentionally DO NOT return here when CSV has not changed.
    #
    # This allows missing or incomplete MD files to be recovered.
    #
    # Existing complete MD files are still skipped below, so this does
    # NOT cause arXiv API calls for every paper on every Action run.
    # ------------------------------------------------------------------

    created_count = 0
    skipped_count = 0
    retry_count = 0

    with CSV_PATH.open(
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        if "task" not in reader.fieldnames:

            raise ValueError(
                "CSV must contain 'task' column"
            )

        for row in reader:

            hf_url = row["task"].strip()

            if not hf_url:
                continue

            try:

                arxiv_id = extract_arxiv_id(
                    hf_url
                )

            except ValueError as e:

                print(
                    f"[WARN] {e}"
                )

                continue

            year_dir = (
                OUTPUT_DIR / DEFAULT_YEAR
            )

            year_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            md_path = (
                year_dir / f"{arxiv_id}.md"
            )

            # ----------------------------------------------------------
            # Existing Markdown handling
            # ----------------------------------------------------------

            if md_path.exists():

                if not is_incomplete_markdown(
                    md_path
                ):

                    # Existing complete MD:
                    # Leave manual edits untouched.
                    print(
                        f"[SKIP] Existing Markdown: "
                        f"{md_path}"
                    )

                    skipped_count += 1

                    continue

                # Existing incomplete MD:
                # Remove it and regenerate from arXiv.
                print(
                    f"[RETRY] Incomplete Markdown detected: "
                    f"{md_path}"
                )

                try:

                    md_path.unlink()

                    print(
                        f"[RETRY] Removed incomplete file: "
                        f"{md_path}"
                    )

                except OSError as e:

                    print(
                        f"[ERROR] Could not remove "
                        f"{md_path}: {e}"
                    )

                    continue

                retry_count += 1

            # ----------------------------------------------------------
            # Generate Markdown
            # ----------------------------------------------------------

            content = generate_markdown(
                arxiv_id,
                hf_url,
            )

            # ----------------------------------------------------------
            # IMPORTANT:
            #
            # If arXiv API failed, content is None.
            # In that case NO MD file is created.
            #
            # The next scheduled Action will retry it.
            # ----------------------------------------------------------

            if content is None:

                print(
                    f"[SKIP] No Markdown created for "
                    f"{arxiv_id}"
                )

                continue

            md_path.write_text(
                content,
                encoding="utf-8",
            )

            print(
                f"Created: {md_path}"
            )

            created_count += 1

    # ------------------------------------------------------------------
    # Update hash
    # ------------------------------------------------------------------

    HASH_PATH.write_text(
        current_hash,
        encoding="utf-8",
    )

    print()
    print("========================================")
    print("Markdown generation completed.")
    print(f"Created : {created_count}")
    print(f"Skipped : {skipped_count}")
    print(f"Retried : {retry_count}")
    print("Hash updated.")
    print("========================================")


if __name__ == "__main__":
    main()