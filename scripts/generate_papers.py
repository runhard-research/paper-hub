import csv
import re
import sys

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from datetime import datetime

#import feedparser
import requests
import xml.etree.ElementTree as ET
from utils.hash_utils import file_hash
import pandas as pd

# ========= setting =========
CSV_PATH = Path("papers.csv")
HASH_PATH = Path("papers.csv.sha256")
OUTPUT_DIR = Path("papers")
DEFAULT_YEAR = "2025"

HF_PATTERN = re.compile(r"/papers/(\d+\.\d+)")

def load_csv(csv_path):
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print("CSV Columns:", df.columns.tolist())
    return df

def extract_key_ideas(abstract: str, max_items: int = 4) -> list[str]:
    
    #　Extract Key Ideas from the Abstract
    if not abstract:
        return []

    # Split sentences simply
    sentences = re.split(r'(?<=[.!?])\s+', abstract)

    keywords = [
        "propose", "present", "introduce",
        "show", "demonstrate", "find",
        "achieve", "outperform",
        "experiment", "evaluate", "result",
        "method", "approach", "framework",
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

    # Fallback：The first few sentences
    if not ideas:
        ideas = sentences[:max_items]

    return ideas


def make_one_liner(abstract: str) -> str:
    
    #　Condense the first sentence of the abstract into a one-liner
    if not abstract:
        return "（What problem does this paper solve?）"

    # Breaking sentences
    for sep in [". ", "? ", "! "]:
        if sep in abstract:
            return abstract.split(sep)[0] + sep.strip()

    return abstract


def extract_arxiv_id(hf_url: str) -> str:
    match = HF_PATTERN.search(hf_url)
    if not match:
        raise ValueError(f"Invalid HuggingFace Papers URL: {hf_url}")
    return match.group(1)


def fetch_arxiv_metadata(arxiv_id: str) -> tuple[str, str]:

    #　Retrieve the title and abstract from the arXiv API
    url = "http://export.arxiv.org/api/query"
    params = {
        "id_list": arxiv_id,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] arXiv API error for {arxiv_id}: {e}")
        return "Unknown Title", ""

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return "Unknown Title", ""

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        return "Unknown Title", ""

    title = entry.findtext("atom:title", default="", namespaces=ns)
    abstract = entry.findtext("atom:summary", default="", namespaces=ns)

    return (
        title.replace("\n", " ").strip(),
        abstract.replace("\n", " ").strip(),
    )


def generate_markdown(arxiv_id: str, hf_url: str) -> str:
    title, abstract = fetch_arxiv_metadata(arxiv_id)

    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    alphaxiv_url = f"https://www.alphaxiv.org/abs/{arxiv_id}"

    #one_liner = abstract if abstract else "（What problem does this paper solve?）"
    one_liner = make_one_liner(abstract)
    key_ideas = extract_key_ideas(abstract)
    
    key_ideas_md = "\n".join(f"- {idea}" for idea in key_ideas) or "- "

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
    csv_path = "papers.csv"
    df = load_csv(csv_path)

    if "task" not in df.columns:
        raise ValueError(
            "CSV must contain 'task' column but found: " + ", ".join(df.columns)
        )
    if not CSV_PATH.exists():
        print("papers.csv not found")
        sys.exit(1)

    # ---- Check for CSV file update ----
    current_hash = file_hash(CSV_PATH)

    if HASH_PATH.exists():
        old_hash = HASH_PATH.read_text().strip()
        if current_hash == old_hash:
            print("CSV not changed. Skip markdown generation.")
            return

    print("CSV updated. Generating markdown files...")

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "task" not in reader.fieldnames:
            raise ValueError("CSV must contain 'task' column")

        for row in reader:
            hf_url = row["task"].strip()
            if not hf_url:
                continue

            arxiv_id = extract_arxiv_id(hf_url)

            year_dir = OUTPUT_DIR / DEFAULT_YEAR
            year_dir.mkdir(parents=True, exist_ok=True)

            md_path = year_dir / f"{arxiv_id}.md"

            # Leave existing content as is (assuming manual editing will be done later)
            if md_path.exists():
                continue

            content = generate_markdown(arxiv_id, hf_url)
            md_path.write_text(content, encoding="utf-8")
            print(f"Created: {md_path}")

    HASH_PATH.write_text(current_hash)
    print("Hash updated.")


if __name__ == "__main__":
    main()

