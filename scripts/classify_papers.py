#!/usr/bin/env python3
"""Classify Paper Hub Markdown files with NVIDIA NIM and write YAML front matter.

Normal runs classify only Markdown files that do not have ``classification``.
``--force`` replaces every classification; ``--retry-failed`` processes only files
recorded in the local failure list. A failed request never changes a Markdown file.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
FAILURES_PATH = ROOT / ".classification_failures.json"
FIELDS = ("categories", "tasks", "technologies", "applications")
MAX_RETRIES = 5
RETRY_DELAYS_SECONDS = (10, 20, 40, 80, 160)
REQUEST_DELAY_SECONDS = 2


class ClassificationError(Exception):
    """A classification failure that can optionally be retried."""

    def __init__(self, kind: str, message: str, retryable: bool):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


def load_taxonomy() -> tuple[dict[str, set[str]], dict[str, dict[str, int]], dict[str, list[dict[str, str]]]]:
    """Return permitted IDs, validation rules, and model-facing definitions."""
    filenames = {
        "categories": "categories.yaml",
        "tasks": "tasks.yaml",
        "technologies": "technologies.yaml",
        "applications": "applications.yaml",
    }
    allowed: dict[str, set[str]] = {}
    rules: dict[str, dict[str, int]] = {}
    catalog: dict[str, list[dict[str, str]]] = {}
    for field, filename in filenames.items():
        data = yaml.safe_load((ROOT / "taxonomy" / filename).read_text(encoding="utf-8")) or {}
        entries = data.get(field, [])
        allowed[field] = {item["id"] for item in entries}
        catalog[field] = [
            {"id": item["id"], "name": item.get("name", item["id"]), "description": item.get("description", item.get("name", item["id"]))}
            for item in entries
        ]
        if field == "categories":
            rules = data.get("classification_rules", {})
    return allowed, rules, catalog


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n", text, re.DOTALL)
    if not match:
        return {}, text
    metadata = yaml.safe_load(match.group(1)) or {}
    if not isinstance(metadata, dict):
        raise ValueError("YAML front matter must be a mapping")
    return metadata, text[match.end():]


def extract_section(body: str, heading: str) -> str:
    match = re.search(rf"^##\s+{re.escape(heading)}\s*\r?\n(.*?)(?=^##\s|\Z)", body, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def paper_context(path: Path, body: str) -> dict[str, str]:
    title_match = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
    arxiv_match = re.search(r"arxiv\.org/abs/([\w.]+)", body, re.IGNORECASE)
    return {"file": path.as_posix(), "arxiv_id": arxiv_match.group(1) if arxiv_match else path.stem,
            "title": title_match.group(1).strip() if title_match else path.stem,
            "one_liner": extract_section(body, "One-liner"), "key_ideas": extract_section(body, "Key Ideas"),
            "abstract": extract_section(body, "Notes")[:12000]}


def schema_instruction(catalog: dict[str, list[dict[str, str]]]) -> str:
    return f"""You are a precise research-paper classifier. Classify the paper's central contribution, not every topic it mentions.

Classification policy:
- For primary_category, prioritize what the paper introduces, evaluates, or contributes. Do not choose a category merely because it is an object of study.
- Use evaluation_benchmarks as primary_category when the main contribution is a benchmark, dataset, evaluation environment, or measurement methodology.
- Use ai_agents as primary_category when the main contribution is an agent architecture, workflow, or training method.
- Add a technology only when it is explicitly proposed, used, or evaluated in the supplied paper text. Never infer common implementation details.
- Add an application only for a direct intended deployment use case. Do not treat benchmark coverage domains, datasets, or examples as applications.
- Select fewer labels when evidence is weak. Empty tasks, technologies, and applications lists are valid when supported by the paper.
- Use each taxonomy item's description to distinguish similarly named IDs. Use an ID exactly as written below; do not singularize, pluralize, replace underscores with hyphens, or invent IDs.

Use only the following taxonomy definitions (ID, name, and meaning):
{json.dumps(catalog, ensure_ascii=False)}

Return exactly one valid JSON object and nothing else: no Markdown, no code fence, no commentary.
The object must have this shape:
{{"primary_category":"one category ID", "categories":["1-3 category IDs"],
"tasks":["0-5 task IDs"], "technologies":["0-5 technology IDs"],
"applications":["0-5 application IDs"], "tags":["free-form strings"]}}.
primary_category must appear in categories. Do not invent taxonomy IDs.
Tags are free-form descriptive text and are not taxonomy IDs."""


def extract_json_object(content: str) -> str:
    """Strip common wrappers and return the first balanced JSON object."""
    text = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip().replace("\\_", "_")
    start = text.find("{")
    if start < 0:
        raise ClassificationError("json", "No JSON object found in model response", True)
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ClassificationError("json", "JSON object is not balanced", True)


def parse_model_json(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        raise ClassificationError("json", "Model response content is not text", True)
    try:
        parsed = json.loads(extract_json_object(content))
    except json.JSONDecodeError as exc:
        raise ClassificationError("json", f"Invalid JSON from NIM: {exc.msg}", True) from exc
    if not isinstance(parsed, dict):
        raise ClassificationError("json", "Model output is not a JSON object", True)
    return parsed


class NimClient:
    def __init__(self, endpoint: str, model: str, api_key: str):
        self.endpoint, self.model, self.api_key = endpoint.rstrip("/"), model, api_key
        self.session = requests.Session()
        self.last_request_at: float | None = None

    def _wait_between_requests(self) -> None:
        if self.last_request_at is not None:
            remaining = REQUEST_DELAY_SECONDS - (time.monotonic() - self.last_request_at)
            if remaining > 0:
                time.sleep(remaining)

    def complete(self, context: dict[str, str], catalog: dict[str, list[dict[str, str]]], summary: Counter[str]) -> dict[str, Any]:
        base_payload = {"model": self.model, "temperature": 0, "messages": [
            {"role": "system", "content": schema_instruction(catalog)},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]}
        # Prefer JSON mode. A 400 from a deployment that lacks response_format falls
        # back once to the prompt-enforced JSON contract.
        for use_json_mode in (True, False):
            payload = dict(base_payload)
            if use_json_mode:
                payload["response_format"] = {"type": "json_object"}
            self._wait_between_requests()
            try:
                self.last_request_at = time.monotonic()
                response = self.session.post(f"{self.endpoint}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=payload, timeout=120)
            except requests.Timeout as exc:
                raise ClassificationError("timeout", "NIM request timed out", True) from exc
            except requests.ConnectionError as exc:
                raise ClassificationError("connection", "NIM connection error", True) from exc
            except requests.RequestException as exc:
                raise ClassificationError("request", f"NIM request error: {exc}", False) from exc
            if response.status_code == 400 and use_json_mode:
                summary["structured_output_fallback"] += 1
                continue
            if response.status_code in (429, 502, 503):
                raise ClassificationError(str(response.status_code), f"NIM HTTP {response.status_code}", True)
            if response.status_code >= 400:
                raise ClassificationError("http", f"NIM HTTP {response.status_code}: {response.text[:300]}", False)
            try:
                content = response.json()["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise ClassificationError("json", "NIM response did not contain chat completion content", True) from exc
            return parse_model_json(content)
        raise ClassificationError("http", "NIM rejected JSON mode and the plain fallback", False)


def validate_result(result: dict[str, Any], allowed: dict[str, set[str]], rules: dict[str, dict[str, int]]) -> list[str]:
    errors: list[str] = []
    primary = result.get("primary_category")
    if primary not in allowed["categories"]:
        errors.append("primary_category is missing or not an allowed category ID")
    for field in FIELDS:
        values = result.get(field, [])
        rule = rules.get(field, {})
        if not isinstance(values, list):
            errors.append(f"{field} must be a list")
            continue
        if len(values) < rule.get("min", 0) or len(values) > rule.get("max", 999):
            errors.append(f"{field} must contain {rule.get('min', 0)}..{rule.get('max', 999)} values")
        if not all(isinstance(value, str) for value in values):
            errors.append(f"{field} must contain only string IDs")
            continue
        invalid = set(values) - allowed[field]
        if invalid:
            errors.append(f"{field} has unknown IDs: {', '.join(sorted(invalid))}")
    if primary and primary not in result.get("categories", []):
        errors.append("primary_category must be included in categories")
    tags = result.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        errors.append("tags must be a list of strings")
    return errors


def classify_with_retries(context: dict[str, str], client: NimClient, allowed: dict[str, set[str]], rules: dict[str, dict[str, int]], catalog: dict[str, list[dict[str, str]]], summary: Counter[str]) -> dict[str, Any]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = client.complete(context, catalog, summary)
            errors = validate_result(result, allowed, rules)
            if errors:
                raise ClassificationError("validation", "; ".join(errors), True)
            return result
        except ClassificationError as exc:
            summary[f"{exc.kind}_failures"] += 1
            if not exc.retryable or attempt == MAX_RETRIES:
                raise
            delay = RETRY_DELAYS_SECONDS[attempt]
            summary[f"{exc.kind}_retries"] += 1
            print(f"RETRY {attempt + 1}/{MAX_RETRIES} {context['file']} ({exc.kind}): waiting {delay}s", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


def normalized_classification(result: dict[str, Any], model: str) -> dict[str, Any]:
    return {"taxonomy_version": "1.0", "primary_category": result["primary_category"], "categories": result["categories"], "tasks": result["tasks"], "technologies": result["technologies"], "applications": result["applications"], "tags": result.get("tags", []), "classifier": {"provider": "nvidia_nim", "model": model, "classified_at": date.today().isoformat()}}


def write_classification(path: Path, metadata: dict[str, Any], body: str, classification: dict[str, Any]) -> None:
    metadata["classification"] = classification
    front_matter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False, default_flow_style=False).strip()
    path.write_text(f"---\n{front_matter}\n---\n{body}", encoding="utf-8")


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_failures() -> dict[str, dict[str, str]]:
    if not FAILURES_PATH.exists():
        return {}
    try:
        data = json.loads(FAILURES_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        print(f"WARN ignoring unreadable failure list: {FAILURES_PATH}", file=sys.stderr)
        return {}


def save_failures(failures: dict[str, dict[str, str]]) -> None:
    if failures:
        FAILURES_PATH.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif FAILURES_PATH.exists():
        FAILURES_PATH.unlink()


def collect_files(inputs: list[Path]) -> list[Path]:
    return sorted({file for item in inputs for file in (item.rglob("*.md") if item.is_dir() else [item]) if file.is_file() and file.suffix.lower() == ".md"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Markdown files or directories (default: papers)")
    parser.add_argument("--dry-run", action="store_true", help="List eligible papers only; never calls NIM or writes files")
    parser.add_argument("--limit", type=int, help="Maximum number of eligible papers")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="Replace classifications, including successful ones")
    mode.add_argument("--retry-failed", action="store_true", help="Process only paths in the local failure list")
    parser.add_argument("--model", default=os.getenv("NVIDIA_NIM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--endpoint", default=os.getenv("NVIDIA_NIM_ENDPOINT", DEFAULT_ENDPOINT))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    inputs = args.paths or [ROOT / "papers"]
    failures = load_failures()
    files = collect_files(inputs)
    if args.retry_failed:
        files = [path for path in files if relative_path(path) in failures]
    allowed, rules, catalog = load_taxonomy()
    eligible: list[tuple[Path, dict[str, Any], str]] = []
    for path in files:
        metadata, body = split_front_matter(path.read_text(encoding="utf-8"))
        if metadata.get("classification") and not args.force:
            continue
        eligible.append((path, metadata, body))
        if args.limit and len(eligible) >= args.limit:
            break
    print(f"Eligible papers: {len(eligible)}")
    summary: Counter[str] = Counter()
    if args.dry_run:
        for path, _, body in eligible:
            print(f"DRY-RUN would classify: {path} ({paper_context(path, body)['title']})")
        return 0
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key and eligible:
        print("ERROR NVIDIA_API_KEY is not set (it is never read from files).", file=sys.stderr)
        return 2
    client = NimClient(args.endpoint, args.model, api_key)
    for path, metadata, body in eligible:
        key = relative_path(path)
        try:
            result = classify_with_retries(paper_context(path, body), client, allowed, rules, catalog, summary)
            write_classification(path, metadata, body, normalized_classification(result, args.model))
            failures.pop(key, None)
            summary["succeeded"] += 1
            print(f"CLASSIFIED: {path}")
        except ClassificationError as exc:
            failures[key] = {"kind": exc.kind, "message": str(exc), "updated_at": date.today().isoformat()}
            summary["failed"] += 1
            print(f"ERROR {path} ({exc.kind}): {exc}", file=sys.stderr)
    save_failures(failures)
    print("Summary:")
    print(f"  succeeded: {summary['succeeded']}")
    print(f"  failed: {summary['failed']}")
    for key in sorted(name for name in summary if name not in {"succeeded", "failed"}):
        print(f"  {key}: {summary[key]}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
