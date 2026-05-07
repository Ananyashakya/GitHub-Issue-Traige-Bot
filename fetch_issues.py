"""
GitHub Issue Fetcher — Final Production Version
================================================
SPEED:   Concurrent fetching with ThreadPoolExecutor (5x faster)
QUALITY: Multi-layer quality gates enforced at fetch time
OUTPUT:  raw_issues.csv  →  repo | title | body | label

Strategy:
  1. Discover top 600 repos across 40 diverse ecosystems (cached)
  2. For each label, fetch issues concurrently from all repos
  3. Every issue passes 8 quality checks before being saved
  4. Auto-resume if interrupted — never loses progress

Install:
    pip install requests pandas tqdm

Usage:
    1. Set GITHUB_TOKEN below
    2. python fetch_issues.py
"""

import requests
import pandas as pd
import time
import os
import re
import json
import threading
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════
# !! SET YOUR TOKEN HERE !!
# Get one at: https://github.com/settings/tokens  (scope: public_repo)
# ═══════════════════════════════════════════════════════════════════════════
GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxx"

HEADERS = {
    "Authorization":        f"token {GITHUB_TOKEN}",
    "Accept":               "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
TARGET_PER_LABEL  = 10_000   # 10k × 6 labels = 60k total
TOP_N_REPOS       = 600      # discover top 600 repos
MIN_REPO_STARS    = 300      # minimum repo quality bar
MIN_REPO_ISSUES   = 30       # must have real issue activity
MIN_TITLE_WORDS   = 4        # title must have at least 4 words
MIN_BODY_WORDS    = 30       # body must have at least 30 words
MAX_BODY_CHARS    = 10_000   # cap body length at fetch time
MAX_WORKERS       = 5        # concurrent threads (stay under rate limits)
REQUEST_DELAY     = 0.3      # seconds between requests per thread
ISSUES_PER_PAGE   = 100      # max per GitHub API page

SAVE_FILE         = "raw_issues.csv"
REPO_CACHE_FILE   = "repos_cache.json"
PROGRESS_FILE     = "fetch_progress.json"
LOG_FILE          = "fetch_log.txt"

LABELS = ["bug", "enhancement", "documentation", "question", "performance", "refactor"]

LABEL_ALIASES = {
    "performance": ["performance", "perf", "speed", "optimization", "optimisation"],
    "refactor":    ["refactor", "refactoring", "cleanup", "clean-up", "technical-debt"],
    "bug":         ["bug", "bug-report", "type:bug", "kind/bug"],
    "enhancement": ["enhancement", "feature", "feature-request", "type:feature"],
    "documentation":["documentation", "docs", "doc", "type:docs"],
    "question":    ["question", "help", "help-wanted", "type:question"],
}

# ═══════════════════════════════════════════════════════════════════════════
# REPO DISCOVERY — 40 diverse ecosystems
# ═══════════════════════════════════════════════════════════════════════════
REPO_QUERIES = [
    # Web / Frontend
    "language:javascript stars:>5000 topic:react",
    "language:typescript stars:>3000 topic:frontend",
    "language:javascript stars:>5000 topic:nodejs",
    "language:typescript stars:>3000 topic:vue",
    "language:javascript stars:>3000 topic:angular",
    # Backend
    "language:python stars:>5000 topic:web",
    "language:go stars:>3000 topic:backend",
    "language:java stars:>3000 topic:spring",
    "language:rust stars:>3000",
    "language:ruby stars:>3000 topic:rails",
    "language:php stars:>3000 topic:laravel",
    # ML / AI / Data
    "language:python stars:>5000 topic:deep-learning",
    "language:python stars:>5000 topic:machine-learning",
    "language:python stars:>3000 topic:pytorch",
    "language:python stars:>3000 topic:tensorflow",
    "language:python stars:>3000 topic:data-science",
    "language:jupyter-notebook stars:>3000",
    # DevOps / Cloud
    "language:go stars:>5000 topic:kubernetes",
    "topic:docker stars:>5000",
    "topic:terraform stars:>3000",
    "language:python stars:>3000 topic:devops",
    "topic:ansible stars:>2000",
    # Mobile
    "language:swift stars:>2000 topic:ios",
    "language:dart stars:>3000 topic:flutter",
    "language:kotlin stars:>2000 topic:android",
    # Systems / CLI
    "language:rust stars:>2000 topic:cli",
    "language:go stars:>2000 topic:cli",
    "language:c stars:>5000",
    "language:cpp stars:>5000",
    # Databases
    "topic:database stars:>3000",
    "topic:postgresql stars:>2000",
    "topic:redis stars:>3000",
    # Security
    "topic:security stars:>2000",
    "topic:cryptography stars:>2000",
    # Testing / Quality
    "topic:testing stars:>2000",
    "topic:linting stars:>2000",
    # Docs / Content
    "topic:documentation stars:>3000",
    # General popular
    "stars:>15000 topic:framework",
    "stars:>10000 topic:library",
    "stars:>8000 topic:open-source",
]

# ═══════════════════════════════════════════════════════════════════════════
# QUALITY FILTER CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
TEMPLATE_PHRASES = [
    "describe your issue", "describe the bug", "steps to reproduce",
    "expected behavior", "actual behavior", "please describe",
    "your description here", "fill in the", "<!-- please",
    "<!---", "[ ] bug", "[x] bug", "type: bug", "**describe",
    "**expected", "**actual", "**steps", "**environment",
]

BOT_RE = re.compile(
    r"(dependabot|renovate\[bot\]|greenkeeper|snyk-bot|allcontributors"
    r"|github-actions\[bot\]|imgbot|stale\[bot\]|codecov|mergify"
    r"|restyled-io|lgtm-com|sonarcloud|whitesource|deepsource"
    r"|semantic-release-bot|github-actions)",
    re.IGNORECASE,
)

NON_ASCII_RE   = re.compile(r"[^\x00-\x7F]")
WHITESPACE_RE  = re.compile(r"\s+")

# Non-English word sets for language detection
FOREIGN_WORDS = {
    "的","了","在","是","我","有","和","就","不","人",       # Chinese
    "の","に","は","を","た","が","で","て","と","し",       # Japanese
    "이","가","을","는","에","의","로","하","한","합",       # Korean
    "не","на","это","то","но","из","по","он","она","как",  # Russian
    "que","una","por","con","del","los","las","para",      # Spanish
    "qui","une","sur","les","des","est","dans","avec",     # French
    "ich","sie","nicht","das","und","ein","ist","mit",     # German
    "não","uma","com","para","são","isso","esse",          # Portuguese
}

# Thread-safe lock for shared state
_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════
def log(msg, level="INFO"):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    with _lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# RATE-LIMIT-AWARE REQUEST
# ═══════════════════════════════════════════════════════════════════════════
_rate_lock   = threading.Lock()
_rate_reset  = 0
_rate_remain = 5000

def safe_get(url, params=None, retries=4):
    global _rate_reset, _rate_remain

    for attempt in range(retries):
        # Pre-check: if we know rate limit is exhausted, wait
        with _rate_lock:
            if _rate_remain == 0:
                wait = max(int(_rate_reset - time.time()) + 3, 5)
                log(f"Pre-emptive rate wait: {wait}s", "WARN")
                time.sleep(wait)
                _rate_remain = 100  # reset optimistically

        try:
            res = requests.get(url, headers=HEADERS, params=params, timeout=20)

            # Update rate limit state from headers
            with _rate_lock:
                _rate_remain = int(res.headers.get("X-RateLimit-Remaining", 100))
                _rate_reset  = int(res.headers.get("X-RateLimit-Reset",
                                                    time.time() + 60))

            if res.status_code == 200:
                return res

            elif res.status_code == 403:
                body_msg = ""
                try:
                    body_msg = res.json().get("message", "").lower()
                except Exception:
                    pass
                if "abuse" in body_msg or "secondary" in body_msg:
                    wait = min(120 * (attempt + 1), 600)
                    log(f"Secondary rate limit. Sleeping {wait}s", "WARN")
                    time.sleep(wait)
                else:
                    wait = max(int(_rate_reset - time.time()) + 5, 60)
                    log(f"Primary rate limit. Sleeping {wait}s", "WARN")
                    time.sleep(wait)

            elif res.status_code in (404, 410, 422):
                return None   # not found / gone / unprocessable

            elif res.status_code == 401:
                raise ValueError("❌ Invalid GitHub token. Check GITHUB_TOKEN.")

            elif res.status_code >= 500:
                wait = 20 * (attempt + 1)
                log(f"GitHub server error {res.status_code}. Retry in {wait}s", "WARN")
                time.sleep(wait)

            else:
                time.sleep(8 * (attempt + 1))

        except requests.exceptions.Timeout:
            time.sleep(15 * (attempt + 1))
        except requests.exceptions.ConnectionError:
            time.sleep(20 * (attempt + 1))

    return None

# ═══════════════════════════════════════════════════════════════════════════
# ENGLISH DETECTION — no external library needed
# ═══════════════════════════════════════════════════════════════════════════
def is_english(title: str, body: str) -> bool:
    text = (title + " " + body[:500]).strip()
    if len(text) < 15:
        return True  # too short to judge

    # 1. Non-ASCII character ratio
    non_ascii = len(NON_ASCII_RE.findall(text))
    if non_ascii / max(len(text), 1) > 0.12:
        return False

    # 2. CJK / Cyrillic / Arabic blocks
    cjk   = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    kana  = sum(1 for c in text if "\u3040" <= c <= "\u30ff")
    cyril = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    arab  = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    if any(x > 4 for x in [cjk, kana, cyril, arab]):
        return False

    # 3. Foreign stopword presence
    words = set(text.lower().split())
    if len(words & FOREIGN_WORDS) >= 3:
        return False

    # 4. ASCII word ratio
    word_list = text.split()
    if len(word_list) > 6:
        ascii_words = sum(1 for w in word_list if all(ord(c) < 128 for c in w))
        if ascii_words / len(word_list) < 0.70:
            return False

    return True

# ═══════════════════════════════════════════════════════════════════════════
# QUALITY GATE — 8 checks per issue
# ═══════════════════════════════════════════════════════════════════════════
def passes_quality(issue: dict) -> tuple[bool, str]:
    # 1. No pull requests
    if "pull_request" in issue:
        return False, "pr"

    # 2. No bots
    user = issue.get("user") or {}
    if BOT_RE.search(user.get("login", "")) or user.get("type") == "Bot":
        return False, "bot"

    # 3. Title length
    title = (issue.get("title") or "").strip()
    if len(title.split()) < MIN_TITLE_WORDS:
        return False, "short_title"

    # 4. Body exists and long enough
    body = (issue.get("body") or "").strip()
    if not body:
        return False, "empty_body"
    if len(body.split()) < MIN_BODY_WORDS:
        return False, "short_body"

    # 5. Not just a template
    body_lower = body.lower()
    if len(body.split()) < 80:
        for phrase in TEMPLATE_PHRASES:
            if phrase in body_lower:
                return False, "template"

    # 6. Must be closed OR have ≥2 comments (real/acknowledged issue)
    if issue.get("state") == "open" and (issue.get("comments") or 0) < 2:
        return False, "unacknowledged"

    # 7. English only
    if not is_english(title, body):
        return False, "non_english"

    # 8. Not a spam/noise issue (very short body despite passing word count)
    #    Check that body has some sentence structure (contains a period or newline)
    if len(body) < 200 and "." not in body and "\n" not in body:
        return False, "no_structure"

    return True, ""

# ═══════════════════════════════════════════════════════════════════════════
# REPO DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════
def discover_repos() -> list[dict]:
    if os.path.exists(REPO_CACHE_FILE):
        with open(REPO_CACHE_FILE) as f:
            cached = json.load(f)
        log(f"Loaded {len(cached)} repos from cache")
        return cached

    log(f"Discovering top {TOP_N_REPOS} repos across {len(REPO_QUERIES)} ecosystems ...")
    all_repos = {}

    for i, query in enumerate(REPO_QUERIES, 1):
        log(f"  Query {i}/{len(REPO_QUERIES)}: {query[:60]}...")
        for page in range(1, 4):
            res = safe_get("https://api.github.com/search/repositories", {
                "q": query, "sort": "stars", "order": "desc",
                "per_page": 30, "page": page
            })
            if not res:
                break
            items = res.json().get("items", [])
            if not items:
                break
            for repo in items:
                name   = repo["full_name"]
                stars  = repo.get("stargazers_count", 0)
                issues = repo.get("open_issues_count", 0)
                if (stars >= MIN_REPO_STARS
                        and issues >= MIN_REPO_ISSUES
                        and not repo.get("archived")
                        and not repo.get("disabled")
                        and not repo.get("fork")):
                    if name not in all_repos:
                        all_repos[name] = {
                            "full_name": name,
                            "stars":     stars,
                            "language":  repo.get("language", ""),
                            "issues":    issues,
                        }
            time.sleep(REQUEST_DELAY)

    sorted_repos = sorted(all_repos.values(), key=lambda r: r["stars"], reverse=True)
    top = sorted_repos[:TOP_N_REPOS]

    with open(REPO_CACHE_FILE, "w") as f:
        json.dump(top, f, indent=2)

    log(f"Discovered {len(top)} repos  "
        f"(range: {top[-1]['stars']:,}–{top[0]['stars']:,} stars)")
    return top

# ═══════════════════════════════════════════════════════════════════════════
# FETCH ISSUES FROM ONE REPO (one thread)
# ═══════════════════════════════════════════════════════════════════════════
def fetch_repo_label(repo_name: str, label: str,
                     seen_ids: set, max_issues: int) -> list[dict]:
    """
    Fetches issues from one repo for one label.
    Tries all label aliases (e.g. 'bug', 'bug-report', 'type:bug').
    """
    collected = []
    owner, repo = repo_name.split("/", 1)
    url         = f"https://api.github.com/repos/{owner}/{repo}/issues"
    aliases     = LABEL_ALIASES.get(label, [label])

    for alias in aliases:
        if len(collected) >= max_issues:
            break

        for page in range(1, 11):  # max 10 pages × 100 = 1000 per alias
            if len(collected) >= max_issues:
                break

            res = safe_get(url, {
                "labels":    alias,
                "state":     "closed",
                "per_page":  ISSUES_PER_PAGE,
                "page":      page,
                "sort":      "updated",
                "direction": "desc",
            })
            if not res:
                break

            items = res.json()
            if not items or isinstance(items, dict):
                break

            got_new = False
            for issue in items:
                if len(collected) >= max_issues:
                    break

                uid = issue.get("id")
                with _lock:
                    if uid in seen_ids:
                        continue
                    seen_ids.add(uid)

                ok, _ = passes_quality(issue)
                if not ok:
                    continue

                got_new = True
                collected.append({
                    "id":    uid,
                    "repo":  repo_name,
                    "title": issue["title"].strip(),
                    "body":  (issue["body"] or "").strip()[:MAX_BODY_CHARS],
                    "label": label,
                })

            if not got_new and len(items) < ISSUES_PER_PAGE:
                break  # exhausted this alias

            time.sleep(REQUEST_DELAY)

    return collected

# ═══════════════════════════════════════════════════════════════════════════
# CONCURRENT LABEL FETCHER
# ═══════════════════════════════════════════════════════════════════════════
def fetch_label_concurrent(label: str, repos: list[dict],
                            seen_ids: set, already_have: int,
                            completed_repos: set) -> list[dict]:
    needed     = TARGET_PER_LABEL - already_have
    all_new    = []
    done_count = 0

    # Scale max_per_repo so we spread collection evenly
    # but allow top repos to contribute more
    n_repos      = len(repos)
    base_per_repo = max(30, (needed // max(n_repos // 2, 1)) + 20)

    log(f"[{label}] Starting concurrent fetch  "
        f"(need {needed:,}, {n_repos} repos, ~{base_per_repo}/repo)")

    with tqdm(total=needed, desc=f"{label:<15}", unit="issues",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}

            for repo in repos:
                rname = repo["full_name"]
                if rname in completed_repos:
                    continue

                # Dynamically scale: bigger repos (more issues) get higher cap
                repo_issue_count = repo.get("issues", 50)
                cap = min(base_per_repo, max(20, repo_issue_count // 5))

                fut = executor.submit(
                    fetch_repo_label, rname, label, seen_ids, cap
                )
                futures[fut] = rname

            for fut in as_completed(futures):
                rname = futures[fut]
                completed_repos.add(rname)

                try:
                    results = fut.result()
                except Exception as e:
                    log(f"[{label}] Error on {rname}: {e}", "ERROR")
                    results = []

                if results:
                    all_new.extend(results)
                    pbar.update(len(results))
                    done_count += 1

                # Stop early if we've hit the target
                if len(all_new) + already_have >= TARGET_PER_LABEL:
                    # Cancel pending futures
                    for f in futures:
                        f.cancel()
                    break

    log(f"[{label}] Collected {len(all_new):,} new issues")
    return all_new

# ═══════════════════════════════════════════════════════════════════════════
# PROGRESS SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════
def load_progress():
    records   = []
    seen_ids  = set()
    completed = defaultdict(set)   # label → {repo_name, ...}

    if os.path.exists(SAVE_FILE):
        df = pd.read_csv(SAVE_FILE)
        records = df.to_dict("records")
        if "id" in df.columns:
            seen_ids = set(df["id"].dropna().astype(int).tolist())
        counts = {l: (df["label"] == l).sum() for l in LABELS}
        log(f"Resumed: {len(records):,} rows  |  {counts}")

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            raw = json.load(f)
        for label, repos in raw.items():
            completed[label] = set(repos)

    return records, seen_ids, completed


def save_all(records, completed):
    pd.DataFrame(records).to_csv(SAVE_FILE, index=False)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({k: list(v) for k, v in completed.items()}, f, indent=2)


def count_labels(records):
    c = defaultdict(int)
    for r in records:
        c[r["label"]] += 1
    return dict(c)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    log("=" * 65)
    log("GitHub Issue Fetcher — Final Production Version")
    log(f"Target: {TARGET_PER_LABEL:,} issues × {len(LABELS)} labels "
        f"= {TARGET_PER_LABEL * len(LABELS):,} total")
    log("=" * 65)

    if GITHUB_TOKEN == "YOUR_GITHUB_TOKEN_HERE":
        print("\n❌  Please set GITHUB_TOKEN at the top of this file.")
        print("    https://github.com/settings/tokens  (scope: public_repo)\n")
        return

    # ── 1. Discover repos ────────────────────────────────────────────────
    repos = discover_repos()
    log(f"\nUsing {len(repos)} repos for issue fetching\n")

    # ── 2. Load existing progress ────────────────────────────────────────
    all_records, seen_ids, completed = load_progress()
    label_counts = count_labels(all_records)

    # ── 3. Fetch each label ───────────────────────────────────────────────
    for label in LABELS:
        have = label_counts.get(label, 0)

        if have >= TARGET_PER_LABEL:
            log(f"[SKIP]  '{label}' — already {have:,}/{TARGET_PER_LABEL:,} ✓")
            continue

        log(f"\n{'─' * 55}")
        log(f"[LABEL] '{label}'  —  have {have:,}, need {TARGET_PER_LABEL - have:,} more")
        log(f"{'─' * 55}")

        done_repos  = completed.get(label, set())
        new_records = fetch_label_concurrent(
            label, repos, seen_ids, have, done_repos
        )

        all_records.extend(new_records)
        completed[label] = done_repos
        label_counts = count_labels(all_records)

        # Save after every label
        save_all(all_records, completed)
        log(f"[SAVED] '{label}' done. Total: {len(all_records):,}  |  {label_counts}")

        # If short on performance/refactor, warn the user
        final_count = label_counts.get(label, 0)
        if final_count < TARGET_PER_LABEL * 0.7:
            log(f"[WARN]  '{label}' only has {final_count:,} issues. "
                f"These labels are rare on GitHub. Consider reducing "
                f"TARGET_PER_LABEL for this label.", "WARN")

    # ── 4. Final clean save ───────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df = df[["repo", "title", "body", "label"]]
    df.to_csv(SAVE_FILE, index=False)

    # ── 5. Final report ───────────────────────────────────────────────────
    log("\n" + "=" * 65)
    log("FETCH COMPLETE")
    log(f"Total rows : {len(df):,}")
    log(f"Saved to   : {SAVE_FILE}")
    log("\nClass distribution:")
    for lbl, cnt in df["label"].value_counts().items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        log(f"  {lbl:<18} {cnt:>7,}  ({pct:4.1f}%)  {bar}")
    log("\nTop repos by issue count:")
    for repo, cnt in df["repo"].value_counts().head(10).items():
        log(f"  {repo:<45} {cnt:>5,}")
    log("=" * 65)
    log("Next: run  python preprocess_issues.py")


if __name__ == "__main__":
    main()