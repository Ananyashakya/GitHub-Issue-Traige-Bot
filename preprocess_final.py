"""
GitHub Issue Triage — Final Preprocessing (Accuracy Fix)
==========================================================
Input:  datasetfinal.csv   (16,813 rows)
Output: dataset_ready.csv  (~14,000 rows, 5 labels)

FIXES vs previous version that gave 67% accuracy:
  ✗ OLD: Replaced code with [CODE] placeholder → RoBERTa wastes ~60 tokens/issue
  ✓ NEW: Remove placeholders entirely → more real content in 384-token budget

  ✗ OLD: Kept [URL],[VERSION],[FILEPATH] tokens → pure noise for classification
  ✓ NEW: Strip all placeholder tokens completely

  ✗ OLD: Refactor label kept (658 rows) → severe imbalance killed minority classes
  ✓ NEW: Refactor removed (as requested)

  ✗ OLD: No minimum content check after cleaning
  ✓ NEW: Require 25+ content words after cleaning

  ✓ Title casing preserved (RoBERTa is case-sensitive)
  ✓ Body casing preserved (Error ≠ error for RoBERTa)
  ✓ Balanced at 2,500 per label (5 labels = 12,500 total — manageable + quality)

Final columns: repo | title | body | label
"""

import pandas as pd
import numpy as np
import re
import unicodedata

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
INPUT_FILE   = "datasetfinal.csv"
OUTPUT_FILE  = "dataset_ready.csv"
REPORT_FILE  = "preprocessing_report.txt"

VALID_LABELS    = ["bug", "enhancement", "documentation", "question", "performance"]
MIN_CONTENT_WORDS = 25    # after cleaning, body must have 25+ real words
MAX_BODY_WORDS    = 280   # cap — fits well in RoBERTa's 384 token budget
BALANCE_CAP       = 2_500 # per label — keeps best quality rows

# ══════════════════════════════════════════════════════════════════
# REGEX
# ══════════════════════════════════════════════════════════════════
# Remove ALL placeholder tokens inserted by previous preprocessing
RE_PLACEHOLDERS = re.compile(
    r'\[\s*(?:CODE|INLINE_CODE|CODE_BLOCK|URL|VERSION|FILEPATH|'
    r'HASH|STACKTRACE|INLINE CODE|CODE BLOCK)\s*\]',
    re.IGNORECASE
)

# Remaining markdown/code noise that survived previous preprocessing
RE_CODE_BLOCK   = re.compile(r'```[\s\S]*?```', re.DOTALL)
RE_INLINE_CODE  = re.compile(r'`[^`\n]{1,300}`')
RE_HTML_COMMENT = re.compile(r'<!--[\s\S]*?-->', re.DOTALL)
RE_HTML_TAG     = re.compile(r'<[^>]{1,300}>')
RE_HTML_ENTITY  = re.compile(r'&[a-zA-Z]{2,8};|&#?\w{1,8};')
RE_URL          = re.compile(r'https?://\S+|www\.\S+')
RE_MD_IMG       = re.compile(r'!\[[^\]]*\]\([^\)]*\)')
RE_MD_LINK      = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
RE_MD_HEADER    = re.compile(r'^#{1,6}\s+', re.MULTILINE)
RE_MD_LIST      = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)
RE_MD_BOLD_IT   = re.compile(r'\*{1,3}([^*\n]{1,200}?)\*{1,3}')
RE_MD_TABLE     = re.compile(r'^\|.*\|.*$', re.MULTILINE)
RE_MD_HR        = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)
RE_STACKTRACE   = re.compile(
    r'(Traceback \(most recent call last\)[\s\S]*?(?=\n\n|\Z)'
    r'|\s+at [\w\.$<>]+\([\w\.]+:\d+\)'
    r'|Exception in thread[\s\S]*?(?=\n\n|\Z))',
    re.MULTILINE
)
RE_LOG_LINE     = re.compile(
    r'^\[?\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}[^\n]*$', re.MULTILINE
)
RE_HEX          = re.compile(r'\b[0-9a-f]{8,40}\b')
RE_ISSUE_REF    = re.compile(r'#\d+')
RE_DIVIDER      = re.compile(r'[-=*_~`]{4,}')
RE_PUNCT_REP    = re.compile(r'([!?.]){3,}')
RE_CRLF         = re.compile(r'\r\n|\r')
RE_SPACES       = re.compile(r'[ \t]+')
RE_BLANK_LINES  = re.compile(r'\n{3,}')

# Language detection
RE_NON_ASCII    = re.compile(r'[^\x00-\x7F]')
FOREIGN_WORDS   = {
    "的","了","在","是","我","有","和","就","不","人",
    "の","に","は","を","た","が","で","て","と","し",
    "이","가","을","는","에","의","로","하","한","합",
    "не","на","это","то","но","из","по","он","она","как",
    "que","una","por","con","del","los","las","para",
    "qui","une","sur","les","des","est","dans","avec",
    "ich","sie","nicht","das","und","ein","ist","mit",
}

BOT_RE = re.compile(
    r'(dependabot|renovate\[bot\]|greenkeeper|snyk-bot'
    r'|github-actions\[bot\]|imgbot|stale\[bot\]|codecov)',
    re.IGNORECASE
)


# ══════════════════════════════════════════════════════════════════
# CLEAN TITLE — preserve casing for RoBERTa
# ══════════════════════════════════════════════════════════════════
def clean_title(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    t = unicodedata.normalize("NFKC", raw)
    t = RE_CRLF.sub(" ", t)
    t = RE_PLACEHOLDERS.sub(" ", t)  # remove any leftover placeholders
    t = RE_HTML_TAG.sub(" ", t)
    t = RE_HTML_ENTITY.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_ISSUE_REF.sub(" ", t)
    t = RE_DIVIDER.sub(" ", t)
    # Remove bracket/paren prefix tags: [BUG], [WIP], (fix)
    t = re.sub(r'^\s*[\[\(][^\]\)]{1,20}[\]\)]\s*[:\-]?\s*', '', t)
    # Remove remaining square bracket tags anywhere
    t = re.sub(r'\[[^\]]{1,25}\]', ' ', t)
    t = RE_INLINE_CODE.sub(' ', t)   # remove `code` in titles
    t = RE_SPACES.sub(" ", t).strip()
    # Keep original casing — RoBERTa is case-sensitive
    return t


# ══════════════════════════════════════════════════════════════════
# CLEAN BODY — remove ALL placeholder tokens, preserve casing
# ══════════════════════════════════════════════════════════════════
def clean_body(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""

    b = unicodedata.normalize("NFKC", raw)
    b = RE_CRLF.sub("\n", b)

    # ── Step 1: Remove ALL placeholder tokens (the main accuracy fix) ──
    b = RE_PLACEHOLDERS.sub(" ", b)

    # ── Step 2: Remove remaining code/HTML that survived preprocessing ──
    b = RE_HTML_COMMENT.sub(" ", b)
    b = RE_CODE_BLOCK.sub(" ", b)
    b = RE_INLINE_CODE.sub(" ", b)
    b = RE_STACKTRACE.sub(" ", b)
    b = RE_LOG_LINE.sub(" ", b)
    b = RE_HTML_TAG.sub(" ", b)
    b = RE_HTML_ENTITY.sub(" ", b)
    b = RE_MD_IMG.sub(" ", b)
    b = RE_MD_TABLE.sub(" ", b)
    b = RE_MD_HR.sub(" ", b)

    # ── Step 3: Strip markdown syntax, keep readable text ──────────────
    b = RE_MD_LINK.sub(r'\1', b)
    b = RE_MD_BOLD_IT.sub(r'\1', b)
    b = RE_MD_HEADER.sub("\n", b)
    b = RE_MD_LIST.sub(" ", b)

    # ── Step 4: Remove URLs, hashes, refs ──────────────────────────────
    b = RE_URL.sub(" ", b)
    b = RE_HEX.sub(" ", b)
    b = RE_ISSUE_REF.sub(" ", b)
    b = RE_PUNCT_REP.sub(r'\1', b)
    b = RE_DIVIDER.sub(" ", b)

    # ── Step 5: Whitespace normalisation ───────────────────────────────
    b = RE_SPACES.sub(" ", b)
    b = RE_BLANK_LINES.sub("\n", b)
    b = b.strip()

    # ── Step 6: Truncate to MAX_BODY_WORDS ─────────────────────────────
    words = b.split()
    if len(words) > MAX_BODY_WORDS:
        b = " ".join(words[:MAX_BODY_WORDS])

    return b


# ══════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════
def is_english(title: str, body: str) -> bool:
    text = (title + " " + body[:400]).strip()
    if len(text) < 15:
        return True
    non_ascii = len(RE_NON_ASCII.findall(text))
    if non_ascii / max(len(text), 1) > 0.12:
        return False
    cjk   = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    kana  = sum(1 for c in text if "\u3040" <= c <= "\u30ff")
    cyril = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    arab  = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    if any(x > 4 for x in [cjk, kana, cyril, arab]):
        return False
    words = set(text.lower().split())
    if len(words & FOREIGN_WORDS) >= 3:
        return False
    return True


# ══════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════
def main():
    stages = []

    # ── Load ──────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    print(f"[LOAD]   {len(df):,} rows")
    stages.append(("1. Loaded", len(df)))

    # ── Drop refactor (as requested) + keep valid labels ──────────
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(VALID_LABELS)].copy()
    stages.append(("2. Removed refactor + invalid", len(df)))
    print(f"[LABEL]  {len(df):,} rows (refactor removed)")

    # ── Remove bot issues ─────────────────────────────────────────
    before = len(df)
    bot_mask = (
        df["title"].astype(str).str.contains(BOT_RE, regex=True, na=False) |
        df["body"].astype(str).str.contains(BOT_RE, regex=True, na=False)
    )
    df = df[~bot_mask].copy()
    stages.append(("3. Bots removed", len(df)))

    # ── Clean text ────────────────────────────────────────────────
    print("[CLEAN]  Cleaning title and body (removing all placeholder tokens)...")
    df["title"] = df["title"].apply(clean_title)
    df["body"]  = df["body"].apply(clean_body)

    # ── Drop rows with insufficient content AFTER cleaning ────────
    before = len(df)
    # Title: at least 3 meaningful words
    df = df[df["title"].str.split().str.len().fillna(0) >= 3].copy()
    # Body: at least MIN_CONTENT_WORDS real words (no placeholder noise)
    df = df[df["body"].str.split().str.len().fillna(0) >= MIN_CONTENT_WORDS].copy()
    stages.append(("4. Min content words", len(df)))
    print(f"[FILTER] {before:,} → {len(df):,}")

    # ── English only ──────────────────────────────────────────────
    print("[LANG]   Filtering non-English...")
    before = len(df)
    eng = [is_english(t, b) for t, b in zip(df["title"].tolist(), df["body"].tolist())]
    df  = df[eng].copy()
    stages.append(("5. English only", len(df)))
    print(f"[LANG]   {before:,} → {len(df):,}")

    # ── Deduplication ─────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset=["title", "label"], keep="first", inplace=True)
    df.drop_duplicates(subset=["body",  "label"], keep="first", inplace=True)
    stages.append(("6. Deduplicated", len(df)))
    print(f"[DEDUP]  {before:,} → {len(df):,}")

    # ── Balance classes ───────────────────────────────────────────
    # Keep best rows = longest bodies (most informative for the model)
    df["_blen"] = df["body"].str.split().str.len()
    balanced    = []
    for lbl in VALID_LABELS:
        subset = (df[df["label"] == lbl]
                    .sort_values("_blen", ascending=False)
                    .head(BALANCE_CAP))
        balanced.append(subset)
        print(f"[BAL]    {lbl:<18} {len(subset):,} rows")
    df = pd.concat(balanced, ignore_index=True)
    df.drop(columns=["_blen"], inplace=True)
    stages.append(("7. Balanced", len(df)))

    # ── Final guarantee ───────────────────────────────────────────
    for col in ["repo", "title", "body", "label"]:
        df = df[df[col].notna()]
        df = df[df[col].astype(str).str.strip() != ""]
    df = df[df["label"].isin(VALID_LABELS)]
    df = df[["repo", "title", "body", "label"]].reset_index(drop=True)
    stages.append(("8. Final validated", len(df)))

    # ── Save ──────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SAVED]  → {OUTPUT_FILE}  ({len(df):,} rows)")

    # ── Report ────────────────────────────────────────────────────
    counts = df["label"].value_counts()
    twords = df["title"].str.split().str.len()
    bwords = df["body"].str.split().str.len()

    # Verify no placeholders remain
    ph_remaining = df["body"].str.count(
        r'\[(CODE|URL|VERSION|FILEPATH|HASH)\]'
    ).sum()

    lines = [
        "=" * 60,
        "  FINAL PREPROCESSING REPORT",
        "=" * 60, "",
        "PIPELINE STAGES:",
        *[f"  {n:<32} {c:>7,}" for n, c in stages],
        "",
        f"  Retention : {len(df)/stages[0][1]*100:.1f}%",
        "",
        "CLASS DISTRIBUTION:",
        *[f"  {l:<18} {c:>6,}  ({c/len(df)*100:.1f}%)"
          for l, c in counts.items()],
        "",
        "TITLE  stats (words):",
        f"  mean={twords.mean():.1f}  median={twords.median():.0f}"
        f"  min={twords.min()}  max={twords.max()}",
        "",
        "BODY   stats (words):",
        f"  mean={bwords.mean():.1f}  median={bwords.median():.0f}"
        f"  min={bwords.min()}  max={bwords.max()}",
        "",
        f"PLACEHOLDER TOKENS REMAINING: {ph_remaining}",
        "(should be 0 — if not 0, something went wrong)",
        "",
        "NULL/EMPTY CHECK:",
        f"  Nulls        : {df.isnull().sum().sum()}",
        f"  Empty strings: {(df == '').sum().sum()}",
        "=" * 60,
    ]

    report = "\n".join(lines)
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()