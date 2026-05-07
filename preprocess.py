"""
GitHub Issue Triage — Preprocessor v3  (targets 85-90% accuracy)
=================================================================
Input  : raw_issues.csv
Output : dataset_final.csv  →  repo | title | body | label

ROOT CAUSES OF 74% FIXED HERE:
  ✓ FIX 1 — NO duplicate oversampling (refactor had 67% duplicates → overfitting)
             Imbalance handled purely in training: class_weight + FocalLoss
  ✓ FIX 2 — MAX_BODY_WORDS = 350 (was 220; bug/performance 75th-pct = 250 words)
  ✓ FIX 3 — Casing fully preserved (RoBERTa + CodeBERT are case-sensitive)
  ✓ FIX 4 — Semantic tokens kept: [CODE] [STACKTRACE] [VERSION] [FILEPATH] [URL]
  ✓ FIX 5 — Majority cap at 3500, minority kept entirely (658 / 2155 rows)

Run: python preprocess_issues.py
"""

import pandas as pd
import re
import unicodedata
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE   = "raw_issues.csv"
OUTPUT_FILE  = "datasetfinal.csv"
REPORT_FILE  = "preprocessing_report.txt"

VALID_LABELS    = ["bug","enhancement","documentation","question","performance","refactor"]
MIN_TITLE_WORDS = 3
MIN_BODY_WORDS  = 20
MAX_BODY_WORDS  = 350   # FIX 2: was 220
MAJORITY_CAP    = 3_500

# ── Regex (compiled once) ────────────────────────────────────────────────
RE_HTML_COMMENT  = re.compile(r'<!--[\s\S]*?-->', re.DOTALL)
RE_HTML_TAG      = re.compile(r'<[^>]{1,300}>')
RE_HTML_ENTITY   = re.compile(r'&[a-zA-Z]{2,8};|&#?\w{1,8};')
RE_MD_IMG        = re.compile(r'!\[[^\]]*\]\([^\)]*\)')
RE_MD_HR         = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)
RE_MD_TABLE      = re.compile(r'^\|.*\|.*$', re.MULTILINE)
RE_MD_BLOCKQUOTE = re.compile(r'^\s*>\s+', re.MULTILINE)
RE_CHECKBOX      = re.compile(r'^[-*]\s*\[[ xX]\][^\n]*$', re.MULTILINE)
RE_CODE_BLOCK    = re.compile(r'```[\s\S]*?```', re.DOTALL)
RE_INLINE_CODE   = re.compile(r'`[^`\n]{1,300}`')
RE_MD_LINK       = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
RE_MD_BOLD_IT    = re.compile(r'\*{1,3}([^*\n]{1,200}?)\*{1,3}')
RE_MD_STRIKE     = re.compile(r'~~([^~\n]+)~~')
RE_MD_HEADER     = re.compile(r'^#{1,6}\s+', re.MULTILINE)
RE_MD_LIST       = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)
RE_MD_NUM_LIST   = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
RE_URL           = re.compile(r'https?://\S+|www\.\S+')
RE_VERSION       = re.compile(r'\bv?\d+\.\d+[\.\d\-\w]*\b')
RE_FILE_PATH     = re.compile(
    r'(?:/[\w.\-]+){2,}|[\w\-]+\.(js|ts|py|java|go|rb|rs|cpp|c|h'
    r'|jsx|tsx|vue|css|html|json|yaml|yml|xml|md|sh|swift|kt|dart)\b'
)
RE_HEX_HASH      = re.compile(r'\b[0-9a-f]{7,40}\b')
RE_ISSUE_REF     = re.compile(r'#\d+')
RE_STACKTRACE    = re.compile(
    r'(Traceback \(most recent call last\)[\s\S]*?(?=\n\n|\Z)'
    r'|Exception in thread[\s\S]*?(?=\n\n|\Z)'
    r'|(?:\s+at [\w\.$<>]+\([\w\.]+:\d+\)\n){3,})',
    re.MULTILINE
)
TEMPLATE_HEADERS = re.compile(
    r'^#{1,4}\s*(version|reproduction\s*link?|steps\s+to\s+reproduce'
    r'|minimal\s+repro(?:duction)?|what\s+is\s+expected|what\s+is\s+actually'
    r'|expected\s+behav|actual\s+behav|environment|platform|system\s+info'
    r'|additional\s+context|describe\s+the\s+bug|to\s+reproduce)[^\n]*',
    re.IGNORECASE | re.MULTILINE
)
API_SPAM_RE = re.compile(
    r'(##\s*api\s*(submission|information)|\*\*api\s*(name|url)\*\*|api\s+url\s*:)',
    re.IGNORECASE | re.MULTILINE
)
RE_NON_ASCII  = re.compile(r'[^\x00-\x7F]')
RE_CRLF       = re.compile(r'\r\n|\r')
RE_SPACES     = re.compile(r'[ \t]+')
RE_BLANK_LINES= re.compile(r'\n{3,}')
RE_PUNCT_RPT  = re.compile(r'([!?.,-]){3,}')

FOREIGN_WORDS = {
    "的","了","在","是","我","有","和","就","不","人",
    "の","に","は","を","た","が","で","て","と","し",
    "이","가","을","는","에","의","로","하","한","합",
    "не","на","это","то","но","из","по","он","она","как",
    "que","una","por","con","del","los","las","para","está",
    "qui","une","sur","les","des","est","dans","avec","pas",
    "ich","sie","nicht","das","und","ein","ist","mit","den",
    "não","uma","com","para","são","isso","esse",
}


def is_english(title: str, body: str) -> bool:
    text = (title + " " + body[:500]).strip()
    if len(text) < 15:
        return True
    if len(RE_NON_ASCII.findall(text)) / max(len(text), 1) > 0.12:
        return False
    for c in text:
        cp = ord(c)
        if 0x4E00<=cp<=0x9FFF or 0x3040<=cp<=0x30FF: return False
        if 0x0400<=cp<=0x04FF or 0x0600<=cp<=0x06FF: return False
    if len(set(text.lower().split()) & FOREIGN_WORDS) >= 3:
        return False
    words = text.split()
    if len(words) > 6:
        if sum(1 for w in words if all(ord(c)<128 for c in w))/len(words) < 0.70:
            return False
    return True


def clean_title(raw: str) -> str:
    if not isinstance(raw, str): return ""
    t = unicodedata.normalize("NFKC", raw)
    t = RE_CRLF.sub(" ", t)
    t = RE_HTML_TAG.sub(" ", t);  t = RE_HTML_ENTITY.sub(" ", t)
    t = RE_MD_IMG.sub(" ", t);    t = RE_URL.sub(" ", t)
    t = RE_ISSUE_REF.sub(" ", t)
    t = re.sub(r'^\s*\[[^\]]{1,25}\]\s*', '', t)
    t = re.sub(r'^\s*\([^\)]{1,20}\)\s*', '', t)
    t = RE_PUNCT_RPT.sub(r'\1', t)
    return RE_SPACES.sub(" ", t).strip()


def clean_body(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip(): return ""
    b = unicodedata.normalize("NFKC", raw)
    b = RE_CRLF.sub("\n", b)
    b = RE_HTML_COMMENT.sub(" ", b);   b = RE_CHECKBOX.sub(" ", b)
    b = RE_MD_IMG.sub(" ", b);         b = RE_MD_HR.sub(" ", b)
    b = RE_MD_TABLE.sub(" ", b);       b = RE_MD_BLOCKQUOTE.sub(" ", b)
    # Signal-preserving token replacements
    b = RE_STACKTRACE.sub(" [STACKTRACE] ", b)
    b = RE_CODE_BLOCK.sub(" [CODE] ", b);    b = RE_INLINE_CODE.sub(" [CODE] ", b)
    b = RE_URL.sub(" [URL] ", b)
    b = RE_VERSION.sub(" [VERSION] ", b);    b = RE_FILE_PATH.sub(" [FILEPATH] ", b)
    # Strip low-signal noise
    b = RE_HEX_HASH.sub(" ", b);       b = RE_ISSUE_REF.sub(" ", b)
    b = RE_HTML_TAG.sub(" ", b);       b = RE_HTML_ENTITY.sub(" ", b)
    # Markdown: strip syntax, keep text
    b = TEMPLATE_HEADERS.sub(" ", b)
    b = RE_MD_LINK.sub(r'\1', b);      b = RE_MD_BOLD_IT.sub(r'\1', b)
    b = RE_MD_STRIKE.sub(r'\1', b);    b = RE_MD_HEADER.sub(" ", b)
    b = RE_MD_LIST.sub(" ", b);        b = RE_MD_NUM_LIST.sub(" ", b)
    b = RE_PUNCT_RPT.sub(r'\1', b)
    b = RE_SPACES.sub(" ", b);         b = RE_BLANK_LINES.sub("\n", b)
    b = b.strip()
    words = b.split()
    if len(words) > MAX_BODY_WORDS:
        b = " ".join(words[:MAX_BODY_WORDS])
    return b


def main():
    stages = []

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    for col in ["repo","title","body","label"]:
        if col not in df.columns: df[col] = ""
    print(f"[LOAD]   {len(df):,} rows"); stages.append(("1. Raw loaded", len(df)))

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(VALID_LABELS)].copy()
    stages.append(("2. Valid labels", len(df)))

    before = len(df)
    spam = df["body"].astype(str).apply(lambda b: bool(API_SPAM_RE.search(b[:500])))
    df = df[~spam].copy()
    print(f"[SPAM]   removed {before-len(df)} rows"); stages.append(("3. Spam removed", len(df)))

    print("[CLEAN]  Cleaning ...")
    df["title"] = df["title"].astype(str).apply(clean_title)
    df["body"]  = df["body"].astype(str).apply(clean_body)

    before = len(df)
    twc = df["title"].str.split().str.len().fillna(0)
    bwc = df["body"].str.split().str.len().fillna(0)
    df  = df[(twc >= MIN_TITLE_WORDS) & (bwc >= MIN_BODY_WORDS)].copy()
    df  = df[(df["title"].str.strip()!="") & (df["body"].str.strip()!="")]
    print(f"[FILTER] {before:,} → {len(df):,}"); stages.append(("4. Quality filter", len(df)))

    print("[LANG]   Filtering non-English ...")
    before = len(df)
    eng    = [is_english(t, b) for t, b in zip(df["title"].tolist(), df["body"].tolist())]
    df     = df[eng].copy()
    print(f"[LANG]   removed {before-len(df)}"); stages.append(("5. English only", len(df)))

    before = len(df)
    df.drop_duplicates(subset=["title","body"], keep="first", inplace=True)
    print(f"[DEDUP]  removed {before-len(df)}"); stages.append(("6. Deduplicated", len(df)))

    # FIX 1: NO oversampling — handle imbalance in training only
    print("\n[BALANCE] Downsampling majority only — no oversampling:")
    balanced = []
    for label in VALID_LABELS:
        sub = df[df["label"]==label].copy()
        n   = len(sub)
        if n == 0: print(f"  {label}: MISSING"); continue
        sub["_q"] = sub["body"].str.split().str.len()
        sub = sub.sort_values("_q", ascending=False).drop(columns=["_q"])
        if n > MAJORITY_CAP:
            sub = sub.head(MAJORITY_CAP).sample(frac=1, random_state=42)
            print(f"  {label:<16}: {n:,} → {MAJORITY_CAP:,}")
        else:
            print(f"  {label:<16}: {n:,} (all kept, class_weight in trainer)")
        balanced.append(sub)

    df = pd.concat(balanced, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    stages.append(("7. Balanced", len(df)))

    for col in ["title","body","label"]:
        df = df[df[col].notna() & (df[col].astype(str).str.strip()!="")]
    df = df[df["label"].isin(VALID_LABELS)]
    df = df[["repo","title","body","label"]].reset_index(drop=True)
    stages.append(("8. Final validated", len(df)))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SAVED]  {OUTPUT_FILE}  ({len(df):,} rows)")

    vc    = df["label"].value_counts()
    bwc_s = df["body"].str.split().str.len()
    ratio = vc.max() / vc.min()

    report = "\n".join([
        "="*62, "  PREPROCESSING REPORT v3", "="*62,"",
        "PIPELINE:", *[f"  {n:<32} {c:>8,}" for n,c in stages],"",
        f"  Retention: {len(df)/stages[0][1]*100:.1f}%","",
        "LABEL DISTRIBUTION:",
        *[f"  {l:<18} {c:>6,}  ({c/len(df)*100:.1f}%)" for l,c in vc.items()],"",
        f"  Balance ratio: {ratio:.2f}x (handled by class_weight+FocalLoss)","",
        f"BODY: mean={bwc_s.mean():.0f}w  median={bwc_s.median():.0f}w  max={bwc_s.max()}w","",
        f"NULL/EMPTY: {df.isnull().sum().sum()} nulls, {(df=='').sum().sum()} empties","",
        "="*62,"  USAGE","="*62,"",
        "  tokenizer(title, body, max_length=384, truncation=True)",
        "  ML: X = df['title'] + ' ' + df['body']",
        "="*62,
    ])
    print("\n" + report)
    with open(REPORT_FILE,"w") as f: f.write(report)

if __name__ == "__main__":
    main()