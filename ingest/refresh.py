"""
Refresh The Watcher's knowledge base.

Runs weekly in GitHub Actions (see .github/workflows/refresh.yml), or locally:

    pip install -r ingest/requirements.txt
    CF_ACCOUNT_ID=... CF_API_TOKEN=... python ingest/refresh.py
    python ingest/refresh.py --dry-run      # crawl + chunk only, no Cloudflare calls

What it does
1. Loads the handbook (data/handbook.md) and crawls the ITU website.
2. Splits everything into chunks with stable ids.
3. Compares with data/index_manifest.json and only embeds chunks that changed.
4. Upserts new/changed chunks into Cloudflare Vectorize, deletes ones that vanished.
5. Writes the new manifest (the workflow commits it back to the repo).

Safety: a page that fails to download keeps its old chunks. If the crawl as a
whole looks broken (too few pages), nothing is deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
HANDBOOK = ROOT / "data" / "handbook.md"
EXTRA_URLS = ROOT / "data" / "extra_urls.txt"
MANIFEST = ROOT / "data" / "index_manifest.json"

DEFAULT_SEEDS = [
    "https://itu.edu.pk/academics/",
    "https://itu.edu.pk/admissions/",
    "https://itu.edu.pk/faculties/",
    "https://itu.edu.pk/examinations/",
    "https://itu.edu.pk/research/",
    "https://itu.edu.pk/administration/",
]
KEYWORDS = [
    "academic", "faculty", "faculties", "program", "department", "admission", "fee",
    "examination", "exam", "research", "administration", "scholarship", "financial",
    "calendar", "policy", "policies", "undergraduate", "graduate", "phd", "ms-", "bs-",
    "hostel", "transport", "contact", "registrar", "student", "school", "center", "centre",
]
SKIP_IN_URL = [".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".doc", "wp-content", "wp-json",
               "login", "/feed", "/tag/", "/author/", "mailto:", "tel:", "?share=", "/page/"]

EMBED_MODEL = os.getenv("EMBED_MODEL", "@cf/baai/bge-small-en-v1.5")
INDEX_NAME = os.getenv("VECTORIZE_INDEX", "watcher-kb")
MAX_PAGES = int(os.getenv("MAX_PAGES", "300"))
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "1.0"))
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 TheWatcherBot/2.0")

CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- chunks

@dataclass
class Chunk:
    source: str   # stable key for the document: "handbook:p26" or a URL
    title: str
    url: str
    text: str
    idx: int

    @property
    def id(self) -> str:
        return hashlib.sha1(f"{self.source}#{self.idx}".encode()).hexdigest()

    @property
    def hash(self) -> str:
        return hashlib.sha1(f"{self.title}\n{self.text}".encode()).hexdigest()[:16]

    @property
    def embed_text(self) -> str:
        return f"{self.title}\n{self.text}"

    def metadata(self) -> dict:
        return {"title": self.title[:200], "url": self.url, "source": self.source, "text": self.text}


def split_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware splitting with a little overlap so facts aren't cut in half."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[-•*\d]+[.)]?\s)", text) if p.strip()]
    pieces: list[str] = []
    for p in paras:  # break up giant paragraphs on sentence/line boundaries
        while len(p) > size:
            cut = max(p.rfind(". ", 0, size), p.rfind("\n", 0, size))
            cut = cut + 1 if cut > size // 3 else size
            pieces.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            pieces.append(p)

    chunks, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 1 > size:
            chunks.append(cur)
            tail = cur[-overlap:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = f"{tail}\n{piece}"
        else:
            cur = f"{cur}\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(re.findall(r"[A-Za-z]{2,}", c)) >= 8]


def handbook_chunks(path: Path = HANDBOOK) -> dict[str, list[Chunk]]:
    out: dict[str, list[Chunk]] = {}
    if not path.exists():
        log(f"! {path} missing, skipping handbook")
        return out
    for m in re.finditer(r"^## Page (\d+)\n(.*?)(?=^## Page |\Z)", path.read_text("utf-8"), re.S | re.M):
        page, body = m.group(1), m.group(2).strip()
        src = f"handbook:p{page}"
        title = f"ITU Student Handbook, page {page}"
        out[src] = [Chunk(src, title, "", t, i) for i, t in enumerate(split_text(body))]
    return out


# --------------------------------------------------------------------------- crawl

session = requests.Session()
session.headers["User-Agent"] = UA


def fetch(url: str, timeout: int = 30) -> requests.Response | None:
    for attempt in range(3):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (404, 410, 403):
                log(f"  {r.status_code} {url}")
                return None
        except requests.RequestException as e:
            log(f"  error {url}: {e}")
        time.sleep(2 * (attempt + 1))
    return None


def normalize(url: str) -> str:
    url, _ = urldefrag(url)
    p = urlparse(url)
    host = p.netloc.lower().removeprefix("www.")
    path = p.path if p.path.endswith("/") or "." in p.path.rsplit("/", 1)[-1] else p.path + "/"
    return f"https://{host}{path}" + (f"?{p.query}" if p.query and p.query.startswith("p=") else "")


def wanted(url: str, domain: str) -> bool:
    p = urlparse(url)
    if p.netloc.lower().removeprefix("www.") != domain:
        return False
    low = url.lower()
    if any(s in low for s in SKIP_IN_URL):
        return False
    return any(k in low for k in KEYWORDS)


def sitemap_urls(domain: str) -> list[str]:
    """WordPress sites usually publish a sitemap; that's far more complete than link-following."""
    found: list[str] = []
    todo = [f"https://{domain}/wp-sitemap.xml", f"https://{domain}/sitemap_index.xml",
            f"https://{domain}/sitemap.xml"]
    seen = set()
    while todo and len(seen) < 40:
        sm = todo.pop(0)
        if sm in seen:
            continue
        seen.add(sm)
        r = fetch(sm, timeout=20)
        if not r or b"<" not in r.content[:200]:
            continue
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError:
            continue
        for loc in root.iter():
            if loc.tag.endswith("loc") and loc.text:
                u = loc.text.strip()
                if u.endswith(".xml"):
                    if any(k in u for k in ("page", "post", "program", "faculty", "sitemap-")) and "user" not in u:
                        todo.append(u)
                else:
                    found.append(u)
        if found and sm.endswith(("wp-sitemap.xml", "sitemap_index.xml")) and not todo:
            break
    return found


def discover(seeds: list[str]) -> list[str]:
    domain = urlparse(seeds[0]).netloc.lower().removeprefix("www.")
    urls: dict[str, None] = {normalize(s): None for s in seeds}

    sm = [normalize(u) for u in sitemap_urls(domain)]
    sm = [u for u in sm if wanted(u, domain)]
    log(f"sitemap: {len(sm)} relevant urls")
    for u in sm:
        urls.setdefault(u)

    for seed in seeds:  # one level of link-following from each seed
        r = fetch(seed)
        if not r:
            continue
        soup = BeautifulSoup(r.content, "html.parser")
        for a in soup.find_all("a", href=True):
            u = normalize(urljoin(seed, a["href"]))
            if wanted(u, domain):
                urls.setdefault(u)

    if EXTRA_URLS.exists():
        for line in EXTRA_URLS.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.setdefault(normalize(line))

    # seeds and extra urls first, then everything else, capped
    return list(urls)[:MAX_PAGES]


NOISE = [".fusion-footer", ".fusion-header-wrapper", "#sliders-container", ".fusion-sliding-bar",
         ".fusion-page-title-bar", "#side-header", ".fusion-sharing-box", "header", "footer",
         "script", "style", "noscript", "iframe", "form", "nav", "svg", ".menu", ".breadcrumb",
         ".breadcrumbs", "#wpadminbar", ".screen-reader-text", ".cookie", ".popup", "aside"]


def page_text(html: bytes) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = re.split(r"\s[|\-–]\s", soup.title.string.strip())[0]
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(" ", strip=True)

    for pricing in soup.select(".fusion-pricing-table"):  # Avada fee tables
        heads = [h.get_text(" ", strip=True) for h in pricing.select(".panel-heading")]
        prices = [p.get_text(" ", strip=True) for p in pricing.select(".panel-body")]
        feats = [f.get_text(" ", strip=True) for f in pricing.select(".list-group-item")]
        pricing.replace_with(f"\n\nFee plans: {', '.join(heads)}\nPrices: {', '.join(prices)}\n"
                             f"Details: {', '.join(feats)}\n\n")

    for table in soup.find_all("table"):  # keep tables as readable rows
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        table.replace_with("\n\n" + "\n".join(rows) + "\n\n")

    for sel in NOISE:
        for el in soup.select(sel):
            el.decompose()

    root = (soup.find(id="main") or soup.find("main") or soup.find("article")
            or soup.find(id="content") or soup.body or soup)
    for br in root.find_all(["br"]):
        br.replace_with("\n")
    for blk in root.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "section"]):
        blk.insert_before("\n\n")
    raw = root.get_text("")
    lines = [re.sub(r"[ \t\xa0]+", " ", l).strip() for l in raw.splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return title, text


def website_chunks(dry_run: bool) -> tuple[dict[str, list[Chunk]], set[str], int]:
    seeds = [u.strip() for u in os.getenv("ITU_LINKS", "").split(",") if u.strip()] or DEFAULT_SEEDS
    urls = discover(seeds)
    log(f"crawling {len(urls)} pages")
    out: dict[str, list[Chunk]] = {}
    failed: set[str] = set()
    for i, url in enumerate(urls, 1):
        r = fetch(url)
        if r is None or "html" not in r.headers.get("content-type", "html"):
            failed.add(url)
            continue
        title, text = page_text(r.content)
        chunks = [Chunk(url, title or url, url, t, j) for j, t in enumerate(split_text(text))]
        out[url] = chunks
        log(f"[{i}/{len(urls)}] {len(chunks):>3} chunks  {url}")
        time.sleep(CRAWL_DELAY)
    return out, failed, len(urls)


def drop_boilerplate(docs: dict[str, list[Chunk]]) -> int:
    """Remove chunks whose text appears on many pages (menus, footers that slipped through)."""
    counts: dict[str, int] = {}
    for chunks in docs.values():
        for c in chunks:
            k = re.sub(r"\W+", "", c.text.lower())
            counts[k] = counts.get(k, 0) + 1
    dropped = 0
    for src, chunks in docs.items():
        keep = [c for c in chunks if counts[re.sub(r"\W+", "", c.text.lower())] < 3]
        dropped += len(chunks) - len(keep)
        docs[src] = [Chunk(c.source, c.title, c.url, c.text, i) for i, c in enumerate(keep)]
    return dropped


# --------------------------------------------------------------------------- cloudflare

class Cloudflare:
    def __init__(self, account: str, token: str, index: str = INDEX_NAME):
        self.base = f"https://api.cloudflare.com/client/v4/accounts/{account}"
        self.index = index
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"

    def _post(self, path: str, **kw) -> dict:
        for attempt in range(5):
            r = self.s.post(self.base + path, timeout=120, **kw)
            if r.status_code < 300:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:500]}")
        raise RuntimeError(f"{path} kept failing")

    def ensure_index(self) -> bool:
        """Create the Vectorize index on first run. Returns True if it was just created."""
        r = self.s.get(f"{self.base}/vectorize/v2/indexes/{self.index}", timeout=30)
        if r.status_code == 200:
            return False
        self._post("/vectorize/v2/indexes", json={
            "name": self.index, "description": "The Watcher knowledge base",
            "config": {"dimensions": 384, "metric": "cosine"},
        })
        log(f"created Vectorize index {self.index}")
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for i in range(0, len(texts), 50):
            res = self._post(f"/ai/run/{EMBED_MODEL}", json={"text": texts[i:i + 50]})
            vecs.extend(res["result"]["data"])
        return vecs

    def upsert(self, rows: list[dict]):
        for i in range(0, len(rows), 200):
            body = "\n".join(json.dumps(r) for r in rows[i:i + 200])
            self._post(f"/vectorize/v2/indexes/{self.index}/upsert",
                       data=body.encode(), headers={"Content-Type": "application/x-ndjson"})

    def delete(self, ids: list[str]):
        for i in range(0, len(ids), 500):
            self._post(f"/vectorize/v2/indexes/{self.index}/delete_by_ids", json={"ids": ids[i:i + 500]})


# --------------------------------------------------------------------------- sync

def plan(docs: dict[str, list[Chunk]], failed: set[str], old: dict[str, dict[str, str]]):
    """Work out what to embed and delete. Returns (to_embed, to_delete, new_manifest)."""
    new_manifest: dict[str, dict[str, str]] = {}
    to_embed: list[Chunk] = []
    for src, chunks in docs.items():
        prev = old.get(src, {})
        new_manifest[src] = {}
        for c in chunks:
            new_manifest[src][c.id] = c.hash
            if prev.get(c.id) != c.hash:
                to_embed.append(c)
    for src in failed:  # keep what we had for pages that failed this time
        if src in old and src not in new_manifest:
            new_manifest[src] = old[src]
    live = {i for ids in new_manifest.values() for i in ids}
    to_delete = sorted({i for ids in old.values() for i in ids} - live)
    return to_embed, to_delete, new_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="crawl and chunk only")
    ap.add_argument("--skip-web", action="store_true", help="handbook only")
    ap.add_argument("--full", action="store_true", help="re-embed everything")
    args = ap.parse_args()

    old = {} if args.full else (json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {})

    cf = None
    if not args.dry_run:
        acct, token = os.getenv("CF_ACCOUNT_ID"), os.getenv("CF_API_TOKEN")
        if not acct or not token:
            sys.exit("CF_ACCOUNT_ID and CF_API_TOKEN must be set (or use --dry-run)")
        cf = Cloudflare(acct, token)
        if cf.ensure_index():
            old = {}  # brand-new index: everything must be uploaded

    docs = handbook_chunks()
    log(f"handbook: {sum(len(v) for v in docs.values())} chunks")

    failed: set[str] = set()
    if not args.skip_web:
        web, failed, attempted = website_chunks(args.dry_run)
        ok = len(web)
        log(f"website: {ok}/{attempted} pages ok, {sum(len(v) for v in web.values())} chunks")
        if attempted and ok < max(3, attempted * 0.3):
            # The site is probably down or blocking us. Don't wipe good data.
            log("! too many pages failed; keeping all previous website data")
            failed |= {s for s in old if not s.startswith("handbook:")}
            web = {}
        docs.update(web)
    else:
        failed |= {s for s in old if not s.startswith("handbook:")}

    dropped = drop_boilerplate(docs)
    to_embed, to_delete, new_manifest = plan(docs, failed, old)
    total = sum(len(v) for v in new_manifest.values())
    log(f"boilerplate dropped: {dropped} | to embed: {len(to_embed)} | to delete: {len(to_delete)} | total: {total}")

    if args.dry_run:
        sample = next((c for cs in docs.values() for c in cs), None)
        if sample:
            log(f"\nsample chunk ({sample.source}):\n{sample.embed_text[:600]}")
        return

    assert cf is not None
    if to_embed:
        vecs = cf.embed([c.embed_text for c in to_embed])
        cf.upsert([{"id": c.id, "values": v, "metadata": c.metadata()} for c, v in zip(to_embed, vecs)])
    if to_delete:
        cf.delete(to_delete)

    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(new_manifest, indent=0, sort_keys=True))
    status = {
        "last_run": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": len(new_manifest), "chunks": total,
        "embedded": len(to_embed), "deleted": len(to_delete), "failed_pages": len(failed),
    }
    (ROOT / "data" / "last_refresh.json").write_text(json.dumps(status, indent=2) + "\n")
    log(json.dumps(status))

    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("### Knowledge refresh\n\n| | |\n|---|---|\n")
            for k, v in status.items():
                f.write(f"| {k} | {v} |\n")
            if failed:
                f.write("\n<details><summary>Failed pages</summary>\n\n" +
                        "\n".join(f"- {u}" for u in sorted(failed)) + "\n</details>\n")


if __name__ == "__main__":
    main()
