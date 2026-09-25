#!/usr/bin/env python3
"""Geology Study Hub ingestion. Python 3.11+. See README.md for setup.

Admin CLI only: never expose URL downloads or database credentials to students.
No scraping of YouTube pages, media downloads, or inferred/fabricated transcripts.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import logging
import math
import os
import random
import re
import sys
import time
import uuid
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit, urlunsplit

import httpx
from dotenv import load_dotenv
from pypdf import PdfReader

LOG = logging.getLogger("geology_ingest")
VERSION = "1.0.0"
DIMENSIONS = 384
LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
YOUTUBE_API = "https://www.googleapis.com/youtube/v3"
YOUTUBE_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
SUPPORTED_LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "Public-Domain",
                      "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0"}


class PipelineError(Exception):
    """A safe, user-facing error without connection strings or auth headers."""


class ApiError(PipelineError):
    def __init__(self, service: str, status: int, reason: str):
        self.status, self.reason = status, reason
        super().__init__(f"{service}: HTTP {status} ({reason})")


class QuotaError(ApiError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def source_id(kind: str, url: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"geology-hub:{kind}:{url}"))


def public_url(value: str) -> str:
    p = urlsplit(value)
    if p.scheme not in {"https", "http"} or not p.hostname or p.username or p.password:
        raise PipelineError("Use a public HTTP(S) URL without embedded credentials")
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path or "/", p.query, ""))


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def api_reason(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        reasons = error.get("errors", [])
        reason = reasons[0].get("reason") if reasons else error.get("code", "request_failed")
        return re.sub(r"[^a-zA-Z0-9_.-]", "", str(reason))[:80]
    except (ValueError, AttributeError, IndexError, TypeError):
        return "request_failed"


def retry_delay(attempt: int, response: httpx.Response | None = None) -> None:
    wait = min(16.0, 2 ** attempt + random.random())
    if response is not None:
        try:
            wait = min(30.0, max(wait, float(response.headers.get("retry-after", "0"))))
        except ValueError:
            pass
    time.sleep(wait)


def request(client: httpx.Client, method: str, url: str, *, service: str, **kwargs: Any) -> httpx.Response:
    """Bounded retries, including YouTube rate limits; daily quota errors stop."""
    for attempt in range(5):
        try:
            response = client.request(method, url, **kwargs)
        except httpx.TransportError:
            if attempt == 4:
                raise PipelineError(f"{service}: connection/timeout failed after five attempts") from None
            retry_delay(attempt)
            continue
        if response.is_success:
            return response
        reason = api_reason(response)
        if reason in {"quotaExceeded", "dailyLimitExceeded", "billing_disabled"}:
            raise QuotaError(service, response.status_code, reason)
        transient = response.status_code in {408, 429, 500, 502, 503, 504} or reason in {
            "rateLimitExceeded", "userRateLimitExceeded"
        }
        if transient and attempt < 4:
            retry_delay(attempt, response)
            continue
        raise ApiError(service, response.status_code, reason)
    raise PipelineError(f"{service}: request did not complete")


def download_pdf(client: httpx.Client, url: str, max_mb: int) -> bytes:
    limit = max_mb * 1024 * 1024
    for attempt in range(5):
        try:
            with client.stream("GET", public_url(url), follow_redirects=True) as response:
                if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < 4:
                    retry_delay(attempt, response)
                    continue
                if not response.is_success:
                    raise PipelineError(f"PDF download returned HTTP {response.status_code}")
                public_url(str(response.url))
                header_size = response.headers.get("content-length", "")
                if header_size.isdigit() and int(header_size) > limit:
                    raise PipelineError(f"PDF exceeds the {max_mb} MiB download limit")
                body = bytearray()
                for part in response.iter_bytes(65536):
                    body.extend(part)
                    if len(body) > limit:
                        raise PipelineError(f"PDF exceeds the {max_mb} MiB download limit")
                data = bytes(body)
                if b"%PDF-" not in data[:1024]:
                    raise PipelineError("URL did not return a PDF (possibly a login, landing page, or HTML error)")
                return data
        except httpx.TransportError:
            if attempt == 4:
                raise PipelineError("PDF download failed after five attempts") from None
            retry_delay(attempt)
    raise PipelineError("PDF download did not complete")


@dataclass
class Unit:
    text: str
    page: int | None = None
    start: float | None = None
    end: float | None = None


def bounded_text(text: str, count: Callable[[str], int], limit: int) -> list[str]:
    """Split while retaining source characters, not decoded/normalized tokens."""
    output: list[str] = []
    text = text.strip()
    while text:
        if count(text) <= limit:
            output.append(text)
            break
        lo, hi = 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count(text[:mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        if count(text[:lo]) > limit:
            raise PipelineError("Token limit is too small for one source character")
        # Prefer a word boundary; retain long formula/URL fragments if necessary.
        boundary = max(text.rfind(" ", 0, lo + 1), text.rfind("\n", 0, lo + 1))
        cut = boundary if boundary > lo // 2 else lo
        output.append(text[:cut].strip())
        text = text[cut:].strip()
    return output


def chunk_units(units: list[Unit], count: Callable[[str], int], limit: int, overlap: int) -> list[dict]:
    """Never cross a chapter; retain PDF page ranges or transcript cue intervals."""
    if limit < 16 or not 0 <= overlap < limit:
        raise PipelineError("Chunk limit must be >=16; overlap must be >=0 and below the limit")
    pieces = [replace(u, text=s) for u in units for s in bounded_text(u.text, count, limit)]
    chunks: list[dict] = []
    i = 0
    while i < len(pieces):
        end, text = i, ""
        while end < len(pieces):
            trial = "\n".join(p.text for p in pieces[i:end + 1])
            if count(trial) > limit:
                break
            text = trial
            end += 1
        if end == i:
            raise PipelineError("Chunker could not fit a source unit")
        window = pieces[i:end]
        pages = [u.page for u in window if u.page is not None]
        starts = [u.start for u in window if u.start is not None]
        ends = [u.end for u in window if u.end is not None]
        chunks.append({
            "content": text, "content_sha256": digest(text),
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "start_seconds": min(starts) if starts else None,
            "end_seconds": max(ends) if ends else None,
        })
        if end == len(pieces):
            break
        # Carry whole paragraphs/cues only, bounded by the overlap budget.
        nxt = end
        while nxt > i + 1 and count("\n".join(p.text for p in pieces[nxt - 1:end])) <= overlap:
            nxt -= 1
        # Ensure the retained tail leaves room for at least one new unit.
        while nxt < end and count("\n".join(p.text for p in pieces[nxt:end + 1])) > limit:
            nxt += 1
        i = nxt
    return chunks


def excluded_pages(spec: str, total: int) -> set[int]:
    result: set[int] = set()
    for item in filter(None, (p.strip() for p in spec.split(","))):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
        if not match:
            raise PipelineError("--exclude-pages must look like 1,4-6,22")
        start, end = int(match[1]), int(match[2] or match[1])
        if not 1 <= start <= end <= total:
            raise PipelineError("Excluded page range lies outside the PDF")
        result.update(range(start, end + 1))
    return result


def detect_chapters(reader: PdfReader, pages: list[str], manual: Path | None = None) -> list[dict]:
    starts: list[tuple[int, str]] = []
    method = "outline"
    if manual:
        method = "manual"
        for item in json.loads(manual.read_text(encoding="utf-8")):
            p, title = int(item["start_page"]), str(item["title"]).strip()
            if not title or not 1 <= p <= len(pages):
                raise PipelineError("Manual chapter start/title is invalid")
            starts.append((p, title))
        if not starts or len({p for p, _ in starts}) != len(starts):
            raise PipelineError("Manual chapters need distinct, valid start pages")
    else:
        candidates: list[tuple[int, int, str]] = []

        def walk(items: Iterable, depth: int = 0) -> None:
            for item in items:
                if isinstance(item, list):
                    walk(item, depth + 1)
                else:
                    try:
                        page = reader.get_destination_page_number(item)
                        title = clean_text(str(item.get("/Title", "")))
                        if page is not None and 0 <= page < len(pages) and title:
                            candidates.append((depth, page + 1, title))
                    except (ValueError, TypeError, KeyError, AttributeError):
                        continue
        try:
            walk(reader.outline)
        except Exception:
            LOG.warning("Unreadable PDF outline; using heading detection")
        if candidates:
            counts = Counter(d for d, _, _ in candidates)
            depth = min((d for d, n in counts.items() if n >= 2), default=min(counts))
            starts = [(p, title) for d, p, title in candidates if d == depth]
        if not starts:
            method = "heading"
            for number, page in enumerate(pages, 1):
                lines = [x.strip() for x in page.splitlines() if x.strip()][:8]
                for i, line in enumerate(lines):
                    if re.match(r"^chapter\s+(?:\d+|[IVXLCDM]+)\b", line, re.I) and "..." not in line:
                        if len(line) < 25 and i + 1 < len(lines):
                            line += " — " + lines[i + 1]
                        starts.append((number, line[:200]))
                        break
    if not starts:
        starts, method = [(1, "Document (no reliable chapter boundaries)")], "fallback"
    by_page: dict[int, str] = {}
    for page, title in sorted(starts):
        by_page.setdefault(page, title)
    if min(by_page) > 1:
        by_page[1] = "Front matter"
    ordered = sorted(by_page.items())
    return [{"ordinal": i, "title": title, "page_start": start,
             "page_end": ordered[i + 1][0] - 1 if i + 1 < len(ordered) else len(pages),
             "detection_method": method} for i, (start, title) in enumerate(ordered)]


def extract_pdf(data: bytes, *, max_pages: int = 2500, exclude: str = "",
                manual: Path | None = None, allow_partial: bool = False) -> tuple[list[dict], list[tuple[dict, list[Unit]]], dict]:
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise PipelineError("Encrypted PDF requires an unlocked, authorized copy")
        total = len(reader.pages)
        if not 1 <= total <= max_pages:
            raise PipelineError(f"PDF must contain 1..{max_pages} pages")
        omitted = excluded_pages(exclude, total)
        pages = [clean_text(p.extract_text() or "") if i not in omitted else ""
                 for i, p in enumerate(reader.pages, 1)]
        empty = [i for i, text in enumerate(pages, 1) if i not in omitted and len(text) < 20]
        usable = total - len(omitted)
        if usable == 0 or not any(pages):
            raise PipelineError("No extractable text; provide a text PDF or run OCR first")
        if len(empty) > max(3, usable * 0.2) and not allow_partial:
            raise PipelineError("Too many pages have little/no text; use OCR, or review and set --allow-partial")
        chapters = detect_chapters(reader, pages, manual)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"PDF extraction failed ({type(exc).__name__})") from None
    if empty:
        LOG.warning("%s PDF pages have little/no text; inspect extraction quality", len(empty))
    groups = []
    for chapter in chapters:
        units = [Unit(paragraph, page=n) for n in range(chapter["page_start"], chapter["page_end"] + 1)
                 for paragraph in pages[n - 1].split("\n\n") if paragraph.strip()]
        groups.append((chapter, units))
    return chapters, groups, {"page_count": total, "low_text_pages": empty,
                              "excluded_pages": sorted(omitted), "extractor": f"pypdf; pipeline {VERSION}"}


def caption_seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) not in {2, 3}:
        raise PipelineError("Invalid subtitle timestamp")
    numbers = list(map(float, parts))
    if any(not math.isfinite(n) or n < 0 for n in numbers) or any(n >= 60 for n in numbers[1:]):
        raise PipelineError("Invalid subtitle timestamp")
    return sum(n * 60 ** i for i, n in enumerate(reversed(numbers)))


def parse_transcript(text: str, suffix: str) -> list[dict]:
    try:
        return _parse_transcript(text, suffix)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        raise PipelineError("Malformed transcript: use VTT, SRT, or JSON timed segments") from None


def _parse_transcript(text: str, suffix: str) -> list[dict]:
    segments: list[dict] = []
    if suffix.lower() == ".json":
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = rows["segments"]
        for row in rows:
            start = float(row["start"])
            end = float(row["end"]) if "end" in row else start + float(row["duration"])
            segments.append({"start": start, "end": end, "text": clean_text(str(row["text"]))})
    else:
        lines = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        i = 0
        while i < len(lines):
            if "-->" not in lines[i]:
                i += 1
                continue
            sides = lines[i].split("-->", 1)
            start, end = caption_seconds(sides[0].strip()), caption_seconds(sides[1].strip().split()[0])
            i += 1
            body = []
            while i < len(lines) and lines[i].strip():
                body.append(lines[i])
                i += 1
            plain = html.unescape(re.sub(r"<[^>]+>", "", " ".join(body)))
            segments.append({"start": start, "end": end, "text": clean_text(plain)})
    result = []
    for s in segments:
        if not all(math.isfinite(s[k]) for k in ("start", "end")) or not 0 <= s["start"] <= s["end"]:
            raise PipelineError("Transcript contains an invalid time interval")
        if s["text"]:
            result.append(s)
    result.sort(key=lambda s: (s["start"], s["end"]))
    if not result:
        raise PipelineError("Transcript contains no usable timed text")
    return result


class Embedder:
    """Free local 384-dimensional embeddings; none mode needs no model download."""
    def __init__(self, backend: str, client: httpx.Client):
        self.backend, self.client = backend, client
        self.model = None
        self.model_id: str | None = None
        self.token_limit = 220
        if backend == "local":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise PipelineError("Install requirements-local.txt for local embeddings") from None
            revision = os.getenv("LOCAL_MODEL_REVISION") or None
            self.model = SentenceTransformer(LOCAL_MODEL, revision=revision, device="cpu", trust_remote_code=False)
            resolved = getattr(self.model[0].auto_model.config, "_commit_hash", None) or revision
            if not resolved:
                raise PipelineError("Could not resolve local model revision; set LOCAL_MODEL_REVISION to a commit")
            self.model_id = f"local:{LOCAL_MODEL}@{resolved}:384"
            self.token_limit = min(220, self.model.max_seq_length - 2)
        elif backend != "none":
            raise PipelineError("Unknown embedding backend")

    def count(self, text: str) -> int:
        if self.model is not None:
            return len(self.model.tokenizer.encode(text, add_special_tokens=False, truncation=False))
        # Export/index-only mode uses UTF-8 byte limits and needs no tokenizer.
        return len(text.encode("utf-8"))

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        if self.backend == "none":
            return [None] * len(texts)
        result = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            if self.model is not None:
                if any(self.count(t) > self.token_limit for t in batch):
                    raise PipelineError("Input exceeds local encoder limit; silent truncation prevented")
                vectors = self.model.encode(batch, batch_size=32, normalize_embeddings=True,
                                            show_progress_bar=False).tolist()
            if len(vectors) != len(batch):
                raise PipelineError("Embedding batch length mismatch")
            for vector in vectors:
                if len(vector) != DIMENSIONS or not all(math.isfinite(v) for v in vector):
                    raise PipelineError("Invalid embedding dimension or non-finite value")
                norm = math.sqrt(sum(v * v for v in vector))
                if norm <= 0:
                    raise PipelineError("Embedding must be nonzero")
                result.append([float(v / norm) for v in vector])
        return result


def base_source(kind: str, url: str, title: str, args: argparse.Namespace,
                *, metadata_only: bool = False, attribution: str | None = None) -> dict:
    url = public_url(url)
    rights = "metadata-only" if metadata_only else args.rights_basis
    code = args.license_code or ("YouTube-standard" if kind == "youtube" else "Unknown")
    if not metadata_only and rights == "open-license" and code not in SUPPORTED_LICENSES:
        raise PipelineError("This license needs review before indexing. Use metadata-only, or record separately obtained permission with --rights-basis permission")
    credit = args.attribution or attribution
    if not credit:
        raise PipelineError("Provide --attribution naming the author/institution and source")
    if not metadata_only and not args.license_code:
        raise PipelineError("Provide --license-code for material being indexed")
    return {"id": source_id(kind, url), "kind": kind, "canonical_url": url,
            "title": title.strip(), "description": "", "language": args.language,
            "license_code": code, "license_url": public_url(args.license_url) if args.license_url else None,
            "attribution": credit, "rights_basis": rights, "is_public": not args.private,
            "status": "metadata" if metadata_only else "indexed", "content_sha256": None,
            "fetched_at": utcnow(), "expires_at": (datetime.now(timezone.utc) + timedelta(days=29)).isoformat()
              if kind == "youtube" else None,
            "metadata": {"pipeline_version": VERSION}}


def make_pdf(args: argparse.Namespace, client: httpx.Client, embedder: Embedder) -> dict:
    kind = args.kind
    source = base_source(kind, args.canonical_url or args.url, args.title, args)
    if args.file:
        if args.file.stat().st_size > args.max_mb * 1024 * 1024:
            raise PipelineError("Local PDF exceeds --max-mb")
        data = args.file.read_bytes()
    else:
        data = download_pdf(client, args.url, args.max_mb)
    chapters, groups, metadata = extract_pdf(data, max_pages=args.max_pages, exclude=args.exclude_pages,
                                             manual=args.chapters, allow_partial=args.allow_partial)
    source["content_sha256"] = digest(data)
    source["metadata"].update(metadata)
    chunks = []
    for chapter, units in groups:
        chapter["id"] = str(uuid.uuid5(uuid.UUID(source["id"]), f"chapter:{chapter['ordinal']}"))
        for chunk in chunk_units(units, embedder.count, args.chunk_size, args.overlap):
            chunk["chapter_id"] = chapter["id"]
            chunks.append(chunk)
    if not chunks:
        raise PipelineError("PDF produced no text chunks")
    detail: dict = {"pdf_url": public_url(args.url), "page_count": metadata["page_count"]}
    if kind == "textbook":
        detail.update(authors=args.author, edition=args.edition)
    else:
        if not args.university or not args.course_code or not args.year:
            raise PipelineError("Exam PDFs require --university, --course-code and --year")
        detail.update(university=args.university, course_code=args.course_code, exam_year=args.year,
                      term=args.term, paper_type=args.paper_type)
    return {"source": source, "detail": detail, "chapters": chapters, "transcript": None,
            "chunks": chunks, "subjects": sorted(set(args.subject))}


def oauth_login(args: argparse.Namespace) -> None:
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        raise PipelineError("Install requirements-oauth.txt first") from None
    flow = InstalledAppFlow.from_client_secrets_file(str(args.client_secrets), scopes=[YOUTUBE_SCOPE])
    credentials = flow.run_local_server(port=0)
    save_private_text(args.token_file, credentials.to_json())
    LOG.info("YouTube OAuth credentials saved; keep the token file private")


def save_private_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        output.write(value)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


class YouTube:
    def __init__(self, client: httpx.Client, token_file: Path | None = None):
        self.client, self.token_file = client, token_file
        self.key = os.getenv("YOUTUBE_API_KEY")
        if not self.key:
            raise PipelineError("Set YOUTUBE_API_KEY for public video metadata")
        self.credentials = None
        if token_file:
            try:
                from google.oauth2.credentials import Credentials
            except ImportError:
                raise PipelineError("Install requirements-oauth.txt for caption downloads") from None
            self.credentials = Credentials.from_authorized_user_file(str(token_file), [YOUTUBE_SCOPE])

    def headers(self) -> dict:
        if self.credentials is None:
            raise PipelineError("Caption API needs --oauth-token from youtube-login")
        if not self.credentials.valid:
            from google.auth.transport.requests import Request
            self.credentials.refresh(Request())
            save_private_text(self.token_file, self.credentials.to_json())
        return {"Authorization": f"Bearer {self.credentials.token}"}

    def get(self, resource: str, params: dict, *, authorized: bool = False) -> dict:
        headers = self.headers() if authorized else {}
        if not authorized:
            params = {**params, "key": self.key}
        return request(self.client, "GET", f"{YOUTUBE_API}/{resource}", service="YouTube",
                       params=params, headers=headers).json()

    def videos(self, channel_id: str, max_videos: int) -> Iterable[dict]:
        channel = self.get("channels", {"part": "contentDetails", "id": channel_id})
        if not channel.get("items"):
            raise PipelineError("Channel ID does not resolve to an accessible YouTube channel")
        uploads = channel["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        token, seen = None, set()
        while True:
            params = {"part": "contentDetails", "playlistId": uploads, "maxResults": 50}
            if token:
                params["pageToken"] = token
            page = self.get("playlistItems", params)
            ids = []
            for item in page.get("items", []):
                video_id = item["contentDetails"]["videoId"]
                if video_id in seen:
                    continue
                if max_videos and len(seen) >= max_videos:
                    break
                seen.add(video_id)
                ids.append(video_id)
            if ids:
                response = self.get("videos", {"part": "snippet,contentDetails,status", "id": ",".join(ids)})
                by_id = {v["id"]: v for v in response.get("items", [])}
                for video_id in ids:
                    v = by_id.get(video_id)
                    # A public upload may become private/deleted between API calls.
                    if not v or v.get("status", {}).get("privacyStatus") != "public":
                        yield {"id": video_id, "unavailable": True}
                    else:
                        yield v
            token = page.get("nextPageToken")
            if not token or (max_videos and len(seen) >= max_videos):
                return

    def captions(self, video_id: str, languages: list[str]) -> tuple[dict | None, str]:
        try:
            tracks = self.get("captions", {"part": "snippet", "videoId": video_id}, authorized=True).get("items", [])
            available = [t for t in tracks if t["snippet"].get("status") == "serving"
                         and not t["snippet"].get("isDraft", False)]
            selected = None
            for language in languages:
                matches = [t for t in available if t["snippet"]["language"] == language
                           or t["snippet"]["language"].startswith(language + "-")]
                if matches:
                    selected = sorted(matches, key=lambda t: (t["snippet"]["language"] != language,
                                                              t["snippet"].get("trackKind") == "ASR"))[0]
                    break
            if not selected:
                return None, "language_unavailable" if available else "unavailable"
            response = request(self.client, "GET", f"{YOUTUBE_API}/captions/{selected['id']}", service="YouTube",
                               headers=self.headers(), params={"tfmt": "vtt"})
            segments = parse_transcript(response.text, ".vtt")
            return {"language": selected["snippet"]["language"], "origin": "youtube-api",
                    "track_id": selected["id"], "is_generated": selected["snippet"].get("trackKind") == "ASR",
                    "segments": segments, "text": "\n".join(s["text"] for s in segments),
                    "fetched_at": utcnow()}, "available"
        except QuotaError:
            raise
        except ApiError as exc:
            if exc.status in {401, 403}:
                return None, "not_authorized"
            if exc.status == 404:
                return None, "unavailable"
            raise


def duration_seconds(iso: str) -> float:
    match = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?", iso)
    if not match:
        raise PipelineError("Unrecognized YouTube duration")
    return sum(float(value or 0) * scale for value, scale in zip(match.groups(), [86400, 3600, 60, 1]))


def make_video(video: dict, args: argparse.Namespace, youtube: YouTube, embedder: Embedder) -> dict:
    video_id, snippet = video["id"], video["snippet"]
    transcript, state = None, "not_requested"
    if args.transcript_dir:
        for extension in (".vtt", ".srt", ".json"):
            path = args.transcript_dir / (video_id + extension)
            if path.is_file():
                if path.stat().st_size > 10 * 1024 * 1024:
                    raise PipelineError("Creator transcript exceeds 10 MiB")
                segments = parse_transcript(path.read_text(encoding="utf-8-sig"), extension)
                transcript = {"language": args.language, "origin": "creator-file", "track_id": None,
                              "is_generated": False, "segments": segments,
                              "text": "\n".join(s["text"] for s in segments), "fetched_at": utcnow()}
                state = "available"
                break
        if not transcript:
            state = "unavailable"
    if not transcript and args.oauth_token:
        transcript, state = youtube.captions(video_id, args.languages.split(","))
    url = f"https://www.youtube.com/watch?v={video_id}"
    source = base_source("youtube", url, snippet["title"], args, metadata_only=transcript is None,
                         attribution=f"{snippet['channelTitle']} — {url}")
    source["description"] = snippet.get("description", "")
    source["language"] = transcript["language"] if transcript else snippet.get("defaultLanguage", args.language)
    chunks = []
    if transcript:
        source["content_sha256"] = digest(transcript["segments"])
        units = [Unit(s["text"], start=s["start"], end=s["end"]) for s in transcript["segments"]]
        chunks = chunk_units(units, embedder.count, args.chunk_size, args.overlap)
    status = video["status"]
    detail = {"video_id": video_id, "channel_id": snippet["channelId"], "channel_title": snippet["channelTitle"],
              "published_at": snippet["publishedAt"], "duration_seconds": duration_seconds(video["contentDetails"]["duration"]),
              "embeddable": status.get("embeddable", False), "api_license": status.get("license", "youtube"),
              "transcript_status": state}
    return {"source": source, "detail": detail, "chapters": [], "transcript": transcript,
            "chunks": chunks, "subjects": sorted(set(args.subject))}


def prepare_package(package: dict, embedder: Embedder, args: argparse.Namespace) -> None:
    sid = package["source"]["id"]
    for i, chunk in enumerate(package["chunks"]):
        chunk.update(id=str(uuid.uuid5(uuid.UUID(sid), f"chunk:{i}:{chunk['content_sha256']}")),
                     ordinal=i, embedding_model=embedder.model_id, embedding=None)
        chunk.setdefault("chapter_id", None)
    source = {k: v for k, v in package["source"].items() if k not in {"fetched_at", "expires_at", "index_fingerprint"}}
    transcript = package["transcript"]
    stable_transcript = {k: v for k, v in transcript.items() if k != "fetched_at"} if transcript else None
    package["source"]["index_fingerprint"] = digest({
        "source": source, "detail": package["detail"], "subjects": package["subjects"],
        "chapters": package["chapters"], "transcript": stable_transcript, "chunks": package["chunks"],
        "chunk_size": args.chunk_size, "overlap": args.overlap,
    })


def embed_package(package: dict, embedder: Embedder, cached: dict[str, list[float]]) -> None:
    if embedder.model_id is None:
        return
    missing: dict[str, str] = {}
    for chunk in package["chunks"]:
        if chunk["content_sha256"] not in cached:
            missing[chunk["content_sha256"]] = chunk["content"]
    if missing:
        vectors = embedder.embed(list(missing.values()))
        cached.update(zip(missing, vectors))
    for chunk in package["chunks"]:
        chunk["embedding"] = cached[chunk["content_sha256"]]


class Database:
    def __init__(self, dsn: str):
        import psycopg
        from psycopg.conninfo import conninfo_to_dict, make_conninfo
        from psycopg.rows import dict_row
        if not dsn:
            raise PipelineError("Set DATABASE_URL, or choose --export-only for JSONL output")
        config = conninfo_to_dict(dsn)
        if config.get("host", "") not in {"localhost", "127.0.0.1", "::1"}:
            config.setdefault("sslmode", "require")
        self.conn = psycopg.connect(make_conninfo(**config), autocommit=True, connect_timeout=15,
                                    prepare_threshold=None, row_factory=dict_row,
                                    application_name="geology-ingest")

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *args: Any) -> None:
        self.conn.close()

    def cache(self, sid: str, model: str | None) -> dict:
        if model is None:
            return {}
        rows = self.conn.execute("select content_sha256,embedding::text as vector from public.content_chunks "
                                 "where source_id=%s and embedding_model=%s and embedding is not null", (sid, model))
        return {r["content_sha256"]: json.loads(r["vector"]) for r in rows}

    def upsert(self, table: str, row: dict) -> None:
        from psycopg import sql
        from psycopg.types.json import Jsonb
        columns = list(row)
        key = "id" if table == "sources" else "source_id"
        statement = sql.SQL("insert into public.{} ({}) values ({}) on conflict ({}) do update set {}") .format(
            sql.Identifier(table), sql.SQL(",").join(map(sql.Identifier, columns)),
            sql.SQL(",").join(sql.Placeholder() for _ in columns), sql.Identifier(key),
            sql.SQL(",").join(sql.SQL("{}=excluded.{}").format(sql.Identifier(c), sql.Identifier(c))
                              for c in columns if c != key))
        values = [Jsonb(row[c]) if isinstance(row[c], dict) or c == "segments" else row[c] for c in columns]
        self.conn.execute(statement, values)

    def save(self, package: dict) -> str:
        source, sid = package["source"], package["source"]["id"]
        with self.conn.transaction():
            self.conn.execute("select pg_advisory_xact_lock(hashtextextended(%s,0))", (sid,))
            existing = self.conn.execute("select index_fingerprint,fetched_at from public.sources where id=%s for update", (sid,)).fetchone()
            # A slower older worker must not overwrite a newer completed fetch.
            if existing and existing["fetched_at"] > datetime.fromisoformat(source["fetched_at"]):
                LOG.info("Skipped an older concurrent fetch for %s", sid)
                return "unchanged"
            unchanged = existing and existing["index_fingerprint"] == source["index_fingerprint"]
            self.upsert("sources", {**source, "updated_at": utcnow()})
            if not unchanged:
                self.conn.execute("delete from public.content_chunks where source_id=%s", (sid,))
                self.conn.execute("delete from public.chapters where source_id=%s", (sid,))
                self.conn.execute("delete from public.source_subjects where source_id=%s", (sid,))
                self.conn.execute("delete from public.video_transcripts where source_id=%s", (sid,))
                if package["detail"]:
                    table = {"textbook": "textbooks", "exam": "exam_papers", "youtube": "youtube_videos"}[source["kind"]]
                    self.upsert(table, {"source_id": sid, **package["detail"]})
                else:
                    from psycopg import sql
                    table = {"textbook": "textbooks", "exam": "exam_papers"}[source["kind"]]
                    self.conn.execute(sql.SQL("delete from public.{} where source_id=%s").format(sql.Identifier(table)), (sid,))
                for subject in package["subjects"]:
                    self.conn.execute("insert into public.source_subjects(source_id,subject_slug) values(%s,%s)", (sid, subject))
                for chapter in package["chapters"]:
                    self.conn.execute("insert into public.chapters(id,source_id,ordinal,title,page_start,page_end,detection_method) "
                                      "values(%s,%s,%s,%s,%s,%s,%s)",
                                      (chapter["id"], sid, chapter["ordinal"], chapter["title"], chapter["page_start"],
                                       chapter["page_end"], chapter["detection_method"]))
                if package["transcript"]:
                    self.upsert("video_transcripts", {"source_id": sid, **package["transcript"]})
                with self.conn.cursor() as cursor:
                    cursor.executemany("insert into public.content_chunks(id,source_id,chapter_id,ordinal,content,content_sha256,"
                        "page_start,page_end,start_seconds,end_seconds,embedding,embedding_model) "
                        "values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::extensions.vector,%s)", [
                            (c["id"], sid, c["chapter_id"], c["ordinal"], c["content"], c["content_sha256"],
                             c["page_start"], c["page_end"], c["start_seconds"], c["end_seconds"],
                             json.dumps(c["embedding"], allow_nan=False) if c["embedding"] is not None else None,
                             c["embedding_model"]) for c in package["chunks"]])
            elif package["transcript"]:
                self.conn.execute("update public.video_transcripts set fetched_at=%s where source_id=%s",
                                  (package["transcript"]["fetched_at"], sid))
            outcome = "unchanged" if unchanged else "written"
            self.conn.execute("insert into public.ingestion_runs(source_id,outcome,chunk_count) values(%s,%s,%s)",
                              (sid, outcome, len(package["chunks"])))
        return outcome

    def delete_video(self, video_id: str) -> None:
        sid = source_id("youtube", f"https://www.youtube.com/watch?v={video_id}")
        with self.conn.transaction():
            self.conn.execute("select pg_advisory_xact_lock(hashtextextended(%s,0))", (sid,))
            self.conn.execute("delete from public.sources where id=%s and kind='youtube'", (sid,))

    def purge(self) -> int:
        return self.conn.execute("select public.purge_expired_youtube() as n").fetchone()["n"]


def emit(package: dict, args: argparse.Namespace, embedder: Embedder, db: Database | None, output: Any) -> None:
    prepare_package(package, embedder, args)
    cached = db.cache(package["source"]["id"], embedder.model_id) if db else {}
    embed_package(package, embedder, cached)
    outcome = db.save(package) if db else "exported"
    if output:
        output.write(json.dumps(package, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()
    LOG.info("%s source=%s chunks=%s", outcome, package["source"]["id"], len(package["chunks"]))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--subject", action="append", default=[], help="Repeat with a seeded subject slug")
    common.add_argument("--language", default="en")
    common.add_argument("--license-code")
    common.add_argument("--license-url")
    common.add_argument("--attribution")
    common.add_argument("--rights-basis", choices=["open-license", "permission"], default="open-license")
    common.add_argument("--private", action="store_true")
    common.add_argument("--embedding", choices=["local", "none"], default=os.getenv("EMBEDDING_BACKEND", "none"))
    common.add_argument("--chunk-size", type=int, default=None, help="Local token limit; UTF-8 byte limit for other backends")
    common.add_argument("--overlap", type=int, default=32, help="Maximum overlap in the same units as chunk size")
    common.add_argument("--export", type=Path, help="Optional JSONL output (overwritten per run)")
    common.add_argument("--export-only", action="store_true", help="Do not connect to PostgreSQL; requires --export")

    pdf = sub.add_parser("pdf", parents=[common], help="Ingest a public/authorized textbook or exam PDF")
    pdf.add_argument("--url", required=True, help="Direct PDF URL used for page citations")
    pdf.add_argument("--canonical-url", help="Stable publisher landing URL; use consistently across reimports")
    pdf.add_argument("--file", type=Path, help="Use an already-downloaded local copy instead of HTTP")
    pdf.add_argument("--title", required=True)
    pdf.add_argument("--kind", choices=["textbook", "exam"], default="textbook")
    pdf.add_argument("--author", action="append", default=[])
    pdf.add_argument("--edition")
    pdf.add_argument("--university")
    pdf.add_argument("--course-code")
    pdf.add_argument("--year", type=int)
    pdf.add_argument("--term")
    pdf.add_argument("--paper-type", default="past-exam")
    pdf.add_argument("--max-mb", type=int, default=80)
    pdf.add_argument("--max-pages", type=int, default=2500)
    pdf.add_argument("--exclude-pages", default="")
    pdf.add_argument("--chapters", type=Path, help="Manual chapter-start JSON, using physical PDF page numbers")
    pdf.add_argument("--allow-partial", action="store_true")

    yt = sub.add_parser("youtube", parents=[common], help="Index a channel's uploads using YouTube Data API v3")
    yt.add_argument("--channel-id", default=os.getenv("YOUTUBE_CHANNEL_ID"))
    yt.add_argument("--max-videos", type=int, default=200, help="0 traverses the complete uploads playlist")
    yt.add_argument("--languages", default="en,en-GB,en-US")
    yt.add_argument("--oauth-token", type=Path)
    yt.add_argument("--transcript-dir", type=Path, help="Creator-supplied VIDEO_ID.vtt/.srt/.json files")

    link = sub.add_parser("catalog", parents=[common], help="Save a reference link without downloading/indexing its text")
    link.add_argument("--kind", choices=["textbook", "exam"], required=True)
    link.add_argument("--url", required=True)
    link.add_argument("--title", required=True)
    auth = sub.add_parser("youtube-login", help="Authorize caption access for an account allowed to edit the videos")
    auth.add_argument("--client-secrets", type=Path, required=True)
    auth.add_argument("--token-file", type=Path, default=Path("youtube_token.json"))
    sub.add_parser("purge-expired", help="Delete expired YouTube rows and derived data; schedule daily")
    sub.add_parser("subjects", help="Print valid subject slugs from PostgreSQL")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # HTTP request logs can include the YouTube API key in its query string.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    args = parser().parse_args(argv)
    try:
        if args.command == "youtube-login":
            oauth_login(args)
            return 0
        if args.command in {"purge-expired", "subjects"}:
            with Database(os.getenv("DATABASE_URL", "")) as db:
                if args.command == "purge-expired":
                    LOG.info("Removed %s expired YouTube sources", db.purge())
                else:
                    for row in db.conn.execute("select slug,name from public.subjects order by name"):
                        print(f"{row['slug']}\t{row['name']}")
            return 0
        if args.export_only and not args.export:
            raise PipelineError("--export-only requires --export PATH")
        metadata_only = args.command == "catalog" or (args.command == "youtube" and not args.oauth_token and not args.transcript_dir)
        backend = "none" if metadata_only else args.embedding
        if args.command == "youtube":
            if not args.channel_id or not re.fullmatch(r"UC[A-Za-z0-9_-]{22}", args.channel_id):
                raise PipelineError("Provide the exact 24-character UC... channel ID or set YOUTUBE_CHANNEL_ID")
            if args.max_videos < 0:
                raise PipelineError("--max-videos cannot be negative")
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
        db_context = nullcontext(None) if args.export_only else Database(os.getenv("DATABASE_URL", ""))
        output_context = args.export.open("w", encoding="utf-8") if args.export else nullcontext(None)
        with httpx.Client(timeout=httpx.Timeout(60, connect=15), headers={"User-Agent": f"GeologyStudyHub/{VERSION}"}) as client, db_context as db, output_context as output:
            embedder = Embedder(backend, client)
            args.chunk_size = args.chunk_size or (220 if backend == "local" else 1000)
            if backend == "local" and args.chunk_size > embedder.token_limit:
                raise PipelineError(f"Local chunk size must be <= {embedder.token_limit}")
            if args.chunk_size < 16 or not 0 <= args.overlap < args.chunk_size:
                raise PipelineError("Chunk size must be >=16 and 0 <= overlap < chunk size")
            if db and args.subject:
                known = {r["slug"] for r in db.conn.execute("select slug from public.subjects")}
                unknown = sorted(set(args.subject) - known)
                if unknown:
                    raise PipelineError("Unknown subject slugs: " + ", ".join(unknown))
            if args.command == "pdf":
                emit(make_pdf(args, client, embedder), args, embedder, db, output)
            elif args.command == "catalog":
                source = base_source(args.kind, args.url, args.title, args, metadata_only=True)
                emit({"source": source, "detail": None, "chapters": [], "transcript": None,
                      "chunks": [], "subjects": sorted(set(args.subject))}, args, embedder, db, output)
            else:
                youtube = YouTube(client, args.oauth_token)
                if db:
                    LOG.info("Removed %s expired YouTube sources", db.purge())
                failures, total = 0, 0
                for video in youtube.videos(args.channel_id, args.max_videos):
                    total += 1
                    if video.get("unavailable"):
                        if db:
                            db.delete_video(video["id"])
                        LOG.info("Unavailable video %s omitted", video["id"])
                        continue
                    try:
                        emit(make_video(video, args, youtube, embedder), args, embedder, db, output)
                    except QuotaError:
                        raise
                    except PipelineError as exc:
                        failures += 1
                        LOG.error("Video %s: %s", video["id"], exc)
                LOG.info("Channel finished: %s considered, %s failed", total, failures)
                return 2 if failures else 0
        return 0
    except PipelineError as exc:
        LOG.error("%s", exc)
        return 1
    except (KeyboardInterrupt, BrokenPipeError):
        return 130
    except Exception as exc:
        # Exception text from network/DB libraries may contain credentials/URLs.
        LOG.error("Failed (%s). Check configuration, connectivity, and source file format; secrets omitted.", type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
