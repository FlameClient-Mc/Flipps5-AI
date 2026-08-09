"""Flipps V0.1 — tool access: web search, page reading, GitHub, YouTube, Telegram.

Tools work key-less where possible (DuckDuckGo search, GitHub public API,
direct page fetching). Optional API keys are read from environment variables
and light up the full integrations:

    GOOGLE_CSE_KEY / GOOGLE_CSE_CX   - real Google Search (Custom Search JSON API)
    YOUTUBE_API_KEY                  - YouTube Data API search
    GITHUB_TOKEN                     - higher GitHub API rate limits
    TELEGRAM_BOT_TOKEN               - send Telegram messages
"""

import html
import html.parser
import os
import re
import subprocess
import sys
import urllib.parse

import requests


def _load_env(path=".env"):
    """Load KEY=VALUE pairs from a local .env file (real env vars win)."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip('\"')
            if not value:
                continue  # leave empty keys unset
            os.environ.setdefault(key.strip(), value)


_load_env()

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FlippsV0.1/0.1"}


# --------------------------------------------------------------------------- #
# Web search
# --------------------------------------------------------------------------- #
def web_search(query, max_results=5):
    """Search the web. Uses Google Custom Search if keys are configured,
    otherwise a key-less DuckDuckGo search. Returns a list of result dicts."""
    cse_key = os.environ.get("GOOGLE_CSE_KEY")
    cse_cx = os.environ.get("GOOGLE_CSE_CX")
    if cse_key and cse_cx:
        try:
            return _google_search(query, cse_key, cse_cx, max_results)
        except Exception:
            pass  # fall through to DuckDuckGo
    return _duckduckgo_search(query, max_results)


def _google_search(query, key, cx, max_results):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": key, "cx": cx, "q": query, "num": min(max_results, 10)}
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", "Google CSE error"))
    return [
        {"title": item.get("title", ""), "url": item.get("link", ""),
         "snippet": item.get("snippet", "")}
        for item in data.get("items", [])
    ]


def _duckduckgo_search(query, max_results):
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    r = requests.post(url, data=params, headers=UA, timeout=15)
    r.raise_for_status()
    results = []
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r.text
    ):
        href = html.unescape(m.group(1))
        if href.startswith("https://duckduckgo.com/y.js"):  # sponsored/ad links
            continue
        title = re.sub(r"<[^>]+>", "", html.unescape(m.group(2))).strip()
        if not title:
            continue
        # Each result block also carries a summary in <a class="result__snippet">
        snippet = ""
        sm = re.search(
            r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
            r.text[m.end():],
        )
        if sm:
            snippet = re.sub(r"<[^>]+>", "", html.unescape(sm.group(1))).strip()
        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


# --------------------------------------------------------------------------- #
# Read a page / URL
# --------------------------------------------------------------------------- #
class _TextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self.skip += 1
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "pre", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def read_url(url, max_chars=6000):
    """Fetch a URL and return its readable text (HTML stripped)."""
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    parser = _TextExtractor()
    parser.feed(r.text)
    text = re.sub(r"\n{3,}", "\n\n", "".join(parser.parts))
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text[:max_chars]


# --------------------------------------------------------------------------- #
# GitHub
# --------------------------------------------------------------------------- #
def _gh_headers():
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "FlippsV0.1"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_search(query, max_results=5):
    """Search GitHub repositories. Public API, no key required."""
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "per_page": max_results}
    data = requests.get(url, params=params, headers=_gh_headers(), timeout=15).json()
    return [
        {"name": item.get("full_name", ""), "url": item.get("html_url", ""),
         "description": (item.get("description") or "")[:200],
         "stars": item.get("stargazers_count", 0), "language": item.get("language", "")}
        for item in data.get("items", [])
    ]


def github_repo(repo):
    """Get details about one repository, e.g. 'FlameClient-Mc/Flipps5-AI'."""
    url = f"https://api.github.com/repos/{repo}"
    data = requests.get(url, headers=_gh_headers(), timeout=15).json()
    if "full_name" not in data:
        return f"GitHub repo not found: {repo}"
    return (
        f"{data['full_name']} — {data.get('description') or 'no description'}\n"
        f"Language: {data.get('language')} | Stars: {data.get('stargazers_count')} | "
        f"Forks: {data.get('forks_count')} | URL: {data['html_url']}"
    )


# --------------------------------------------------------------------------- #
# YouTube
# --------------------------------------------------------------------------- #
def youtube_search(query, max_results=3):
    """Search YouTube. Uses the Data API if YOUTUBE_API_KEY is set, otherwise
    a web search scoped to youtube.com."""
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if api_key:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {"part": "snippet", "q": query, "type": "video",
                  "maxResults": max_results, "key": api_key}
        data = requests.get(url, params=params, timeout=15).json()
        return [
            {"title": i["snippet"]["title"],
             "url": f"https://www.youtube.com/watch?v={i['id']['videoId']}",
             "channel": i["snippet"]["channelTitle"]}
            for i in data.get("items", [])
        ]
    results = web_search(f"youtube {query}", max_results * 2)
    vids = [r for r in results if "youtube.com" in r["url"]][:max_results]
    return vids or results[:max_results]


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def telegram_send(chat_id, text):
    """Send a Telegram message using TELEGRAM_BOT_TOKEN. Returns status text."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return "Telegram not configured — set TELEGRAM_BOT_TOKEN (get one from @BotFather)."
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text[:4000]}, timeout=15)
    if r.status_code == 200:
        return f"Sent to Telegram chat {chat_id}."
    return f"Telegram error: {r.text[:200]}"


# --------------------------------------------------------------------------- #
# Twitter / Instagram
# --------------------------------------------------------------------------- #
def social_search(platform, query):
    """Twitter/X and Instagram have no key-less public search APIs. If the
    user adds API keys (X_API_BEARER / INSTAGRAM_TOKEN) this can light up;
    for now, fall back to a scoped web search."""
    if platform == "twitter" and os.environ.get("X_API_BEARER"):
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {"query": query, "max_results": 10}
        headers = {"Authorization": f"Bearer {os.environ['X_API_BEARER']}"}
        data = requests.get(url, params=params, headers=headers, timeout=15).json()
        return [t.get("text", "") for t in data.get("data", [])][:5]
    if platform == "instagram" and os.environ.get("INSTAGRAM_TOKEN"):
        return ("Instagram Graph API is available, but needs a business account "
                "token with the right permissions to search.")
    return web_search(f"site:{'twitter.com' if platform == 'twitter' else 'instagram.com'} {query}", 4)


# --------------------------------------------------------------------------- #
# Research: search + read the best pages
# --------------------------------------------------------------------------- #
def research(query, max_results=3):
    """Combine web search with reading top pages, for real research answers."""
    results = web_search(query, max_results)
    if not results:
        return "No search results found."
    chunks = [f"Search results for: {query}"]
    for i, r in enumerate(results, 1):
        chunks.append(f"{i}. {r['title']}\n   {r['url']}\n   {r.get('snippet', '')}")
    chunks.append("\n--- Page contents ---")
    for r in results:
        url = r["url"]
        try:
            text = read_url(url, max_chars=2500)
            chunks.append(f"### {r['title']} ({url})\n{text[:2500]}")
        except Exception:
            chunks.append(f"### {r['title']} ({url})\n[could not read page]")
    return "\n\n".join(chunks)


TOOL_HELP = """Flipps V0.1 tools — type any of these:
  search: <query>      web search (Google if keys set, else DuckDuckGo)
  research: <query>    search + read the top pages and synthesize
  youtube: <query>     find YouTube videos
  github: <query>      search GitHub repos
  repo: <owner/name>   details on one GitHub repo
  fetch: <url>         read the text of a web page
  run: <code>          EXECUTE Python code (prefix 'js ' for JavaScript)
  run: made/app.py     run a file you saved
  make: <file> <text>  save generated code/content to a file in made/
  game: <name>         make a game — snake, pong, tetris, voxel (mini-Minecraft)
  game: list           list all games I can make
  telegram: <chat_id> <text>   send a Telegram message (needs bot token)
  twitter: <query> / instagram: <query>   scoped web search
"""


# --------------------------------------------------------------------------- #
# Run / Make — execute code and save files
# --------------------------------------------------------------------------- #
_DANGEROUS = re.compile(
    r"\b(rm\s+-rf\s+/|format\s+[a-z]:|del\s+/[a-z]/|shutdown\s+-s|remove\s+-recurse\s+[a-z]:|mkfs)",
    re.I,
)


def _run_file(path, timeout=30):
    if path.endswith(".js"):
        cmd, label = ["node", os.path.abspath(path)], "JavaScript"
    else:
        cmd, label = [sys.executable, os.path.abspath(path)], "Python"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           cwd=os.path.dirname(os.path.abspath(path)) or None)
    except subprocess.TimeoutExpired:
        return f"Timed out after {timeout}s."
    out = "\n".join(x for x in (r.stdout, r.stderr) if x).strip()
    return f"[ran {label} file {path}]\n" + (out[:4000] or "(no output)")


def run_code(text, timeout=30):
    """Execute Python or JavaScript code (or an existing file) and return output.

    Usage: run: <code>           Python by default; prefix 'js ' for JavaScript
           run: made/foo.py      run a saved file
    """
    code = text.strip()
    lang = "py"
    if code.lower().startswith(("js ", "javascript ")):
        lang, code = "js", code.split(None, 1)[1].strip()
    elif code.lower().startswith(("py ", "python ")):
        code = code.split(None, 1)[1].strip()
    elif os.path.isfile(code):
        return _run_file(code, timeout)
    if not code:
        return "Nothing to run."
    if _DANGEROUS.search(code):
        return "Blocked: that looks destructive, I won't run it."
    try:
        if lang == "js":
            r = subprocess.run(["node", "-e", code], capture_output=True, text=True,
                               timeout=timeout)
        else:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"Timed out after {timeout}s."
    out = "\n".join(x for x in (r.stdout, r.stderr) if x).strip()
    return out[:4000] or "(no output)"


def make_file(filename, content):
    """Save generated content to a file inside the made/ folder.

    Usage: make: <filename> <content>
    """
    name = filename.replace("\\", "/").split("/")[-1].strip()
    if not name or name in {".", ".."}:
        return "Invalid filename."
    folder = os.path.join(os.getcwd(), "made")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return f"Saved {len(content)} chars to {path} — you can run it with: run: {os.path.join('made', name)}"


GAMES = {
    "snake": "Classic Snake — arrow keys, eat the food, grow (tkinter, no dependencies)",
    "pong": "Two-player Pong — W/S vs Up/Down arrows (tkinter, no dependencies)",
    "tetris": "Tetris — rotate and stack falling blocks (tkinter, no dependencies)",
    "voxel": "Voxel World — a 3D mini-Minecraft from scratch: procedural terrain, "
             "gravity physics, break and place blocks (pyglet + OpenGL)",
    "minecraft": "Alias for voxel — a mini-Minecraft from scratch",
}


def scaffold_game(name):
    """Create a playable game from a built-in template.

    Usage: game: <name>    (snake, pong, tetris, voxel / minecraft)
           game: list
    """
    key = name.strip().lower()
    if key == "list":
        return "I can make these games:\n" + "\n".join(
            f"  {k}: {v}" for k, v in GAMES.items())
    if key == "minecraft":
        key = "voxel"
    if key not in GAMES:
        return ("I can make these games: " + ", ".join(GAMES)
                + ". Try: game: snake")
    template = "voxel_world.py" if key == "voxel" else key + ".py"
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "templates", template)
    if not os.path.isfile(src):
        return f"Game template {template} is missing."
    folder = os.path.join(os.getcwd(), "made")
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, template)
    with open(src, encoding="utf-8") as fh:
        data = fh.read()
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(data)
    return (f"Created {template} in made/ — {GAMES[key]}\n"
            f"Run it with: run: {os.path.join('made', template)}")
