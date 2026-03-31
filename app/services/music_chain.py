from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from googleapiclient.discovery import build
from dotenv import load_dotenv
from typing import Any
import importlib
import os

load_dotenv()

INPUT_SONG_COUNT = 5
OUTPUT_LINK_COUNT = 3
SPOTIFY_SEARCH_LIMIT = 10

llm = ChatOpenAI(
    model="gpt-5.4-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.4,
)

prompt = PromptTemplate.from_template("""
You are a music trend curator who specializes in finding currently popular songs.

## Input Songs
{recent_songs}

## Step 1 — Weighted Taste Analysis
Analyze all input songs and count how often each attribute appears.
The more songs share an attribute, the higher its weight.

Extract and weight the following:
- genres: List all genres found. Mark frequently appearing ones as [HIGH], others as [LOW].
- mood: List all moods found. Mark frequently appearing ones as [HIGH], others as [LOW].
- vocal_style: Common vocal/rap preferences with weight.
- language_region: Primary language/region.

Weighting rule:
- [HIGH] attributes MUST appear in all 3 queries
- [LOW] attributes are optional, use only when it strengthens the query

## Step 2 — Build 3 Spotify Search Queries

**Query A — "Mood match recent"**
Goal: Songs released in 2025–2026 that most closely match the mood and atmosphere of the input songs.
Formula: [HIGH mood] [HIGH genres] atmospheric similar vibe [language_region] 2025 2026
Priority: Mood and atmosphere similarity over genre accuracy.

**Query B — "Popular now"**
Goal: Songs that are currently the most streamed and trending in the same genre, 2025–2026.
Formula: [HIGH genres] [HIGH vocal_style] viral chart topping most streamed [language_region] 2025 2026
Priority: Popularity and recency over mood accuracy.

**Query C — "All-time similar"**
Goal: Songs from any era that are most sonically and emotionally similar to the input songs.
Formula: [HIGH mood] [HIGH genres] [HIGH vocal_style] classic beloved all time [language_region]
Priority: Sonic and emotional similarity regardless of release year.

## Hard Rules
- Same artist as input songs is allowed — different songs only
- Never recommend the exact same song titles from the input
- Never use vague terms alone: "good", "best", "music", "song"
- Every query must contain at least 7 meaningful words
- Queries must be in English regardless of input language
- Query A and B must target 2025–2026 only
- Query C must have NO year restriction — cover all eras
- Output ONLY the JSON below — no explanation, no markdown, no extra text

{{
    "analysis": {{
        "genre": "<genres with weights>",
        "mood": "<mood with weights>",
        "tempo": "<tempo_feel>",
        "artist_style": "<vocal_style with weights>"
    }},
    "search_keywords": [
        "<Query A>",
        "<Query B>",
        "<Query C>"
    ]
}}
""")

parser = JsonOutputParser()
music_chain = prompt | llm | parser


def _format_songs(songs: list[dict]) -> str:
    return "\n".join(
        f"{i+1}. {s.get('title', '?')} - {s.get('artist', '?')}"
        for i, s in enumerate(songs)
    )


def _get_youtube_url(title: str, artist: str) -> str | None:
    youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
    response = youtube.search().list(
        q=f"{title} {artist}",
        part="id",
        type="video",
        maxResults=1,
    ).execute()

    items = response.get("items", [])
    if not items:
        return None

    video_id = items[0]["id"]["videoId"]
    return f"https://www.youtube.com/watch?v={video_id}"


def _get_spotify_client() -> Any | None:
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    try:
        spotipy_module = importlib.import_module("spotipy")
        oauth_module = importlib.import_module("spotipy.oauth2")
        credentials_class = getattr(oauth_module, "SpotifyClientCredentials")
    except Exception:
        return None

    credentials = credentials_class(client_id=client_id, client_secret=client_secret)
    return spotipy_module.Spotify(auth_manager=credentials)


def _normalize_recent_songs(recent_songs: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for song in recent_songs[:INPUT_SONG_COUNT]:
        title = str(song.get("title", "")).strip()
        artist = str(song.get("artist", "")).strip()
        if not title or not artist:
            continue
        normalized.append({"title": title, "artist": artist})
    return normalized


def _song_key(title: str, artist: str) -> tuple[str, str]:
    return title.strip().lower(), artist.strip().lower()


def _exclude_songs(recommendations: list[dict], exclude_list: list[dict]) -> list[dict]:
    exclude_keys = {
        _song_key(song.get("title", ""), song.get("artist", ""))
        for song in exclude_list
    }
    return [
        song for song in recommendations
        if _song_key(song.get("title", ""), song.get("artist", "")) not in exclude_keys
    ]


def _build_spotify_queries(llm_result: dict) -> list[str]:
    analysis = llm_result.get("analysis", {})
    genre = str(analysis.get("genre", "")).strip()
    mood = str(analysis.get("mood", "")).strip()
    tempo = str(analysis.get("tempo", "")).strip()
    artist_style = str(analysis.get("artist_style", "")).strip()

    def _make_specific_query(query: str) -> str:
        base_query = query.strip()
        if not base_query:
            return ""

        token_count = len(base_query.split())
        if token_count >= 6:
            return base_query

        extra_parts = [genre, mood, tempo, artist_style]
        extra_text = " ".join(part for part in extra_parts if part)
        if not extra_text:
            return base_query

        return f"{base_query} {extra_text}".strip()

    search_keywords = llm_result.get("search_keywords", [])
    queries = []
    for keyword in search_keywords:
        specific_query = _make_specific_query(str(keyword))
        if specific_query and specific_query not in queries:
            queries.append(specific_query)

    if queries:
        return queries[:OUTPUT_LINK_COUNT]

    fallback_query = " ".join(part for part in [genre, mood, tempo, artist_style] if part)
    return [fallback_query] if fallback_query else []


def _search_spotify_tracks(queries: list[str], exclude_songs: list[dict]) -> list[dict]:
    spotify = _get_spotify_client()
    if not spotify:
        return []

    exclude_keys = {
        _song_key(song.get("title", ""), song.get("artist", ""))
        for song in exclude_songs
    }

    seen_keys: set[tuple[str, str]] = set()
    tracks: list[dict] = []

    for query in queries:
        try:
            response = spotify.search(
                q=query,
                type="track",
                limit=SPOTIFY_SEARCH_LIMIT,
                market="KR",
            )
        except Exception:
            continue

        items = response.get("tracks", {}).get("items", [])
        for item in items:
            title = str(item.get("name", "")).strip()
            artists = item.get("artists", [])
            artist_names = [str(a.get("name", "")).strip() for a in artists if a.get("name")]
            artist = artist_names[0] if artist_names else ""
            if not title or not artist:
                continue

            key = _song_key(title, artist)
            if key in exclude_keys or key in seen_keys:
                continue

            seen_keys.add(key)
            tracks.append({"title": title, "artist": artist})
            if len(tracks) >= OUTPUT_LINK_COUNT:
                return tracks

    return tracks


def _extract_top3_links(recommendations: list[dict]) -> list[str]:
    links: list[str] = []
    for song in recommendations:
        youtube_url = _get_youtube_url(song.get("title", ""), song.get("artist", ""))
        if not youtube_url:
            continue
        links.append(youtube_url)
        if len(links) >= OUTPUT_LINK_COUNT:
            break
    return links


async def recommend_top3_youtube_links(recent_songs: list[dict]) -> list[str]:
    normalized_songs = _normalize_recent_songs(recent_songs)
    recent_songs_text = _format_songs(normalized_songs)
    if not recent_songs_text:
        recent_songs_text = "최근 신청곡 데이터 없음"

    llm_result = await music_chain.ainvoke({"recent_songs": recent_songs_text})
    queries = _build_spotify_queries(llm_result)
    spotify_tracks = _search_spotify_tracks(queries, normalized_songs)
    spotify_tracks = _exclude_songs(spotify_tracks, normalized_songs)

    return _extract_top3_links(spotify_tracks)