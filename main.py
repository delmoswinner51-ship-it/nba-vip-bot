"""Telegram bot entry point.

Handles the /start command today. Designed so NBA prediction commands
and VIP-only features can be added under the same Application later.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import Forbidden, TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

ESPN_NBA_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
ESPN_NBA_STANDINGS_URL = (
    "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
)
ESPN_NBA_TEAM_SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule"
)
HTTP_TIMEOUT_SECONDS = 10.0

VIP_SUBSCRIBERS_PATH = Path(__file__).parent / "vip_subscribers.json"
VIP_LOCK = asyncio.Lock()
EDGE_PICK_THRESHOLD = 0.60

PICKS_HISTORY_PATH = Path(__file__).parent / "picks_history.json"
PICKS_HISTORY_LOCK = asyncio.Lock()

DAILY_PUSH_TIMEZONE = ZoneInfo("America/New_York")
DAILY_PUSH_TIME = time(hour=9, minute=0, tzinfo=DAILY_PUSH_TIMEZONE)
PICKS_RESOLVE_TIME = time(hour=23, minute=30, tzinfo=DAILY_PUSH_TIMEZONE)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


WELCOME_MESSAGE = (
    "Welcome to the NBA Predictions Bot.\n\n"
    "I'll soon serve daily NBA picks, model-driven predictions, and VIP-only "
    "premium plays.\n\n"
    "Available commands:\n"
    "/start - Show this welcome message\n"
    "/help - List available commands\n"
    "/predictions - Today's NBA games, live scores, and model picks\n"
    "/standings - Current East and West conference standings\n"
    "/team <name> - Team record, form, seed, and next game (e.g. /team Lakers)\n"
    "/vip - VIP-only edge picks for today's games (subscribers only)\n"
    "/subscribe - Join the VIP list to unlock edge picks\n"
    "/unsubscribe - Leave the VIP list\n"
    "/picks_history - Model performance: record, hit rate, ROI\n"
    "/last_n <number> - Recent form over the last N settled picks (default 10, max 50)\n"
    "/streak - Current win/loss streak plus longest streaks ever\n"
    "/best_picks <number> - Highest-probability picks ever and how they finished (default 10, max 25)"
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start: greet the user."""
    user = update.effective_user
    if user is None or update.message is None:
        return

    name = user.first_name or "there"
    logger.info("/start from user_id=%s username=%s", user.id, user.username)
    await update.message.reply_text(f"Hi {name}!\n\n{WELCOME_MESSAGE}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help: list commands."""
    if update.message is None:
        return
    await update.message.reply_text(WELCOME_MESSAGE)


async def fetch_nba_scoreboard() -> dict[str, Any]:
    """Fetch today's NBA scoreboard from ESPN's public endpoint."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(ESPN_NBA_SCOREBOARD_URL)
        response.raise_for_status()
        return response.json()


async def fetch_nba_scoreboard_for_date(date_yyyymmdd: str) -> dict[str, Any]:
    """Fetch the NBA scoreboard for a specific date (format YYYYMMDD)."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(
            ESPN_NBA_SCOREBOARD_URL, params={"dates": date_yyyymmdd}
        )
        response.raise_for_status()
        return response.json()


def _record_to_winpct(summary: str) -> float | None:
    """Parse a 'W-L' record string into a win percentage."""
    try:
        wins_str, losses_str = summary.split("-")
        wins = int(wins_str)
        losses = int(losses_str)
    except (ValueError, AttributeError):
        return None
    games = wins + losses
    if games == 0:
        return None
    return wins / games


def _record_by_type(competitor: dict[str, Any], type_name: str) -> str | None:
    for record in competitor.get("records") or []:
        if record.get("type") == type_name:
            summary = record.get("summary")
            if summary:
                return summary
    return None


def _predict(home: dict[str, Any], away: dict[str, Any]) -> dict[str, Any] | None:
    """Compute a win probability for the home team.

    Uses Log5 on splits:
      - home team's home win percentage
      - away team's road win percentage
    Falls back to overall records, then signals no prediction if insufficient.
    """
    home_pct = _record_to_winpct(_record_by_type(home, "home") or "")
    away_pct = _record_to_winpct(_record_by_type(away, "road") or "")
    basis = "home/road splits"

    if home_pct is None or away_pct is None:
        home_pct = _record_to_winpct(_record_by_type(home, "total") or "")
        away_pct = _record_to_winpct(_record_by_type(away, "total") or "")
        basis = "season records"
        if home_pct is None or away_pct is None:
            return None

    # Avoid degenerate 0% / 100% from tiny samples.
    home_pct = min(max(home_pct, 0.05), 0.95)
    away_pct = min(max(away_pct, 0.05), 0.95)

    # Log5: P(home beats away) given home_pct and away_pct (both vs avg opp).
    numerator = home_pct * (1 - away_pct)
    denominator = numerator + away_pct * (1 - home_pct)
    if denominator == 0:
        return None
    p_home = numerator / denominator

    home_team = (home.get("team") or {}).get("displayName") or "Home"
    away_team = (away.get("team") or {}).get("displayName") or "Away"
    pick_team = home_team if p_home >= 0.5 else away_team
    pick_prob = p_home if p_home >= 0.5 else (1 - p_home)

    edge = pick_prob - 0.5
    if edge < 0.05:
        confidence = "Coin flip"
    elif edge < 0.10:
        confidence = "Lean"
    elif edge < 0.18:
        confidence = "Pick"
    elif edge < 0.28:
        confidence = "Strong"
    else:
        confidence = "Lock"

    return {
        "pick": pick_team,
        "prob": pick_prob,
        "confidence": confidence,
        "basis": basis,
    }


def _format_event(event: dict[str, Any]) -> str:
    """Format a single ESPN event into a Telegram-friendly line."""
    competitions = event.get("competitions") or []
    if not competitions:
        return ""
    competition = competitions[0]
    competitors = competition.get("competitors") or []

    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return ""

    home_team = (home.get("team") or {}).get("displayName") or "Home"
    away_team = (away.get("team") or {}).get("displayName") or "Away"
    home_record = next(iter(home.get("records") or []), {}).get("summary", "")
    away_record = next(iter(away.get("records") or []), {}).get("summary", "")
    home_record_str = f" ({home_record})" if home_record else ""
    away_record_str = f" ({away_record})" if away_record else ""

    status = (event.get("status") or {}).get("type") or {}
    state = status.get("state", "pre")
    detail = status.get("shortDetail", "")

    matchup = f"{away_team}{away_record_str} @ {home_team}{home_record_str}"

    if state == "pre":
        try:
            start_dt = datetime.fromisoformat(
                event["date"].replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            tip = start_dt.strftime("%H:%M UTC")
        except (KeyError, ValueError):
            tip = detail or "TBD"
        line = f"• {matchup} — tips off {tip}"
        prediction = _predict(home, away)
        if prediction:
            line += (
                f"\n   Pick: *{prediction['pick']}* "
                f"({prediction['prob'] * 100:.1f}% — {prediction['confidence']})"
            )
        return line

    home_score = home.get("score", "0")
    away_score = away.get("score", "0")
    score = f"{away_score}-{home_score}"

    if state == "in":
        return f"• {matchup} — LIVE {score} ({detail})"
    if state == "post":
        return f"• {matchup} — Final {score}"
    return f"• {matchup} — {detail}"


async def fetch_nba_standings() -> dict[str, Any]:
    """Fetch current NBA standings from ESPN's public endpoint."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(ESPN_NBA_STANDINGS_URL)
        response.raise_for_status()
        return response.json()


def _stat(entry: dict[str, Any], name: str) -> str:
    for stat in entry.get("stats") or []:
        if stat.get("name") == name:
            value = stat.get("displayValue")
            if value is not None:
                return str(value)
    return ""


def _seed(entry: dict[str, Any]) -> int:
    raw = _stat(entry, "playoffSeed")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 99


def _format_conference(conference: dict[str, Any]) -> str:
    name = conference.get("name") or "Conference"
    entries = (conference.get("standings") or {}).get("entries") or []
    sorted_entries = sorted(entries, key=_seed)

    lines = [f"*{name}*"]
    for entry in sorted_entries:
        team_name = (entry.get("team") or {}).get("displayName") or "Team"
        seed = _stat(entry, "playoffSeed") or "-"
        wins = _stat(entry, "wins") or "0"
        losses = _stat(entry, "losses") or "0"
        last10 = _stat(entry, "Last Ten Games") or "—"
        streak = _stat(entry, "streak") or "—"
        lines.append(
            f"{seed:>2}. {team_name} — {wins}-{losses}  "
            f"(L10 {last10}, {streak})"
        )
    return "\n".join(lines)


async def standings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current NBA standings (seeds, records, last 10, streak) by conference."""
    if update.message is None:
        return

    user = update.effective_user
    logger.info(
        "/standings from user_id=%s username=%s",
        user.id if user else None,
        user.username if user else None,
    )

    try:
        data = await fetch_nba_standings()
    except (httpx.HTTPError, ValueError) as exc:
        logger.exception("Failed to fetch NBA standings: %s", exc)
        await update.message.reply_text(
            "Couldn't reach the NBA standings feed right now. Please try again in a moment."
        )
        return

    conferences = data.get("children") or []
    if not conferences:
        await update.message.reply_text("No standings data is available right now.")
        return

    season = ""
    for conf in conferences:
        season_name = (conf.get("standings") or {}).get("seasonDisplayName")
        if season_name:
            season = season_name
            break

    blocks: list[str] = []
    if season:
        blocks.append(f"*NBA Standings — {season}*")
    for conf in conferences:
        if conf.get("isConference"):
            blocks.append(_format_conference(conf))
    blocks.append(
        "_Seeds 1-6 are locked into the playoffs; 7-10 fight through the play-in._"
    )

    await update.message.reply_text(
        "\n\n".join(blocks),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


async def fetch_team_schedule(team_id: str) -> dict[str, Any]:
    """Fetch a single team's schedule from ESPN."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(
            ESPN_NBA_TEAM_SCHEDULE_URL.format(team_id=team_id)
        )
        response.raise_for_status()
        return response.json()


def _build_team_index(
    standings_data: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Return list of (team, entry, conference_name) from standings data."""
    index: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for conf in standings_data.get("children") or []:
        if not conf.get("isConference"):
            continue
        conf_name = conf.get("name") or ""
        for entry in (conf.get("standings") or {}).get("entries") or []:
            team = entry.get("team") or {}
            if team:
                index.append((team, entry, conf_name))
    return index


def _match_team(
    query: str,
    index: list[tuple[dict[str, Any], dict[str, Any], str]],
) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    """Find the best team match for a freeform query."""
    q = query.strip().lower()
    if not q:
        return None

    # Exact abbreviation match (e.g. "OKC", "LAL").
    for team, entry, conf in index:
        if (team.get("abbreviation") or "").lower() == q:
            return team, entry, conf

    # Exact match on common name fields.
    name_fields = ("displayName", "shortDisplayName", "name", "location", "nickname")
    for team, entry, conf in index:
        for field in name_fields:
            value = (team.get(field) or "").lower()
            if value and value == q:
                return team, entry, conf

    # Substring match (handles "lakers", "boston", "trail blazers", etc.).
    for team, entry, conf in index:
        for field in name_fields:
            value = (team.get(field) or "").lower()
            if value and (q in value or value in q):
                return team, entry, conf

    return None


def _format_next_game(
    schedule_data: dict[str, Any], team_id: str
) -> str | None:
    """Find and format the next non-completed game for the given team."""
    events = schedule_data.get("events") or []
    for event in events:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        status_type = (comp.get("status") or {}).get("type") or {}
        if status_type.get("completed"):
            continue

        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        home_name = (home.get("team") or {}).get("displayName") or "Home"
        away_name = (away.get("team") or {}).get("displayName") or "Away"
        own_is_home = (home.get("team") or {}).get("id") == team_id
        opponent = away_name if own_is_home else home_name
        location = "vs" if own_is_home else "@"

        try:
            start_dt = datetime.fromisoformat(
                event["date"].replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            when = start_dt.strftime("%a %b %d, %H:%M UTC")
        except (KeyError, ValueError):
            when = status_type.get("detail") or "TBD"

        return f"{location} {opponent} — {when}"
    return None


async def team(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Look up a single team's record, form, conference seed, and next game."""
    if update.message is None:
        return

    user = update.effective_user
    query = " ".join(context.args or []).strip()
    logger.info(
        "/team query=%r from user_id=%s username=%s",
        query,
        user.id if user else None,
        user.username if user else None,
    )

    if not query:
        await update.message.reply_text(
            "Usage: `/team <team name or abbreviation>`\n"
            "Examples: `/team Lakers`, `/team OKC`, `/team Trail Blazers`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    try:
        standings_data = await fetch_nba_standings()
    except (httpx.HTTPError, ValueError) as exc:
        logger.exception("Failed to fetch NBA standings: %s", exc)
        await update.message.reply_text(
            "Couldn't reach the NBA data feed right now. Please try again in a moment."
        )
        return

    index = _build_team_index(standings_data)
    match = _match_team(query, index)
    if not match:
        await update.message.reply_text(
            f"Couldn't find a team matching “{query}”. Try the full name "
            f"(e.g. Lakers) or 3-letter abbreviation (e.g. LAL)."
        )
        return

    team_obj, entry, conf_name = match
    team_id = str(team_obj.get("id") or "")
    team_name = team_obj.get("displayName") or "Team"
    seed = _stat(entry, "playoffSeed") or "-"
    wins = _stat(entry, "wins") or "0"
    losses = _stat(entry, "losses") or "0"
    win_pct = _stat(entry, "winPercent") or "—"
    last10 = _stat(entry, "Last Ten Games") or "—"
    streak = _stat(entry, "streak") or "—"
    home_record = _stat(entry, "Home") or "—"
    road_record = _stat(entry, "Road") or "—"
    differential = _stat(entry, "differential") or "—"

    lines = [
        f"*{team_name}*",
        f"{conf_name} — Seed #{seed}",
        "",
        f"Record: {wins}-{losses} ({win_pct})",
        f"Home: {home_record}   Road: {road_record}",
        f"Last 10: {last10}   Streak: {streak}",
        f"Point differential: {differential}",
    ]

    if team_id:
        try:
            schedule_data = await fetch_team_schedule(team_id)
            next_game = _format_next_game(schedule_data, team_id)
        except (httpx.HTTPError, ValueError) as exc:
            logger.exception("Failed to fetch team schedule: %s", exc)
            next_game = None

        lines.append("")
        if next_game:
            lines.append(f"*Next game:* {next_game}")
        else:
            lines.append("_No upcoming games on the schedule right now._")

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


async def predictions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show today's NBA games (matchups, tip-off times, and live scores)."""
    if update.message is None:
        return

    user = update.effective_user
    logger.info(
        "/predictions from user_id=%s username=%s",
        user.id if user else None,
        user.username if user else None,
    )

    try:
        data = await fetch_nba_scoreboard()
    except (httpx.HTTPError, ValueError) as exc:
        logger.exception("Failed to fetch NBA scoreboard: %s", exc)
        await update.message.reply_text(
            "Couldn't reach the NBA data feed right now. Please try again in a moment."
        )
        return

    events = data.get("events") or []
    if not events:
        await update.message.reply_text(
            "No NBA games are scheduled today. Check back tomorrow for the next slate."
        )
        return

    day = (data.get("day") or {}).get("date") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d"
    )

    lines = [f"*NBA Games — {day}*", ""]
    for event in events:
        formatted = _format_event(event)
        if formatted:
            lines.append(formatted)
    lines.append("")
    lines.append(
        "_Picks use a Log5 model on each team's home/road splits. "
        "VIPs will get sharper, model-backed edges first._"
    )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


async def _load_subscribers() -> set[int]:
    """Load the VIP subscriber set from disk."""
    if not VIP_SUBSCRIBERS_PATH.exists():
        return set()
    try:
        raw = VIP_SUBSCRIBERS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read VIP subscribers file: %s", exc)
        return set()
    if not isinstance(data, list):
        return set()
    return {int(uid) for uid in data if isinstance(uid, (int, str)) and str(uid).lstrip("-").isdigit()}


async def _save_subscribers(subscribers: set[int]) -> None:
    """Persist the VIP subscriber set to disk."""
    payload = json.dumps(sorted(subscribers))
    tmp_path = VIP_SUBSCRIBERS_PATH.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(VIP_SUBSCRIBERS_PATH)
    except OSError as exc:
        logger.error("Could not save VIP subscribers file: %s", exc)


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add the user to the VIP subscriber list."""
    if update.message is None or update.effective_user is None:
        return
    user = update.effective_user
    logger.info("/subscribe from user_id=%s username=%s", user.id, user.username)

    async with VIP_LOCK:
        subscribers = await _load_subscribers()
        if user.id in subscribers:
            await update.message.reply_text(
                "You're already on the VIP list. Use /vip to see today's edge picks."
            )
            return
        subscribers.add(user.id)
        await _save_subscribers(subscribers)

    await update.message.reply_text(
        "You're in. Welcome to VIP.\n\n"
        "Use /vip to see today's high-confidence edge picks. "
        "Use /unsubscribe to leave at any time."
    )


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove the user from the VIP subscriber list."""
    if update.message is None or update.effective_user is None:
        return
    user = update.effective_user
    logger.info("/unsubscribe from user_id=%s username=%s", user.id, user.username)

    async with VIP_LOCK:
        subscribers = await _load_subscribers()
        if user.id not in subscribers:
            await update.message.reply_text("You weren't on the VIP list.")
            return
        subscribers.discard(user.id)
        await _save_subscribers(subscribers)

    await update.message.reply_text(
        "You've been removed from the VIP list. Use /subscribe to rejoin any time."
    )


def _format_edge_pick(event: dict[str, Any], prediction: dict[str, Any]) -> str:
    """Format a single edge pick line for the VIP message."""
    competitions = event.get("competitions") or []
    if not competitions:
        return ""
    competition = competitions[0]
    competitors = competition.get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return ""

    home_team = (home.get("team") or {}).get("displayName") or "Home"
    away_team = (away.get("team") or {}).get("displayName") or "Away"
    matchup = f"{away_team} @ {home_team}"

    try:
        start_dt = datetime.fromisoformat(
            event["date"].replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        tip = start_dt.strftime("%H:%M UTC")
    except (KeyError, ValueError):
        tip = "TBD"

    return (
        f"• *{prediction['pick']}* over {matchup} — "
        f"{prediction['prob'] * 100:.1f}% ({prediction['confidence']}) — tips {tip}"
    )


async def _load_picks_history() -> list[dict[str, Any]]:
    """Load the persisted picks history. Returns empty list on first run / error."""
    if not PICKS_HISTORY_PATH.exists():
        return []
    try:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, PICKS_HISTORY_PATH.read_text)
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        logger.warning("Picks history file is not a list; ignoring.")
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load picks history")
    return []


async def _save_picks_history(history: list[dict[str, Any]]) -> None:
    """Persist picks history atomically."""
    tmp = PICKS_HISTORY_PATH.with_suffix(".json.tmp")
    payload = json.dumps(history, indent=2)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, tmp.write_text, payload)
    await loop.run_in_executor(None, os.replace, str(tmp), str(PICKS_HISTORY_PATH))


async def _record_edge_picks(
    date: str,
    edge_picks: list[tuple[dict[str, Any], dict[str, Any]]],
) -> int:
    """Record edge picks to history. Idempotent on (date, event_id). Returns count added."""
    if not edge_picks:
        return 0
    async with PICKS_HISTORY_LOCK:
        history = await _load_picks_history()
        existing = {(r.get("date"), r.get("event_id")) for r in history}
        added = 0
        for event, prediction in edge_picks:
            event_id = str(event.get("id") or "")
            if not event_id or (date, event_id) in existing:
                continue
            competition = (event.get("competitions") or [{}])[0]
            competitors = competition.get("competitors") or []
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            pick_name = prediction.get("pick")
            if (home.get("team") or {}).get("displayName") == pick_name:
                pick, opp = home, away
            else:
                pick, opp = away, home
            history.append(
                {
                    "date": date,
                    "event_id": event_id,
                    "pick_team_id": str((pick.get("team") or {}).get("id") or ""),
                    "pick_team": pick_name,
                    "opponent_team": (opp.get("team") or {}).get("displayName") or "",
                    "prob": prediction.get("prob"),
                    "confidence": prediction.get("confidence"),
                    "result": None,
                }
            )
            added += 1
        if added:
            await _save_picks_history(history)
        return added


async def resolve_pending_picks(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Settle any pending picks by checking final scores. Scheduled nightly."""
    async with PICKS_HISTORY_LOCK:
        history = await _load_picks_history()
        pending_idx_by_date: dict[str, list[int]] = {}
        for i, rec in enumerate(history):
            if rec.get("result") is None:
                date = rec.get("date")
                if date:
                    pending_idx_by_date.setdefault(date, []).append(i)

        if not pending_idx_by_date:
            logger.info("Resolve picks: nothing pending.")
            return

        resolved = 0
        for date, indices in pending_idx_by_date.items():
            try:
                date_compact = date.replace("-", "")
                data = await fetch_nba_scoreboard_for_date(date_compact)
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("Resolve picks: fetch failed for %s: %s", date, exc)
                continue
            events_by_id = {
                str(e.get("id") or ""): e for e in (data.get("events") or [])
            }
            for i in indices:
                rec = history[i]
                event = events_by_id.get(rec.get("event_id"))
                if not event:
                    continue
                state = ((event.get("status") or {}).get("type") or {}).get("state")
                if state != "post":
                    continue
                competition = (event.get("competitions") or [{}])[0]
                winner = next(
                    (
                        c
                        for c in (competition.get("competitors") or [])
                        if c.get("winner") is True
                    ),
                    None,
                )
                if not winner:
                    continue
                winner_id = str((winner.get("team") or {}).get("id") or "")
                rec["result"] = "W" if winner_id == rec.get("pick_team_id") else "L"
                resolved += 1

        if resolved:
            await _save_picks_history(history)
        logger.info("Resolve picks: resolved=%s", resolved)


async def picks_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show model performance: record, hit rate, and ROI assuming flat -110 bets."""
    if update.message is None:
        return

    async with PICKS_HISTORY_LOCK:
        history = await _load_picks_history()

    if not history:
        await update.message.reply_text(
            "*Picks History*\n\n"
            "No picks recorded yet. The model logs each day's edge picks at "
            "9:00 AM ET, then settles them after games end. Check back tomorrow.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    settled = [r for r in history if r.get("result") in ("W", "L")]
    pending = [r for r in history if r.get("result") is None]
    wins = sum(1 for r in settled if r.get("result") == "W")
    losses = len(settled) - wins
    total = len(settled)

    lines = ["*Picks History — Model Performance*", ""]
    if total == 0:
        lines.append(f"No settled picks yet ({len(pending)} pending).")
    else:
        hit_rate = wins / total * 100
        roi = (wins * 0.909 - losses) / total * 100
        lines.append(f"Record: *{wins}-{losses}* ({total} settled)")
        lines.append(f"Hit rate: *{hit_rate:.1f}%*")
        lines.append(f"ROI (flat -110 units): *{roi:+.1f}%*")
        if pending:
            lines.append(f"Pending: {len(pending)}")

        tier_order = ["Coin flip", "Lean", "Pick", "Strong", "Lock"]
        tier_stats: dict[str, dict[str, int]] = {
            t: {"w": 0, "l": 0} for t in tier_order
        }
        for r in settled:
            tier = r.get("confidence")
            if tier in tier_stats:
                key = "w" if r.get("result") == "W" else "l"
                tier_stats[tier][key] += 1

        active_tiers = [
            (t, tier_stats[t]) for t in tier_order if tier_stats[t]["w"] + tier_stats[t]["l"] > 0
        ]
        if active_tiers:
            lines.append("")
            lines.append("*By confidence tier*")
            for tier, s in active_tiers:
                w, l = s["w"], s["l"]
                n = w + l
                hr = w / n * 100
                tier_roi = (w * 0.909 - l) / n * 100
                lines.append(
                    f"• {tier}: *{w}-{l}* ({n}) — {hr:.1f}% hit, ROI {tier_roi:+.1f}%"
                )

    recent = list(reversed(settled[-10:]))
    if recent:
        lines.append("")
        lines.append("*Recent settled picks*")
        for r in recent:
            mark = "[W]" if r.get("result") == "W" else "[L]"
            prob = r.get("prob")
            prob_str = f"{prob * 100:.1f}%" if isinstance(prob, (int, float)) else "—"
            lines.append(
                f"{mark} {r.get('date')} — {r.get('pick_team')} vs "
                f"{r.get('opponent_team')} ({prob_str}, {r.get('confidence', '')})"
            )

    lines.append("")
    lines.append("_ROI assumes flat 1-unit bets at -110 odds (52.4% breakeven)._")

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


async def last_n_picks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the model's record over the last N settled picks (default 10, max 50)."""
    if update.message is None:
        return

    n = 10
    max_n = 50
    args = context.args or []
    if args:
        try:
            requested = int(args[0])
        except ValueError:
            await update.message.reply_text(
                "Usage: /last_n <number>  (e.g. /last_n 20). Max is 50."
            )
            return
        if requested <= 0:
            await update.message.reply_text("Please pass a positive number.")
            return
        n = min(requested, max_n)

    async with PICKS_HISTORY_LOCK:
        history = await _load_picks_history()

    settled = [r for r in history if r.get("result") in ("W", "L")]
    if not settled:
        await update.message.reply_text(
            "No settled picks yet. Once games finish and the nightly resolver runs, "
            "they'll show up here.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    recent = settled[-n:]
    actual_n = len(recent)
    wins = sum(1 for r in recent if r.get("result") == "W")
    losses = actual_n - wins
    hit_rate = wins / actual_n * 100
    roi = (wins * 0.909 - losses) / actual_n * 100

    label = f"Last {actual_n}"
    if actual_n < n:
        label = f"Last {actual_n} (only {actual_n} settled so far)"

    lines = [
        f"*Recent Form — {label} settled picks*",
        "",
        f"Record: *{wins}-{losses}*",
        f"Hit rate: *{hit_rate:.1f}%*",
        f"ROI (flat -110 units): *{roi:+.1f}%*",
        "",
        "*Picks (newest first)*",
    ]
    for r in reversed(recent):
        mark = "[W]" if r.get("result") == "W" else "[L]"
        prob = r.get("prob")
        prob_str = f"{prob * 100:.1f}%" if isinstance(prob, (int, float)) else "—"
        lines.append(
            f"{mark} {r.get('date')} — {r.get('pick_team')} vs "
            f"{r.get('opponent_team')} ({prob_str}, {r.get('confidence', '')})"
        )

    lines.append("")
    lines.append("_ROI assumes flat 1-unit bets at -110 odds (52.4% breakeven)._")

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


def _streaks(results: list[str]) -> tuple[tuple[str, int] | None, int, int]:
    """Compute current streak and longest W/L streaks from chronological results.

    Returns: ((current_kind, current_len) | None, longest_w, longest_l).
    """
    if not results:
        return None, 0, 0

    longest_w = longest_l = run = 0
    last = None
    for r in results:
        if r == last:
            run += 1
        else:
            run = 1
            last = r
        if r == "W":
            longest_w = max(longest_w, run)
        else:
            longest_l = max(longest_l, run)

    current_kind = results[-1]
    current_len = 0
    for r in reversed(results):
        if r == current_kind:
            current_len += 1
        else:
            break
    return (current_kind, current_len), longest_w, longest_l


async def streak(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the model's current streak and longest W/L streaks."""
    if update.message is None:
        return

    async with PICKS_HISTORY_LOCK:
        history = await _load_picks_history()

    settled = [r for r in history if r.get("result") in ("W", "L")]
    if not settled:
        await update.message.reply_text(
            "No settled picks yet. Once games finish and the nightly resolver runs, "
            "streak stats will show up here.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    results = [r["result"] for r in settled]
    current, longest_w, longest_l = _streaks(results)

    lines = ["*Model Streaks*", ""]
    if current:
        kind, length = current
        label = "win" if kind == "W" else "loss"
        plural = "" if length == 1 else "s"
        last = settled[-1]
        last_prob = last.get("prob")
        prob_str = (
            f"{last_prob * 100:.1f}%" if isinstance(last_prob, (int, float)) else "—"
        )
        lines.append(f"Current: *{length}-game {label} streak*")
        lines.append(
            f"Most recent: [{kind}] {last.get('date')} — {last.get('pick_team')} "
            f"vs {last.get('opponent_team')} ({prob_str}, {last.get('confidence', '')})"
        )
        lines.append("")

    lines.append(f"Longest win streak: *{longest_w}*")
    lines.append(f"Longest losing streak: *{longest_l}*")
    lines.append(f"Total settled picks: {len(settled)}")

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


async def best_picks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the highest-probability settled picks and how they finished."""
    if update.message is None:
        return

    n = 10
    max_n = 25
    args = context.args or []
    if args:
        try:
            requested = int(args[0])
        except ValueError:
            await update.message.reply_text(
                "Usage: /best_picks <number>  (e.g. /best_picks 15). Max is 25."
            )
            return
        if requested <= 0:
            await update.message.reply_text("Please pass a positive number.")
            return
        n = min(requested, max_n)

    async with PICKS_HISTORY_LOCK:
        history = await _load_picks_history()

    settled = [
        r
        for r in history
        if r.get("result") in ("W", "L") and isinstance(r.get("prob"), (int, float))
    ]
    if not settled:
        await update.message.reply_text(
            "No settled picks yet. Once games finish and the nightly resolver runs, "
            "the model's biggest swings will show up here.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    ranked = sorted(settled, key=lambda r: r["prob"], reverse=True)[:n]
    actual_n = len(ranked)
    wins = sum(1 for r in ranked if r["result"] == "W")
    losses = actual_n - wins
    hit_rate = wins / actual_n * 100
    roi = (wins * 0.909 - losses) / actual_n * 100

    overall_wins = sum(1 for r in settled if r["result"] == "W")
    overall_total = len(settled)
    overall_hit = overall_wins / overall_total * 100

    label = f"Top {actual_n}"
    if actual_n < n:
        label = f"Top {actual_n} (only {actual_n} settled so far)"

    lines = [
        f"*Best Picks — {label} by predicted probability*",
        "",
        f"Record: *{wins}-{losses}*",
        f"Hit rate: *{hit_rate:.1f}%*  (vs *{overall_hit:.1f}%* lifetime)",
        f"ROI (flat -110 units): *{roi:+.1f}%*",
        "",
        "*Picks (highest probability first)*",
    ]
    for r in ranked:
        mark = "[W]" if r["result"] == "W" else "[L]"
        prob_str = f"{r['prob'] * 100:.1f}%"
        lines.append(
            f"{mark} {prob_str} — {r.get('pick_team')} vs {r.get('opponent_team')} "
            f"({r.get('confidence', '')}, {r.get('date')})"
        )

    lines.append("")
    lines.append("_ROI assumes flat 1-unit bets at -110 odds (52.4% breakeven)._")

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


def _compute_edge_picks(
    scoreboard_data: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Filter today's upcoming games down to high-confidence edge picks."""
    edge_picks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for event in scoreboard_data.get("events") or []:
        competition = (event.get("competitions") or [{}])[0]
        state = ((event.get("status") or {}).get("type") or {}).get("state")
        if state != "pre":
            continue
        competitors = competition.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        prediction = _predict(home, away)
        if prediction and prediction["prob"] >= EDGE_PICK_THRESHOLD:
            edge_picks.append((event, prediction))
    edge_picks.sort(key=lambda pair: pair[1]["prob"], reverse=True)
    return edge_picks


def _build_vip_message(scoreboard_data: dict[str, Any], header_prefix: str = "") -> str:
    """Build the VIP edge picks message for today's slate."""
    edge_picks = _compute_edge_picks(scoreboard_data)
    day = (scoreboard_data.get("day") or {}).get("date") or datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")
    header = f"*{header_prefix}VIP Edge Picks — {day}*"

    if not edge_picks:
        return (
            f"{header}\n\n"
            "No high-confidence edges in today's slate. The model wants better than "
            "60% win probability before flagging a pick. Check back tomorrow."
        )

    lines = [header, ""]
    for event, prediction in edge_picks:
        line = _format_edge_pick(event, prediction)
        if line:
            lines.append(line)
    lines.append("")
    lines.append(
        "_For VIPs only. Picks use the same Log5 model on home/road splits, "
        "filtered to ≥60% win probability._"
    )
    return "\n".join(lines)


async def vip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show VIP-only edge picks for today's upcoming games."""
    if update.message is None or update.effective_user is None:
        return

    user = update.effective_user
    logger.info("/vip from user_id=%s username=%s", user.id, user.username)

    async with VIP_LOCK:
        subscribers = await _load_subscribers()

    if user.id not in subscribers:
        await update.message.reply_text(
            "*VIP Access Required*\n\n"
            "VIP unlocks today's highest-confidence edge picks — only the games "
            "where the model sees a real lean (≥60% win probability).\n\n"
            "Use /subscribe to join. It's free during the beta.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    try:
        data = await fetch_nba_scoreboard()
    except (httpx.HTTPError, ValueError) as exc:
        logger.exception("Failed to fetch NBA scoreboard: %s", exc)
        await update.message.reply_text(
            "Couldn't reach the NBA data feed right now. Please try again in a moment."
        )
        return

    await update.message.reply_text(
        _build_vip_message(data),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True,
    )


async def daily_vip_push(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send today's VIP edge picks to every subscriber.

    Scheduled via JobQueue to run once per day. If a subscriber has blocked
    the bot, prune them from the list so we don't keep retrying them.
    """
    async with VIP_LOCK:
        subscribers = await _load_subscribers()

    if not subscribers:
        logger.info("Daily VIP push: no subscribers, skipping.")
        return

    try:
        data = await fetch_nba_scoreboard()
    except (httpx.HTTPError, ValueError) as exc:
        logger.exception("Daily VIP push: failed to fetch scoreboard: %s", exc)
        return

    edge_picks = _compute_edge_picks(data)

    day = (data.get("day") or {}).get("date") or datetime.now(
        DAILY_PUSH_TIMEZONE
    ).strftime("%Y-%m-%d")
    try:
        added = await _record_edge_picks(day, edge_picks)
        if added:
            logger.info("Daily VIP push: recorded %s new pick(s) for %s", added, day)
    except OSError:
        logger.exception("Daily VIP push: failed to persist picks history")

    if not edge_picks:
        logger.info("Daily VIP push: no edge picks today, skipping send.")
        return

    message = _build_vip_message(data, header_prefix="Daily ")
    blocked: set[int] = set()
    sent = 0

    for user_id in subscribers:
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
            sent += 1
        except Forbidden:
            logger.info("Daily VIP push: user %s blocked the bot, pruning.", user_id)
            blocked.add(user_id)
        except TelegramError as exc:
            logger.warning("Daily VIP push: failed to message %s: %s", user_id, exc)

    if blocked:
        async with VIP_LOCK:
            current = await _load_subscribers()
            remaining = current - blocked
            if remaining != current:
                await _save_subscribers(remaining)

    logger.info(
        "Daily VIP push: sent=%s blocked=%s total_subscribers=%s",
        sent,
        len(blocked),
        len(subscribers),
    )


def build_application(token: str) -> Application:
    """Construct the Application and register handlers."""
    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("predictions", predictions))
    application.add_handler(CommandHandler("standings", standings))
    application.add_handler(CommandHandler("team", team))
    application.add_handler(CommandHandler("vip", vip))
    application.add_handler(CommandHandler("subscribe", subscribe))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe))
    application.add_handler(CommandHandler("picks_history", picks_history))
    application.add_handler(CommandHandler("last_n", last_n_picks))
    application.add_handler(CommandHandler("streak", streak))
    application.add_handler(CommandHandler("best_picks", best_picks))

    if application.job_queue is not None:
        application.job_queue.run_daily(
            daily_vip_push,
            time=DAILY_PUSH_TIME,
            name="daily_vip_push",
        )
        application.job_queue.run_daily(
            resolve_pending_picks,
            time=PICKS_RESOLVE_TIME,
            name="resolve_pending_picks",
        )
        logger.info(
            "Scheduled daily VIP push at %s %s and picks resolver at %s %s",
            DAILY_PUSH_TIME.strftime("%H:%M"),
            DAILY_PUSH_TIMEZONE.key,
            PICKS_RESOLVE_TIME.strftime("%H:%M"),
            DAILY_PUSH_TIMEZONE.key,
        )
    else:
        logger.warning(
            "JobQueue is unavailable; daily VIP push and picks resolver will not run. "
            "Install python-telegram-bot[job-queue] to enable scheduling."
        )

    return application


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error(
            "TELEGRAM_BOT_TOKEN is not set. Add it to Replit Secrets to start the bot."
        )
        sys.exit(1)

    application = build_application(token)
    logger.info("Starting Telegram bot (long polling)...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
