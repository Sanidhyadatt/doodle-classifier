from __future__ import annotations

from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
import hashlib
import hmac
import asyncio
import json
import pickle
import random
import secrets
import sqlite3

import numpy as np
import socketio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.ml.extractor import extract_features
from backend.ml.model_manager import NeuroModel
from backend.ml.pretrain import main as pretrain_main
from backend.ml.quickdraw_importer import fetch_quickdraw_images_for_class, get_quickdraw_classes
from backend.scraper.wiki import get_object_definition

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
MODEL_DIR = APP_ROOT / "models"
TRAINING_DATA_PATH = DATA_DIR / "training_data.pkl"
MODEL_PATH = MODEL_DIR / "neuro_model.pkl"


def get_user_data_path(user_id: int) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"user_{user_id}_training.pkl"


def get_user_model_path(user_id: int) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f"user_{user_id}_model.pkl"


DB_PATH = DATA_DIR / "neurosketch.db"
QUICKDRAW_CACHE_DIR = DATA_DIR / "quickdraw"
SESSION_COOKIE_NAME = "neuro_session"
SESSION_TTL_DAYS = 30
SESSION_TTL_SECONDS = SESSION_TTL_DAYS * 24 * 60 * 60
ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
ROUND_DURATION_SECONDS = 75
ROUND_PAUSE_SECONDS = 4
MAX_CHAT_HISTORY = 40
PROMPT_WORDS = [
    "cat",
    "house",
    "tree",
    "car",
    "bicycle",
    "cloud",
    "guitar",
    "rocket",
    "fish",
    "banana",
    "computer",
    "cup",
    "clock",
    "shoe",
    "chair",
]

fastapi_app = FastAPI(title="NeuroSketch API")
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=ALLOWED_ORIGINS)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

model_manager = NeuroModel()


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)
    display_name: str | None = Field(default=None, max_length=40)


class PredictRequest(BaseModel):
    pixel_data: list[list[float]]


class TrainClassRequest(BaseModel):
    class_name: str = Field(min_length=1, max_length=48)
    pixel_data: list[list[list[float]]]


class GuessFeedbackRequest(BaseModel):
    guess_id: int
    is_correct: bool


class QuickDrawTrainRequest(BaseModel):
    class_names: list[str]
    samples_per_class: int = Field(default=40, ge=10, le=120)


class LeaderboardItem(BaseModel):
    username: str
    display_name: str
    level: int
    xp: int
    coins: int
    streak: int


class DashboardPayload(BaseModel):
    user: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    missions: list[dict[str, Any]]
    achievements: list[dict[str, Any]]
    recent_activity: list[dict[str, Any]]
    correct_guesses: list[dict[str, Any]]
    model_status: dict[str, Any]


def get_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    with closing(get_db()) as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                xp INTEGER NOT NULL DEFAULT 0,
                coins INTEGER NOT NULL DEFAULT 0,
                level INTEGER NOT NULL DEFAULT 1,
                streak INTEGER NOT NULL DEFAULT 0,
                last_active_date TEXT,
                predictions_count INTEGER NOT NULL DEFAULT 0,
                training_count INTEGER NOT NULL DEFAULT 0,
                multiplayer_count INTEGER NOT NULL DEFAULT 0,
                wiki_lookups INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                detail TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS guess_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_correct INTEGER,
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS training_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                pixel_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    password_salt = salt or secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        password_salt.encode("utf-8"),
        120_000,
    ).hex()
    return password_salt, password_hash


def verify_password(password: str, salt: str, password_hash: str) -> bool:
    _, derived_hash = hash_password(password, salt)
    return hmac.compare_digest(derived_hash, password_hash)


def compute_level(xp: int) -> int:
    return max(1, xp // 100 + 1)


def load_training_data(user_id: int) -> tuple[list[list[float]], list[str]]:
    path = get_user_data_path(user_id)
    if not path.exists():
        return [], []

    with path.open("rb") as file_handle:
        data = pickle.load(file_handle)

    features = data.get("features", [])
    labels = data.get("labels", [])

    if not isinstance(features, list) or not isinstance(labels, list):
        raise ValueError("Invalid training data format")

    return features, labels


def save_training_data(user_id: int, features: list[list[float]], labels: list[str]) -> None:
    path = get_user_data_path(user_id)
    with path.open("wb") as file_handle:
        pickle.dump({"features": features, "labels": labels}, file_handle)


_model_cache: dict[int, NeuroModel] = {}
multiplayer_rooms: dict[str, dict[str, Any]] = {}
user_sid_index: dict[str, dict[str, Any]] = {}
room_round_tasks: dict[str, asyncio.Task[None]] = {}


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _mask_prompt(prompt: str) -> str:
    return " ".join("_" if char != " " else " " for char in prompt)


def _ensure_room(room_id: str) -> dict[str, Any]:
    if room_id in multiplayer_rooms:
        return multiplayer_rooms[room_id]

    room_state: dict[str, Any] = {
        "room_id": room_id,
        "phase": "waiting",
        "round_number": 0,
        "round_ends_at": None,
        "drawer_user_id": None,
        "drawer_sid": None,
        "prompt": "",
        "players": {},
        "player_order": [],
        "chat_history": [],
    }
    multiplayer_rooms[room_id] = room_state
    return room_state


def _build_room_state_payload(room: dict[str, Any], viewer_user_id: int) -> dict[str, Any]:
    drawer_user_id = room.get("drawer_user_id")
    players: dict[int, dict[str, Any]] = room.get("players", {})
    drawer = players.get(drawer_user_id) if drawer_user_id is not None else None
    is_drawer = drawer_user_id == viewer_user_id
    can_draw = room.get("phase") == "drawing" and is_drawer
    can_guess = room.get("phase") == "drawing" and not is_drawer

    prompt = room.get("prompt", "")
    visible_prompt = prompt if is_drawer else _mask_prompt(prompt)

    round_ends_at: datetime | None = room.get("round_ends_at")
    if room.get("phase") != "drawing" or round_ends_at is None:
        seconds_left = None
    else:
        seconds_left = max(0, int((round_ends_at - _utc_now()).total_seconds()))

    if room.get("phase") == "waiting":
        status = "Waiting for players"
    elif room.get("phase") == "drawing":
        drawer_name = drawer["display_name"] if drawer else "Unknown"
        status = f"{drawer_name} is drawing"
    else:
        status = "Round over"

    return {
        "room_id": room["room_id"],
        "phase": room["phase"],
        "round_number": room["round_number"],
        "round_ends_at": _iso_or_none(round_ends_at),
        "drawer_display_name": drawer["display_name"] if drawer else None,
        "is_drawer": is_drawer,
        "can_draw": can_draw,
        "can_guess": can_guess,
        "prompt": visible_prompt,
        "prompt_length": len(prompt),
        "seconds_left": seconds_left,
        "status": status,
        "players": [
            {
                "user_id": user_id,
                "username": player["username"],
                "display_name": player["display_name"],
                "score": int(player.get("score", 0)),
                "is_drawer": user_id == drawer_user_id,
                "has_guessed": bool(player.get("has_guessed", False)),
            }
            for user_id, player in players.items()
        ],
        "chat_history": room.get("chat_history", []),
        "self": {
            "user_id": viewer_user_id,
            "display_name": players.get(viewer_user_id, {}).get("display_name", ""),
            "username": players.get(viewer_user_id, {}).get("username", ""),
        },
    }


async def _emit_room_states(room_id: str) -> None:
    room = multiplayer_rooms.get(room_id)
    if room is None:
        return

    players: dict[int, dict[str, Any]] = room.get("players", {})
    for player in players.values():
        sid = player.get("sid")
        if not isinstance(sid, str):
            continue
        user_info = user_sid_index.get(sid)
        if not user_info:
            continue
        payload = _build_room_state_payload(room, int(user_info["user_id"]))
        await sio.emit("room_state", payload, to=sid)


def _append_room_history(room: dict[str, Any], event: dict[str, Any]) -> None:
    history = room.get("chat_history", [])
    history.append(event)
    if len(history) > MAX_CHAT_HISTORY:
        del history[: len(history) - MAX_CHAT_HISTORY]


def _cancel_room_timer(room_id: str) -> None:
    task = room_round_tasks.pop(room_id, None)
    if task and not task.done():
        task.cancel()


def _pick_next_drawer(room: dict[str, Any]) -> tuple[int, str] | None:
    order: list[int] = room.get("player_order", [])
    if not order:
        return None

    current_drawer = room.get("drawer_user_id")
    if current_drawer not in order:
        next_user_id = order[0]
    else:
        index = order.index(current_drawer)
        next_user_id = order[(index + 1) % len(order)]

    sid_for_drawer = room["players"][next_user_id]["sid"]
    return next_user_id, sid_for_drawer


async def _finish_round(room_id: str, reason: str, winner_user_id: int | None = None) -> None:
    room = multiplayer_rooms.get(room_id)
    if room is None:
        return

    room["phase"] = "round_over"
    room["round_ends_at"] = None

    if winner_user_id is not None:
        winner_name = room["players"].get(winner_user_id, {}).get("display_name", "Unknown")
        event = {
            "kind": "guess",
            "message": f"{winner_name} guessed correctly!",
            "guess": room.get("prompt", ""),
            "correct": True,
            "user": room["players"].get(winner_user_id),
        }
    else:
        event = {
            "kind": "round",
            "message": f"Round ended: {reason}",
            "guess": room.get("prompt", ""),
            "correct": False,
            "user": None,
        }

    _append_room_history(room, event)
    await sio.emit("room_event", event, room=room_id)
    await _emit_room_states(room_id)

    if len(room.get("player_order", [])) >= 1:
        await asyncio.sleep(ROUND_PAUSE_SECONDS)
        await _start_round(room_id)
    else:
        room["phase"] = "waiting"
        room["prompt"] = ""
        await _emit_room_states(room_id)


async def _round_timer(room_id: str, round_number: int) -> None:
    try:
        await asyncio.sleep(ROUND_DURATION_SECONDS)
    except asyncio.CancelledError:
        return

    room = multiplayer_rooms.get(room_id)
    if room is None:
        return

    if room.get("round_number") != round_number or room.get("phase") != "drawing":
        return

    await _finish_round(room_id, "time up")


async def _start_round(room_id: str) -> None:
    room = multiplayer_rooms.get(room_id)
    if room is None:
        return

    if len(room.get("player_order", [])) < 1:
        room["phase"] = "waiting"
        room["round_ends_at"] = None
        room["prompt"] = ""
        await _emit_room_states(room_id)
        return

    next_drawer = _pick_next_drawer(room)
    if next_drawer is None:
        return

    drawer_user_id, drawer_sid = next_drawer
    room["drawer_user_id"] = drawer_user_id
    room["drawer_sid"] = drawer_sid
    room["phase"] = "drawing"
    room["round_number"] = int(room.get("round_number", 0)) + 1
    room["round_ends_at"] = _utc_now() + timedelta(seconds=ROUND_DURATION_SECONDS)
    room["prompt"] = random.choice(PROMPT_WORDS)

    for player in room.get("players", {}).values():
        player["has_guessed"] = False

    event = {
        "kind": "round",
        "message": f"Round {room['round_number']} started",
        "guess": None,
        "correct": None,
        "user": room["players"].get(drawer_user_id),
    }
    _append_room_history(room, event)
    await sio.emit("room_event", event, room=room_id)
    await sio.emit("clear_canvas", room=room_id)
    await _emit_room_states(room_id)

    _cancel_room_timer(room_id)
    room_round_tasks[room_id] = asyncio.create_task(_round_timer(room_id, room["round_number"]))


def get_user_model(user_id: int) -> NeuroModel:
    if user_id in _model_cache:
        return _model_cache[user_id]

    model = NeuroModel()
    model_path = get_user_model_path(user_id)
    if model_path.exists():
        model.load(model_path)
    _model_cache[user_id] = model
    return model


def load_or_bootstrap_model() -> None:
    if MODEL_PATH.exists():
        model_manager.load(MODEL_PATH)
        return

    # If a model is missing, build the synthetic starter model so the platform works.
    pretrain_main()
    model_manager.load(MODEL_PATH)


def serialize_user(row: sqlite3.Row) -> dict[str, Any]:
    xp = int(row["xp"])
    level = int(row["level"])
    next_level_xp = level * 100
    current_level_xp = (level - 1) * 100
    progress = 0 if next_level_xp <= current_level_xp else max(0, xp - current_level_xp)
    progress_percent = 0 if next_level_xp <= current_level_xp else min(
        100,
        int((progress / (next_level_xp - current_level_xp)) * 100),
    )

    return {
        "id": int(row["id"]),
        "username": row["username"],
        "display_name": row["display_name"],
        "xp": xp,
        "coins": int(row["coins"]),
        "level": level,
        "streak": int(row["streak"]),
        "predictions_count": int(row["predictions_count"]),
        "training_count": int(row["training_count"]),
        "multiplayer_count": int(row["multiplayer_count"]),
        "wiki_lookups": int(row["wiki_lookups"]),
        "wins": int(row["wins"]),
        "losses": int(row["losses"]),
        "next_level_xp": next_level_xp,
        "level_progress": progress_percent,
        "achievements": build_achievements(row),
    }


def build_achievements(row: sqlite3.Row) -> list[dict[str, Any]]:
    level = int(row["level"])
    predictions = int(row["predictions_count"])
    training = int(row["training_count"])
    multiplayer = int(row["multiplayer_count"])
    wiki_lookups = int(row["wiki_lookups"])
    streak = int(row["streak"])
    wins = int(row["wins"])

    achievement_rules = [
        ("first-guess", "First Spark", "Make your first prediction.", predictions >= 1),
        ("trainer", "Sketch Coach", "Train your first custom class.", training >= 1),
        ("battle-ready", "Battle Ready", "Join a multiplayer room.", multiplayer >= 1),
        ("scholar", "Sketch Scholar", "Look up 5 object definitions.", wiki_lookups >= 5),
        ("level-five", "Level 5", "Reach level 5.", level >= 5),
        ("streak-five", "On Fire", "Maintain a 5-day streak.", streak >= 5),
        ("winner", "Arena Winner", "Win 3 multiplayer battles.", wins >= 3),
    ]

    return [
        {
            "id": rule_id,
            "title": title,
            "description": description,
            "earned": earned,
        }
        for rule_id, title, description, earned in achievement_rules
    ]


def build_missions(row: sqlite3.Row) -> list[dict[str, Any]]:
    predictions = int(row["predictions_count"])
    training = int(row["training_count"])
    multiplayer = int(row["multiplayer_count"])

    missions = [
        {
            "id": "prediction-run",
            "title": "Prediction Run",
            "description": "Make 5 predictions to earn bonus XP.",
            "progress": min(predictions, 5),
            "target": 5,
            "reward": "+25 XP",
        },
        {
            "id": "training-camp",
            "title": "Training Camp",
            "description": "Teach 1 custom object to level up fast.",
            "progress": min(training, 1),
            "target": 1,
            "reward": "+60 XP",
        },
        {
            "id": "battle-wave",
            "title": "Battle Wave",
            "description": "Join 1 multiplayer room and start a sketch duel.",
            "progress": min(multiplayer, 1),
            "target": 1,
            "reward": "+20 XP",
        },
    ]
    return missions


def get_user_by_session_token(token: str | None) -> sqlite3.Row | None:
    if not token:
        return None

    now_iso = datetime.now(tz=UTC).isoformat()
    with closing(get_db()) as conn:
        row = conn.execute(
            """
            SELECT users.*
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ? AND sessions.expires_at > ?
            """,
            (token, now_iso),
        ).fetchone()
    return row


def extract_token_from_cookie_header(cookie_header: str | None) -> str | None:
    if not cookie_header:
        return None
    cookies = SimpleCookie()
    cookies.load(cookie_header)
    morsel = cookies.get(SESSION_COOKIE_NAME)
    return morsel.value if morsel else None


def get_current_user(request: Request) -> sqlite3.Row:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    row = get_user_by_session_token(token)
    if row is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return row


def create_session_for_user(user_id: int) -> tuple[str, datetime]:
    token = secrets.token_urlsafe(32)
    created_at = datetime.now(tz=UTC)
    expires_at = created_at + timedelta(seconds=SESSION_TTL_SECONDS)

    with closing(get_db()) as conn:
        conn.execute(
            "INSERT INTO sessions(token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, created_at.isoformat(), expires_at.isoformat()),
        )
        conn.commit()

    return token, expires_at


def delete_session(token: str | None) -> None:
    if not token:
        return
    with closing(get_db()) as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


def log_activity(user_id: int, kind: str, detail: str) -> None:
    with closing(get_db()) as conn:
        conn.execute(
            """
            INSERT INTO activity_log(user_id, kind, detail, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, kind, detail, datetime.now(tz=UTC).isoformat()),
        )
        conn.commit()


def get_recent_activity(user_id: int, limit: int = 8) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT kind, detail, created_at
            FROM activity_log
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return [
        {"kind": row["kind"], "detail": row["detail"], "created_at": row["created_at"]}
        for row in rows
    ]


def add_guess_history(user_id: int, prediction: str, confidence: float) -> int:
    with closing(get_db()) as conn:
        cursor = conn.execute(
            """
            INSERT INTO guess_history(user_id, prediction, confidence, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, prediction, confidence, datetime.now(tz=UTC).isoformat()),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_correct_guesses(user_id: int, limit: int = 10) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT id, prediction, confidence, created_at
            FROM guess_history
            WHERE user_id = ? AND is_correct = 1
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return [
        {
            "guess_id": int(row["id"]),
            "prediction": row["prediction"],
            "confidence": float(row["confidence"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def update_user_progress(
    user_id: int,
    *,
    xp_gain: int = 0,
    coins_gain: int = 0,
    count_increments: dict[str, int] | None = None,
) -> sqlite3.Row:
    count_increments = count_increments or {}
    with closing(get_db()) as conn:
        conn.execute("BEGIN")
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="User not found")

        today_iso = date.today().isoformat()
        yesterday_iso = (date.today() - timedelta(days=1)).isoformat()
        last_active = row["last_active_date"]
        streak = int(row["streak"])
        if last_active == today_iso:
            new_streak = streak
        elif last_active == yesterday_iso:
            new_streak = streak + 1
        else:
            new_streak = 1

        new_xp = int(row["xp"]) + xp_gain
        new_level = compute_level(new_xp)
        new_coins = int(row["coins"]) + coins_gain

        updates: dict[str, Any] = {
            "xp": new_xp,
            "coins": new_coins,
            "level": new_level,
            "streak": new_streak,
            "last_active_date": today_iso,
        }

        for field_name, delta in count_increments.items():
            updates[field_name] = int(row[field_name]) + delta

        assignments = ", ".join(f"{field} = ?" for field in updates)
        values = list(updates.values()) + [user_id]
        conn.execute(f"UPDATE users SET {assignments} WHERE id = ?", values)
        conn.commit()
        updated_row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if updated_row is None:
            raise HTTPException(status_code=500, detail="Failed to update user stats")
        return updated_row


def get_leaderboard(limit: int = 5) -> list[dict[str, Any]]:
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM users
            ORDER BY xp DESC, coins DESC, level DESC, id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "username": row["username"],
            "display_name": row["display_name"],
            "level": int(row["level"]),
            "xp": int(row["xp"]),
            "coins": int(row["coins"]),
            "streak": int(row["streak"]),
        }
        for row in rows
    ]


def build_dashboard_payload(user_row: sqlite3.Row) -> dict[str, Any]:
    user_id = int(user_row["id"])
    user_model = get_user_model(user_id)
    return {
        "user": serialize_user(user_row),
        "leaderboard": get_leaderboard(),
        "missions": build_missions(user_row),
        "achievements": build_achievements(user_row),
        "recent_activity": get_recent_activity(user_id),
        "correct_guesses": get_correct_guesses(user_id),
        "model_status": {
            "ready": user_model.model is not None,
            "classes": list(getattr(user_model.model, "classes_", [])) if user_model.model is not None else [],
        },
    }


def register_or_login(username: str, password: str, display_name: str | None = None) -> sqlite3.Row:
    with closing(get_db()) as conn:
        existing = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if existing is not None:
            raise HTTPException(status_code=409, detail="Username already exists")

        password_salt, password_hash = hash_password(password)
        user_display_name = display_name.strip() if display_name and display_name.strip() else username
        conn.execute(
            """
            INSERT INTO users(
                username, display_name, password_salt, password_hash, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                username,
                user_display_name,
                password_salt,
                password_hash,
                datetime.now(tz=UTC).isoformat(),
            ),
        )
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user is None:
            raise HTTPException(status_code=500, detail="Failed to create user")
        return user


def authenticate_user(username: str, password: str) -> sqlite3.Row:
    with closing(get_db()) as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        if not verify_password(password, user["password_salt"], user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")

        return user


def build_response_with_cookie(payload: dict[str, Any], token: str, expires_at: datetime) -> JSONResponse:
    response = JSONResponse(payload)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
        max_age=SESSION_TTL_SECONDS,
        expires=expires_at,
    )
    return response


@fastapi_app.on_event("startup")
def startup_event() -> None:
    initialize_database()


@fastapi_app.delete("/training_data/{class_name}")
def delete_training_data_class(
    class_name: str,
    current_user: sqlite3.Row = Depends(get_current_user),
) -> dict[str, Any]:
    user_id = int(current_user["id"])
    class_name = class_name.strip()

    try:
        with closing(get_db()) as conn:
            conn.execute(
                "DELETE FROM training_samples WHERE user_id = ? AND class_name = ?",
                (user_id, class_name),
            )
            conn.commit()

            existing_features, existing_labels = load_training_data(user_id)

            new_features = []
            new_labels = []
            for feat, label in zip(existing_features, existing_labels):
                if label != class_name:
                    new_features.append(feat)
                    new_labels.append(label)

            save_training_data(user_id, new_features, new_labels)

            user_model = NeuroModel()
            unique_classes = sorted(set(new_labels))

            if len(unique_classes) >= 2:
                user_model.train(new_features, new_labels)
                user_model.save(get_user_model_path(user_id))
                message = f"Class {class_name} removed. Model retrained."
            else:
                model_path = get_user_model_path(user_id)
                if model_path.exists():
                    model_path.unlink()
                message = f"Class {class_name} removed. Add more classes."

            if user_id in _model_cache:
                del _model_cache[user_id]

            log_activity(
                user_id,
                "training_remove",
                f"Removed the class {class_name} and its samples.",
            )

            updated_user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if updated_user is None:
                raise HTTPException(status_code=500, detail="User state error")

            return {
                "message": message,
                "class_removed": class_name,
                "dashboard": build_dashboard_payload(updated_user),
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to remove class: {exc}") from exc


@fastapi_app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@fastapi_app.post("/auth/register")
def register(payload: AuthRequest) -> JSONResponse:
    user = register_or_login(payload.username.strip(), payload.password, payload.display_name)
    token, expires_at = create_session_for_user(int(user["id"]))
    log_activity(int(user["id"]), "register", "Opened a new NeuroSketch account.")
    updated_user = update_user_progress(int(user["id"]), xp_gain=15, coins_gain=5)
    return build_response_with_cookie(
        {
            "message": "Account created",
            "user": serialize_user(updated_user),
            "dashboard": build_dashboard_payload(updated_user),
        },
        token,
        expires_at,
    )


@fastapi_app.post("/auth/login")
def login(payload: AuthRequest) -> JSONResponse:
    user = authenticate_user(payload.username.strip(), payload.password)
    token, expires_at = create_session_for_user(int(user["id"]))
    log_activity(int(user["id"]), "login", "Signed in to NeuroSketch.")
    updated_user = update_user_progress(int(user["id"]), xp_gain=5, coins_gain=1)
    return build_response_with_cookie(
        {
            "message": "Logged in",
            "user": serialize_user(updated_user),
            "dashboard": build_dashboard_payload(updated_user),
        },
        token,
        expires_at,
    )


@fastapi_app.post("/auth/logout")
def logout(request: Request) -> JSONResponse:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    delete_session(token)
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return response


@fastapi_app.get("/auth/me")
def auth_me(current_user: sqlite3.Row = Depends(get_current_user)) -> dict[str, Any]:
    return serialize_user(current_user)


@fastapi_app.get("/dashboard")
def dashboard(current_user: sqlite3.Row = Depends(get_current_user)) -> dict[str, Any]:
    return build_dashboard_payload(current_user)


@fastapi_app.post("/predict")
async def predict(
    payload: PredictRequest,
    current_user: sqlite3.Row = Depends(get_current_user),
) -> dict[str, Any]:
    user_id = int(current_user["id"])
    user_model = get_user_model(user_id)

    if user_model.model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not ready. Train at least 2 classes to enable predictions.",
        )

    try:
        image = np.asarray(payload.pixel_data, dtype=np.uint8)
        feature_vector = extract_features(image)
        prediction, confidence = user_model.predict(feature_vector)
        definition = await get_object_definition(prediction)
        updated_user = update_user_progress(
            user_id,
            xp_gain=8,
            coins_gain=2,
            count_increments={"predictions_count": 1, "wiki_lookups": 1},
        )
        guess_id = add_guess_history(user_id, prediction, confidence)
        log_activity(
            user_id,
            "prediction",
            f"Guessed {prediction} with confidence {confidence:.2f}.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "guess_id": guess_id,
        "prediction": prediction,
        "confidence": confidence,
        "definition": definition,
        "user": serialize_user(updated_user),
        "achievements": build_achievements(updated_user),
    }


@fastapi_app.get("/training_samples/{class_name}")
def training_samples(
    class_name: str,
    current_user: sqlite3.Row = Depends(get_current_user),
) -> dict[str, Any]:
    user_id = int(current_user["id"])
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT pixel_data
            FROM training_samples
            WHERE user_id = ? AND class_name = ?
            ORDER BY id DESC
            LIMIT 5
            """,
            (user_id, class_name),
        ).fetchall()

    samples = [json.loads(row["pixel_data"]) for row in rows]
    return {"class_name": class_name, "samples": samples}


@fastapi_app.post("/predict/feedback")
def predict_feedback(
    payload: GuessFeedbackRequest,
    current_user: sqlite3.Row = Depends(get_current_user),
) -> dict[str, Any]:
    user_id = int(current_user["id"])

    with closing(get_db()) as conn:
        guess_row = conn.execute(
            "SELECT * FROM guess_history WHERE id = ? AND user_id = ?",
            (payload.guess_id, user_id),
        ).fetchone()
        if guess_row is None:
            raise HTTPException(status_code=404, detail="Guess record not found")

        if guess_row["is_correct"] is not None:
            updated_user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if updated_user is None:
                raise HTTPException(status_code=500, detail="Failed to fetch user")
            return {
                "message": "Feedback already recorded",
                "user": serialize_user(updated_user),
                "correct_guesses": get_correct_guesses(user_id),
            }

        conn.execute(
            """
            UPDATE guess_history
            SET is_correct = ?, reviewed_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                1 if payload.is_correct else 0,
                datetime.now(tz=UTC).isoformat(),
                payload.guess_id,
                user_id,
            ),
        )
        conn.commit()

    if payload.is_correct:
        updated_user = update_user_progress(user_id, xp_gain=12, coins_gain=4)
        log_activity(user_id, "guess_correct", "Marked a model guess as correct.")
        message = "Great! Correct guess recorded and rewards applied."
    else:
        updated_user = update_user_progress(user_id, xp_gain=2, coins_gain=0)
        log_activity(user_id, "guess_incorrect", "Marked a model guess as incorrect.")
        message = "Feedback recorded. Keep training for better accuracy."

    return {
        "message": message,
        "user": serialize_user(updated_user),
        "correct_guesses": get_correct_guesses(user_id),
    }


@fastapi_app.post("/train_class")
async def train_class(
    payload: TrainClassRequest,
    current_user: sqlite3.Row = Depends(get_current_user),
) -> dict[str, Any]:
    user_id = int(current_user["id"])
    class_name = payload.class_name.strip()
    if not class_name:
        raise HTTPException(status_code=400, detail="class_name must be non-empty")

    if len(payload.pixel_data) != 5:
        raise HTTPException(
            status_code=400,
            detail="pixel_data must contain exactly 5 pixel_data arrays",
        )

    try:
        with closing(get_db()) as conn:
            for pixel_data in payload.pixel_data:
                conn.execute(
                    """
                    INSERT INTO training_samples(user_id, class_name, pixel_data, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, class_name, json.dumps(pixel_data), datetime.now(tz=UTC).isoformat()),
                )
            conn.commit()

        existing_features, existing_labels = load_training_data(user_id)

        new_features: list[list[float]] = []
        for pixel_data in payload.pixel_data:
            image = np.asarray(pixel_data, dtype=np.uint8)
            feature_vector = extract_features(image)
            new_features.append(feature_vector.astype(np.float32).tolist())

        existing_features.extend(new_features)
        existing_labels.extend([class_name] * len(new_features))
        unique_class_count = len(set(existing_labels))

        save_training_data(user_id, existing_features, existing_labels)

        user_model = get_user_model(user_id)
        model_ready = False
        if unique_class_count >= 2:
            user_model.train(existing_features, existing_labels)
            user_model.save(get_user_model_path(user_id))
            model_ready = True
            train_message = "Class trained successfully"
        else:
            train_message = "Samples saved. Add at least one more class to train your personal SVM model."

        updated_user = update_user_progress(
            user_id,
            xp_gain=30,
            coins_gain=10,
            count_increments={"training_count": 1},
        )
        log_activity(
            user_id,
            "training",
            f"Taught the class {class_name} using 5 drawings.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    return {
        "message": train_message,
        "class_name": class_name,
        "added_samples": len(payload.pixel_data),
        "total_samples": len(existing_labels),
        "unique_classes": unique_class_count,
        "model_ready": model_ready,
        "user": serialize_user(updated_user),
        "dashboard": build_dashboard_payload(updated_user),
    }


@fastapi_app.delete("/train_class/{class_name}")
def delete_class(
    class_name: str,
    current_user: sqlite3.Row = Depends(get_current_user),
) -> dict[str, Any]:
    user_id = int(current_user["id"])
    class_name = class_name.strip()
    if not class_name:
        raise HTTPException(status_code=400, detail="class_name must be non-empty")

    try:
        with closing(get_db()) as conn:
            conn.execute(
                "DELETE FROM training_samples WHERE user_id = ? AND class_name = ?",
                (user_id, class_name),
            )
            conn.commit()

        features, labels = load_training_data(user_id)
        keep_indices = [i for i, label in enumerate(labels) if label != class_name]
        updated_features = [features[i] for i in keep_indices]
        updated_labels = [labels[i] for i in keep_indices]

        save_training_data(user_id, updated_features, updated_labels)

        user_model = get_user_model(user_id)
        unique_classes = sorted(list(set(updated_labels)))

        if len(unique_classes) >= 2:
            user_model.train(updated_features, updated_labels)
            user_model.save(get_user_model_path(user_id))
        else:
            user_model.model = None
            user_model.scaler = None
            model_path = get_user_model_path(user_id)
            if model_path.exists():
                model_path.unlink()

        log_activity(
            user_id,
            "remove_class",
            f"Removed the object class '{class_name}' from your library.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to remove class: {exc}") from exc

    return build_dashboard_payload(current_user)


@fastapi_app.get("/quickdraw/classes")
def quickdraw_classes(current_user: sqlite3.Row = Depends(get_current_user)) -> dict[str, Any]:
    del current_user
    classes = get_quickdraw_classes(QUICKDRAW_CACHE_DIR)
    return {"classes": classes}


@fastapi_app.get("/training/recommendation")
def training_recommendation(current_user: sqlite3.Row = Depends(get_current_user)) -> dict[str, Any]:
    user_id = int(current_user["id"])

    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT class_name, COUNT(*) AS sample_count
            FROM training_samples
            WHERE user_id = ?
            GROUP BY class_name
            ORDER BY class_name ASC
            """,
            (user_id,),
        ).fetchall()

    trained_counts = {str(row["class_name"]): int(row["sample_count"]) for row in rows}
    trained_classes = set(trained_counts.keys())
    trained_class_count = len(trained_classes)

    try:
        quickdraw_classes_list = get_quickdraw_classes(QUICKDRAW_CACHE_DIR)
    except Exception:
        quickdraw_classes_list = []

    if quickdraw_classes_list:
        untrained = [name for name in quickdraw_classes_list if name not in trained_classes]
        if untrained:
            suggested = untrained[0]
            return {
                "class_name": suggested,
                "reason": "Recommended because you have not trained this QuickDraw class yet.",
                "trained_class_count": trained_class_count,
            }

        # All known classes are trained, so suggest reinforcing the weakest one.
        if trained_counts:
            weakest_class = min(trained_counts.items(), key=lambda item: item[1])[0]
            return {
                "class_name": weakest_class,
                "reason": "All classes are unlocked. Reinforce this class with more samples for better accuracy.",
                "trained_class_count": trained_class_count,
            }

    if trained_counts:
        weakest_class = min(trained_counts.items(), key=lambda item: item[1])[0]
        return {
            "class_name": weakest_class,
            "reason": "QuickDraw catalog unavailable. Reinforce this class using more drawings.",
            "trained_class_count": trained_class_count,
        }

    return {
        "class_name": "cat",
        "reason": "Start by training a simple object to bootstrap your personal model.",
        "trained_class_count": 0,
    }


@fastapi_app.post("/quickdraw/train")
def quickdraw_train(
    payload: QuickDrawTrainRequest,
    current_user: sqlite3.Row = Depends(get_current_user),
) -> dict[str, Any]:
    user_id = int(current_user["id"])
    raw_classes = [name.strip() for name in payload.class_names if name and name.strip()]
    class_names = sorted(set(raw_classes))
    if not class_names:
        raise HTTPException(status_code=400, detail="At least one QuickDraw class is required")

    try:
        existing_features, existing_labels = load_training_data(user_id)
        imported_total = 0
        imported_per_class: dict[str, int] = {}
        raw_samples_to_insert: list[tuple[int, str, str, str]] = []

        for class_name in class_names:
            images = fetch_quickdraw_images_for_class(
                class_name,
                payload.samples_per_class,
                QUICKDRAW_CACHE_DIR,
            )
            if not images:
                continue

            count = 0
            for image in images:
                feature_vector = extract_features(image)
                existing_features.append(feature_vector.astype(np.float32).tolist())
                existing_labels.append(class_name)
                raw_samples_to_insert.append(
                    (
                        user_id,
                        class_name,
                        json.dumps(image.astype(np.uint8).tolist()),
                        datetime.now(tz=UTC).isoformat(),
                    )
                )
                count += 1

            if count > 0:
                imported_per_class[class_name] = count
                imported_total += count

        if imported_total == 0:
            raise HTTPException(status_code=400, detail="No QuickDraw samples could be imported")

        with closing(get_db()) as conn:
            conn.executemany(
                """
                INSERT INTO training_samples(user_id, class_name, pixel_data, created_at)
                VALUES (?, ?, ?, ?)
                """,
                raw_samples_to_insert,
            )
            conn.commit()

        save_training_data(user_id, existing_features, existing_labels)

        user_model = get_user_model(user_id)
        unique_classes = len(set(existing_labels))
        model_ready = False
        if unique_classes >= 2:
            user_model.train(existing_features, existing_labels)
            user_model.save(get_user_model_path(user_id))
            model_ready = True

        updated_user = update_user_progress(
            user_id,
            xp_gain=20 + imported_total // 6,
            coins_gain=8 + imported_total // 20,
            count_increments={"training_count": 1},
        )
        log_activity(
            user_id,
            "quickdraw_train",
            f"Imported {imported_total} QuickDraw samples across {len(imported_per_class)} classes.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"QuickDraw import failed: {exc}") from exc

    return {
        "message": "QuickDraw training completed",
        "imported_total": imported_total,
        "imported_per_class": imported_per_class,
        "total_samples": len(existing_labels),
        "unique_classes": unique_classes,
        "model_ready": model_ready,
        "user": serialize_user(updated_user),
        "dashboard": build_dashboard_payload(updated_user),
    }


@sio.event
async def connect(sid: str, environ: dict, auth: Any | None = None) -> bool | None:
    del auth
    token = extract_token_from_cookie_header(environ.get("HTTP_COOKIE"))
    user_row = get_user_by_session_token(token)
    if user_row is None:
        return False

    await sio.save_session(
        sid,
        {
            "user_id": int(user_row["id"]),
            "username": user_row["username"],
            "display_name": user_row["display_name"],
            "room_id": None,
        },
    )
    user_sid_index[sid] = {
        "user_id": int(user_row["id"]),
        "username": user_row["username"],
        "display_name": user_row["display_name"],
    }
    return None


@sio.event
async def disconnect(sid: str) -> None:
    session = await sio.get_session(sid)
    room_id = None
    if session:
        room_id = session.get("room_id")

    if room_id and room_id in multiplayer_rooms:
        room = multiplayer_rooms[room_id]
        user_id = int(session["user_id"]) if session else None

        if user_id is not None and user_id in room.get("players", {}):
            left_player = room["players"].pop(user_id)
            if user_id in room.get("player_order", []):
                room["player_order"].remove(user_id)

            leave_event = {
                "kind": "leave",
                "message": f"{left_player['display_name']} left the room",
                "correct": None,
                "user": {
                    "display_name": left_player["display_name"],
                    "username": left_player["username"],
                },
            }
            _append_room_history(room, leave_event)
            await sio.emit("room_event", leave_event, room=room_id)
            log_activity(user_id, "multiplayer_leave", f"Left multiplayer room {room_id}.")

            if room.get("drawer_user_id") == user_id:
                await _start_round(room_id)
            else:
                await _emit_room_states(room_id)

            if not room.get("player_order"):
                _cancel_room_timer(room_id)
                multiplayer_rooms.pop(room_id, None)

    user_sid_index.pop(sid, None)


@sio.event
async def join_room(sid: str, data: dict[str, Any]) -> None:
    session = await sio.get_session(sid)
    if not session:
        return

    room_id = str(data.get("room_id", "")).strip()
    if not room_id:
        return

    previous_room = session.get("room_id")
    if previous_room and previous_room != room_id:
        await sio.leave_room(sid, previous_room)

    await sio.enter_room(sid, room_id)
    room = _ensure_room(room_id)
    user_id = int(session["user_id"])
    if user_id not in room["players"]:
        room["players"][user_id] = {
            "sid": sid,
            "user_id": user_id,
            "username": session["username"],
            "display_name": session["display_name"],
            "score": 0,
            "has_guessed": False,
        }
        room["player_order"].append(user_id)
    else:
        room["players"][user_id]["sid"] = sid

    session["room_id"] = room_id
    await sio.save_session(sid, session)

    updated_user = update_user_progress(
        user_id,
        xp_gain=4,
        coins_gain=1,
        count_increments={"multiplayer_count": 1},
    )

    join_event = {
        "kind": "join",
        "message": f"{updated_user['display_name']} joined {room_id}",
        "correct": None,
        "user": {"display_name": updated_user["display_name"], "username": updated_user["username"]},
    }
    _append_room_history(room, join_event)
    await sio.emit("room_event", join_event, room=room_id)
    await _emit_room_states(room_id)
    if room.get("phase") != "drawing":
        await _start_round(room_id)

    log_activity(user_id, "multiplayer", f"Joined multiplayer room {room_id}.")


@sio.event
async def draw_stroke(sid: str, data: dict[str, Any]) -> None:
    session = await sio.get_session(sid)
    if not session:
        return

    room_id = str(data.get("room_id", "")).strip()
    if not room_id:
        return

    room = multiplayer_rooms.get(room_id)
    if room is None:
        return

    if room.get("phase") != "drawing" or int(session["user_id"]) != room.get("drawer_user_id"):
        return

    x = data.get("x")
    y = data.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return

    await sio.emit(
        "draw_stroke",
        {"x": x, "y": y, "room_id": room_id},
        room=room_id,
        skip_sid=sid,
    )


@sio.event
async def clear_canvas(sid: str, data: dict[str, Any]) -> None:
    session = await sio.get_session(sid)
    if not session:
        return

    room_id = str(data.get("room_id", "")).strip()
    if not room_id:
        return

    room = multiplayer_rooms.get(room_id)
    if room is None:
        return

    if room.get("phase") != "drawing" or int(session["user_id"]) != room.get("drawer_user_id"):
        return

    await sio.emit("clear_canvas", room=room_id)


@sio.event
async def submit_guess(sid: str, data: dict[str, Any]) -> None:
    session = await sio.get_session(sid)
    if not session:
        return

    room_id = str(data.get("room_id", "")).strip()
    guess = str(data.get("guess", "")).strip()
    if not room_id or not guess:
        return

    room = multiplayer_rooms.get(room_id)
    if room is None or room.get("phase") != "drawing":
        return

    user_id = int(session["user_id"])
    if user_id == room.get("drawer_user_id"):
        return

    player = room.get("players", {}).get(user_id)
    if player is not None:
        player["has_guessed"] = True

    is_correct = guess.lower() == str(room.get("prompt", "")).lower()
    display_name = player["display_name"] if player else session["display_name"]

    if is_correct:
        if player is not None:
            player["score"] = int(player.get("score", 0)) + 10
        update_user_progress(user_id, xp_gain=14, coins_gain=4, count_increments={"wins": 1})
        event = {
            "kind": "guess",
            "message": f"{display_name} guessed the word",
            "guess": guess,
            "correct": True,
            "user": {"display_name": display_name, "username": session["username"]},
        }
        _append_room_history(room, event)
        await sio.emit("room_event", event, room=room_id)
        await _emit_room_states(room_id)
        _cancel_room_timer(room_id)
        await _finish_round(room_id, "correct guess", winner_user_id=user_id)
        log_activity(user_id, "multiplayer_guess", f"Correctly guessed in room {room_id}.")
    else:
        event = {
            "kind": "guess",
            "message": f"{display_name} guessed",
            "guess": guess,
            "correct": False,
            "user": {"display_name": display_name, "username": session["username"]},
        }
        _append_room_history(room, event)
        await sio.emit("room_event", event, room=room_id)


@sio.event
async def trigger_ai_guess(sid: str, data: dict[str, Any]) -> None:
    session = await sio.get_session(sid)
    if not session:
        return

    room_id = str(data.get("room_id", "")).strip()
    pixel_data = data.get("pixel_data")

    if not room_id or pixel_data is None:
        return

    room = multiplayer_rooms.get(room_id)
    if room is None:
        return
    if room.get("phase") != "drawing" or int(session["user_id"]) != room.get("drawer_user_id"):
        return

    try:
        image = np.asarray(pixel_data, dtype=np.uint8)
        feature_vector = extract_features(image)
        user_id = int(session["user_id"])
        user_model = get_user_model(user_id)
        if user_model.model is None:
            raise ValueError("Model not ready")

        prediction, confidence = user_model.predict(feature_vector)
        definition = await get_object_definition(prediction)
        message = f"AI guess: {prediction} ({confidence:.2f})"
        updated_user = update_user_progress(
            user_id,
            xp_gain=6,
            coins_gain=1,
        )
        log_activity(
            int(session["user_id"]),
            "battle_guess",
            f"Broadcast {prediction} to room {room_id}.",
        )
    except Exception:
        prediction = "unknown"
        confidence = 0.0
        definition = "AI guess unavailable. Train the model with more classes."
        message = "AI guess unavailable. Train the model with more classes."
        updated_user = None

    await sio.emit(
        "ai_chat_message",
        {
            "room_id": room_id,
            "message": message,
            "prediction": prediction,
            "confidence": confidence,
            "definition": definition,
            "user": serialize_user(updated_user) if updated_user is not None else None,
        },
        room=room_id,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
