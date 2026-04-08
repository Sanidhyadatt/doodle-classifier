from fastapi import FastAPI, HTTPException, WebSocket, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import json
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from pydantic import BaseModel
import aiohttp
import asyncio
import uuid

from backend.ml.model_manager import NeuroModel
from backend.ml.extractor import extract_features
import cv2

# ============================================================================
# Configuration
# ============================================================================
APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
MODEL_DIR = APP_ROOT / "models"
DB_PATH = APP_ROOT / "neurosketch_arena.db"
TRAINING_DATA_PATH = DATA_DIR / "training_data.pkl"
MODEL_PATH = MODEL_DIR / "neuro_model.pkl"
QUICKDRAW_DIR = DATA_DIR / "quickdraw"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Global state
# ============================================================================
user_models = {}  # {user_id: NeuroModel instance}
user_training_data = {}  # {user_id: {"features": [...], "labels": [...]}}
broadcaster_connections = {}  # WebSocket connections for multiplayer

# Load default model
default_model = NeuroModel()
if MODEL_PATH.exists():
    try:
        default_model.load(str(MODEL_PATH))
    except Exception as e:
        print(f"Warning: Could not load default model: {e}")


# ============================================================================
# Database initialization
# ============================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            correct BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            class_name TEXT,
            samples_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    
    conn.commit()
    conn.close()


# ============================================================================
# Pydantic models
# ============================================================================
class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str


class PredictRequest(BaseModel):
    pixels: list[int]  # Grayscale pixel array


class PredictFeedbackRequest(BaseModel):
    prediction_id: int
    correct: bool


class QuickDrawTrainRequest(BaseModel):
    classes: list[str]
    samples_per_class: int = 40


class DashboardData(BaseModel):
    user: dict
    leaderboard: list
    missions: list
    achievements: list
    recent_activity: list
    correct_guesses: list
    model_status: dict


# ============================================================================
# Authentication (basic, no hashing for demo)
# ============================================================================
def hash_password(password: str) -> str:
    """Simple hash for demo (use bcrypt in production!)"""
    return str(hash(password))


def get_user_by_username(username: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, display_name FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "username": row[1], "display_name": row[2]}
    return None


def create_user(username: str, password: str, display_name: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, display_name) VALUES (?, ?, ?)",
            (username, hash_password(password), display_name),
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {"id": user_id, "username": username, "display_name": display_name}
    except sqlite3.IntegrityError:
        conn.close()
        return None


def verify_user(username: str, password: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, display_name, password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row and row[3] == hash_password(password):
        return {"id": row[0], "username": row[1], "display_name": row[2]}
    return None


# ============================================================================
# QuickDraw support
# ============================================================================
def load_quickdraw_classes() -> list[str]:
    """Load available QuickDraw classes from metadata."""
    metadata_path = QUICKDRAW_DIR / "quickdraw_classes.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def load_quickdraw_samples(class_name: str, count: int = 40) -> list[np.ndarray]:
    """Load sample drawings from QuickDraw dataset, or generate synthetic data if not available."""
    ndjson_path = QUICKDRAW_DIR / f"quickdraw_{class_name}.ndjson"
    samples = []
    
    # Try to load from NDJSON if available
    if ndjson_path.exists():
        try:
            with open(ndjson_path) as f:
                for i, line in enumerate(f):
                    if i >= count:
                        break
                    data = json.loads(line)
                    if "drawing" in data:
                        # Convert drawing strokes to image
                        image = _strokes_to_image(data["drawing"], 64)
                        samples.append(image)
        except Exception as e:
            print(f"Error loading QuickDraw samples from {ndjson_path}: {e}")
    
    # If NDJSON not found or loading failed, generate synthetic samples
    if not samples:
        print(f"Generating synthetic training data for class: {class_name}")
        samples = _generate_synthetic_samples(class_name, count)
    
    return samples


def _generate_synthetic_samples(class_name: str, count: int) -> list[np.ndarray]:
    """Generate synthetic doodle samples for training when real data is unavailable."""
    samples = []
    np.random.seed(hash(class_name) % (2 ** 32))  # Deterministic seed per class
    
    for i in range(count):
        image = np.full((64, 64), 255, dtype=np.uint8)
        
        # Generate random strokes
        num_strokes = np.random.randint(2, 6)
        for _ in range(num_strokes):
            # Random starting point
            start_x = np.random.randint(5, 59)
            start_y = np.random.randint(5, 59)
            
            # Random direction and length
            num_points = np.random.randint(5, 20)
            angle = np.random.uniform(0, 2 * np.pi)
            length_step = np.random.uniform(1, 3)
            
            points = []
            x, y = start_x, start_y
            for j in range(num_points):
                points.append((int(x), int(y)))
                x += length_step * np.cos(angle + np.random.normal(0, 0.3))
                y += length_step * np.sin(angle + np.random.normal(0, 0.3))
            
            # Draw stroke on image
            for j in range(len(points) - 1):
                pt1 = (max(0, min(63, points[j][0])), max(0, min(63, points[j][1])))
                pt2 = (max(0, min(63, points[j + 1][0])), max(0, min(63, points[j + 1][1])))
                cv2.line(image, pt1, pt2, 0, thickness=2)
        
        samples.append(image)
    
    return samples


def _strokes_to_image(strokes: list, size: int = 64) -> np.ndarray:
    """Convert stroke data to grayscale image."""
    image = np.full((size, size), 255, dtype=np.uint8)
    for stroke in strokes:
        if len(stroke[0]) < 2:
            continue
        xs = np.array(stroke[0], dtype=np.float32) * (size / 256.0)
        ys = np.array(stroke[1], dtype=np.float32) * (size / 256.0)
        for i in range(len(xs) - 1):
            pt1 = (int(xs[i]), int(ys[i]))
            pt2 = (int(xs[i + 1]), int(ys[i + 1]))
            cv2.line(image, pt1, pt2, 0, thickness=2)
    return image


# ============================================================================
# Session management
# ============================================================================
sessions = {}  # {session_id: user}
SESSION_COOKIE_NAME = "session_token"
SESSION_COOKIE_MAX_AGE = 86400 * 7  # 7 days


def create_session_token(user: dict) -> str:
    """Create a session token (simple UUID)"""
    token = str(uuid.uuid4())
    sessions[token] = user
    return token


def get_user_from_request(request: Request) -> dict | None:
    """Extract user from cookies"""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token and token in sessions:
        return sessions[token]
    return None


# ============================================================================
# FastAPI setup
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown
    pass


app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Routes: Authentication
# ============================================================================
@app.post("/auth/register")
async def register(req: RegisterRequest, response: Response):
    user = create_user(req.username, req.password, req.display_name)
    if not user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    token = create_session_token(user)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    
    # Build complete user profile
    user_profile = {
        **user,  # id, username, display_name
        "xp": 0,
        "coins": 0,
        "level": 1,
        "streak": 0,
        "predictions_count": 0,
        "training_count": 0,
        "multiplayer_count": 0,
        "wiki_lookups": 0,
        "wins": 0,
        "losses": 0,
        "next_level_xp": 100,
        "level_progress": 0,
        "achievements": [],
    }
    
    return {
        "message": f"Welcome to the Doodle Arena, {user['display_name']}!",
        "user": user,
        "dashboard": DashboardData(
            user=user_profile,
            leaderboard=[],
            missions=[],
            achievements=[],
            recent_activity=[],
            correct_guesses=[],
            model_status={"ready": False, "classes": []},
        ).model_dump(),
    }


@app.post("/auth/login")
async def login(req: LoginRequest, response: Response):
    user = verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_session_token(user)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    
    # Get trained classes for model status
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user["id"],))
    predictions_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM training_logs WHERE user_id = ?", (user["id"],))
    training_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT DISTINCT class_name FROM training_logs WHERE user_id = ?", (user["id"],))
    trained_classes = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    # Build complete user profile
    user_profile = {
        **user,  # id, username, display_name
        "xp": 0,
        "coins": 0,
        "level": 1,
        "streak": 0,
        "predictions_count": predictions_count,
        "training_count": training_count,
        "multiplayer_count": 0,
        "wiki_lookups": 0,
        "wins": 0,
        "losses": 0,
        "next_level_xp": 100,
        "level_progress": 0,
        "achievements": [],
    }
    
    return {
        "message": f"Welcome back, {user['display_name']}!",
        "user": user,
        "dashboard": DashboardData(
            user=user_profile,
            leaderboard=[],
            missions=[],
            achievements=[],
            recent_activity=[],
            correct_guesses=[],
            model_status={
                "ready": training_count > 0,
                "classes": trained_classes,
            },
        ).model_dump(),
    }


@app.post("/auth/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token and token in sessions:
        del sessions[token]
    
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return {"ok": True}


# ============================================================================
# Routes: Dashboard & User
# ============================================================================
@app.get("/dashboard")
async def get_dashboard(request: Request):
    user = get_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user["id"],))
    predictions_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM training_logs WHERE user_id = ?", (user["id"],))
    training_count = cursor.fetchone()[0]
    
    conn.close()
    
    # Get trained classes for model status
    trained_classes = []
    if training_count > 0:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT class_name FROM training_logs WHERE user_id = ?", (user["id"],))
        trained_classes = [row[0] for row in cursor.fetchall()]
        conn.close()
    
    # Build complete user profile
    user_profile = {
        **user,  # id, username, display_name
        "xp": 0,
        "coins": 0,
        "level": 1,
        "streak": 0,
        "predictions_count": predictions_count,
        "training_count": training_count,
        "multiplayer_count": 0,
        "wiki_lookups": 0,
        "wins": 0,
        "losses": 0,
        "next_level_xp": 100,
        "level_progress": 0,
        "achievements": [],
    }
    
    return DashboardData(
        user=user_profile,
        leaderboard=[],
        missions=[],
        achievements=[],
        recent_activity=[],
        correct_guesses=[],
        model_status={
            "ready": training_count > 0,
            "classes": trained_classes,
        },
    ).model_dump()


# ============================================================================
# Routes: Prediction
# ============================================================================
@app.post("/predict")
async def predict(req: PredictRequest, request: Request):
    user = get_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Convert pixel array to numpy array
    try:
        pixels = np.array(req.pixels, dtype=np.uint8)
        image = pixels.reshape(64, 64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid pixel data")
    
    # Extract features
    try:
        features = extract_features(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")
    
    # Get user model or default
    model = user_models.get(user["id"], default_model)
    
    # Predict
    try:
        label, confidence = model.predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    # Log to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (user_id, prediction, confidence) VALUES (?, ?, ?)",
        (user["id"], label, confidence),
    )
    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    
    return {
        "prediction_id": pred_id,
        "prediction": label,
        "confidence": float(confidence),
    }


@app.post("/predict/feedback")
async def predict_feedback(req: PredictFeedbackRequest, request: Request):
    user = get_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE predictions SET correct = ? WHERE id = ? AND user_id = ?",
        (req.correct, req.prediction_id, user["id"]),
    )
    conn.commit()
    conn.close()
    
    return {"ok": True}


# ============================================================================
# Routes: QuickDraw Training
# ============================================================================
@app.get("/quickdraw/classes")
async def get_quickdraw_classes():
    return {"classes": load_quickdraw_classes()}


@app.get("/training_samples/{class_name}")
async def get_training_samples(class_name: str):
    """Return raw pixel samples"""
    samples = load_quickdraw_samples(class_name, count=5)
    if not samples:
        raise HTTPException(status_code=404, detail="Class not found")
    
    # Convert to pixel arrays
    pixel_data = []
    for sample in samples:
        pixels = sample.flatten().tolist()
        pixel_data.append(pixels)
    
    return {"samples": pixel_data}


@app.get("/training_data/{class_name}")
async def get_training_data(class_name: str):
    """Return training data (1 sample)"""
    samples = load_quickdraw_samples(class_name, count=1)
    if not samples:
        raise HTTPException(status_code=404, detail="Class not found")
    
    pixels = samples[0].flatten().tolist()
    return {"sample": pixels}


@app.post("/quickdraw/train")
async def train_quickdraw(req: QuickDrawTrainRequest, request: Request):
    user = get_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    feature_vectors = []
    labels = []
    
    # Load data for each class
    for class_name in req.classes:
        samples = load_quickdraw_samples(class_name, count=req.samples_per_class)
        for sample in samples:
            try:
                features = extract_features(sample)
                feature_vectors.append(features.tolist())
                labels.append(class_name)
            except Exception as e:
                print(f"Error extracting features from {class_name}: {e}")
                continue
    
    if not feature_vectors:
        raise HTTPException(status_code=400, detail="No training data generated")
    
    # Train user model
    try:
        model = NeuroModel()
        model.train(feature_vectors, labels)
        user_models[user["id"]] = model
        user_training_data[user["id"]] = {
            "features": feature_vectors,
            "labels": labels,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
    
    # Log training
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for class_name in req.classes:
        cursor.execute(
            "INSERT INTO training_logs (user_id, class_name, samples_count) VALUES (?, ?, ?)",
            (user["id"], class_name, req.samples_per_class),
        )
    conn.commit()
    conn.close()
    
    return {
        "ok": True,
        "samples_trained": len(feature_vectors),
        "classes": req.classes,
    }


# ============================================================================
# Routes: Training recommendations
# ============================================================================
@app.get("/training/recommendation")
async def get_training_recommendation(request: Request):
    user = get_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Get trained classes
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT class_name FROM training_logs WHERE user_id = ?",
        (user["id"],),
    )
    trained_classes = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    available_classes = load_quickdraw_classes()
    untrained = [c for c in available_classes if c not in trained_classes]
    
    recommendation = untrained[0] if untrained else "all_classes_trained"
    
    return {
        "recommendation": recommendation,
        "trained_class_count": len(trained_classes),
    }


# ============================================================================
# WebSocket: Multiplayer Arena
# ============================================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    room_id = None
    user = None
    
    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("type")
            
            if event == "join_room":
                room_id = data.get("room_id", "arena")
                # Initialize room if needed
                if room_id not in broadcaster_connections:
                    broadcaster_connections[room_id] = set()
                broadcaster_connections[room_id].add(websocket)
                
                # Send room state
                await websocket.send_json({
                    "type": "room_state",
                    "room_id": room_id,
                    "round_ends_at": datetime.now().isoformat(),
                })
            
            elif event == "draw_stroke" and room_id:
                # Broadcast stroke to other players
                for conn in broadcaster_connections.get(room_id, set()):
                    if conn != websocket:
                        await conn.send_json({
                            "type": "draw_stroke",
                            "x": data.get("x"),
                            "y": data.get("y"),
                        })
            
            elif event == "clear_canvas" and room_id:
                # Broadcast clear signal
                for conn in broadcaster_connections.get(room_id, set()):
                    if conn != websocket:
                        await conn.send_json({"type": "clear_canvas"})
            
            elif event == "ai_prediction" and room_id:
                # Broadcast AI prediction to room
                for conn in broadcaster_connections.get(room_id, set()):
                    await conn.send_json({
                        "type": "ai_chat_message",
                        "prediction": data.get("prediction"),
                        "confidence": data.get("confidence"),
                    })
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    
    finally:
        if room_id and room_id in broadcaster_connections:
            broadcaster_connections[room_id].discard(websocket)


# ============================================================================
# Health check
# ============================================================================
@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
