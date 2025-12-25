from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import pickle
import os
from insightface.app import FaceAnalysis
from numpy.linalg import norm

DATA_DIR = "data"
EMBED_FILE = f"{DATA_DIR}/embeddings.pkl"
THRESHOLD = 0.5

os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Face Recognition Server")

face_app = FaceAnalysis(name="buffalo_s",  root="./models")
face_app.prepare(ctx_id=0, det_size=(320, 320))

if os.path.exists(EMBED_FILE):
    with open(EMBED_FILE, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}  # {name: [embeddings]}


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def get_embedding(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    if len(faces) == 0:
        return None

    return faces[0].embedding


@app.get("/")
def root():
    return {"status": "Face recognition server running"}


@app.post("/register")
async def register_face(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    embedding = get_embedding(image_bytes)

    if embedding is None:
        return {"success": False, "message": "No face detected"}

    if name not in face_db:
        face_db[name] = []

    face_db[name].append(embedding)

    with open(EMBED_FILE, "wb") as f:
        pickle.dump(face_db, f)

    return {
        "success": True,
        "name": name,
        "total_samples": len(face_db[name])
    }


@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    embedding = get_embedding(image_bytes)

    if embedding is None:
        return {"name": "Unknown", "score": 0}

    best_match = "Unknown"
    best_score = 0

    for name, embeddings in face_db.items():
        for emb in embeddings:
            score = cosine_similarity(embedding, emb)
            if score > best_score:
                best_score = score
                best_match = name

    if best_score < THRESHOLD:
        return {"name": "Unknown", "score": round(float(best_score), 3)}

    return {
        "name": best_match,
        "score": round(float(best_score), 3)
    }
