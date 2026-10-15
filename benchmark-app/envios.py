"""Envios dos usuários do app Juiz de Bocha (Firebase Auth + Firestore, projeto juizdebocha).

Lê as coleções `Photos` (um documento por foto enviada; campo `user` aponta para `users/{uid}`) e `users`,
e guarda uma cópia no SQLite local para consultar, filtrar por UID e exportar.
Precisa de uma chave de conta de serviço do Firebase (Admin SDK) — o arquivo fica só na sua máquina."""
import os
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS usuarios (
    uid TEXT PRIMARY KEY, email TEXT, nome TEXT, criado_em TEXT, telefone TEXT, foto_url TEXT,
    dados_json TEXT, sincronizado_em TEXT
);
CREATE TABLE IF NOT EXISTS envios (
    doc_id TEXT PRIMARY KEY, uid TEXT, foto_id TEXT, data TEXT, latitude REAL, longitude REAL,
    url TEXT, thumb TEXT, local TEXT, ref TEXT, numero_foto_usuario REAL, dados_json TEXT, sincronizado_em TEXT
);
CREATE INDEX IF NOT EXISTS envios_uid ON envios (uid);
"""
TABELAS = ["usuarios", "envios"]


def caminho_credencial(raiz: Path) -> Path:
    return Path(os.environ.get("BENCH_FIREBASE_CRED", raiz / "credenciais" / "firebase-admin.json"))


def preparar(con: sqlite3.Connection):
    con.executescript(SCHEMA)


def _texto(v):
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def _json(d: dict) -> str:
    import json

    def conv(v):
        if hasattr(v, "latitude") and hasattr(v, "longitude"):
            return {"lat": v.latitude, "lng": v.longitude}
        if hasattr(v, "path") and hasattr(v, "id"):  # DocumentReference
            return v.path
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return str(v)
    return json.dumps(d, ensure_ascii=False, default=conv)


def firebase_app(credencial: Path):
    """App do Firebase Admin (um só por processo), com o bucket do projeto juizdebocha."""
    import firebase_admin
    from firebase_admin import credentials

    nome_app = "juiz-bocha-benchmark"
    try:
        return firebase_admin.get_app(nome_app)
    except ValueError:
        return firebase_admin.initialize_app(credentials.Certificate(str(credencial)),
                                             {"storageBucket": "juizdebocha.appspot.com"}, name=nome_app)


def sincronizar(con: sqlite3.Connection, credencial: Path) -> dict:
    """Baixa users e Photos do Firestore e substitui a cópia local. Retorna contagens."""
    from firebase_admin import firestore

    db = firestore.client(app=firebase_app(credencial))
    agora = datetime.now().isoformat(timespec="seconds")

    usuarios = []
    for doc in db.collection("users").stream():
        d = doc.to_dict() or {}
        usuarios.append((doc.id, d.get("email"), d.get("display_name") or d.get("name"), _texto(d.get("created_time")),
                         d.get("phone_number"), d.get("photo_url"), _json(d), agora))

    envios = []
    for doc in db.collection("Photos").stream():
        d = doc.to_dict() or {}
        user = d.get("user")
        uid = getattr(user, "id", None) or (str(user).rstrip("/").split("/")[-1] if user else None)
        geo = d.get("takedOn")
        envios.append((doc.id, uid, d.get("id"), _texto(d.get("date")),
                       getattr(geo, "latitude", None), getattr(geo, "longitude", None),
                       d.get("url"), d.get("thumb"), d.get("place"), d.get("ref"), d.get("userPhotoNumber"),
                       _json(d), agora))

    with con:
        con.execute("DELETE FROM usuarios")
        con.execute("DELETE FROM envios")
        con.executemany("INSERT INTO usuarios VALUES (?,?,?,?,?,?,?,?)", usuarios)
        con.executemany("INSERT INTO envios VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", envios)
    return dict(usuarios=len(usuarios), envios=len(envios), sem_usuario=sum(1 for e in envios if not e[1]))
