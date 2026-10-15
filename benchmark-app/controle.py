"""Controle dos treinos pelo app: fila na coleção `benchmark_treinos` do Firestore (projeto juizdebocha).

O notebook JuizDeBocha_Experimentos (FONTE_FILA = "app") roda no Colab, reserva os treinos da GPU em que foi aberto,
atualiza `progresso`/`batimento`, obedece a `cancelar` e, ao terminar, grava o relatório no Storage em
benchmark/relatorios/<id>/relatorio.json e marca `status = concluido`."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from envios import firebase_app

COLECAO = "benchmark_treinos"
GPUS = ["qualquer", "A100", "G4", "L4", "T4", "H100"]
MODELOS = {
    "R50": "Mask R-CNN R50-FPN (atual)",
    "R101": "Mask R-CNN R101-FPN",
    "X101": "Mask R-CNN X101-32x8d-FPN",
    "CASCADE_R50": "Cascade Mask R-CNN R50-FPN",
}
STATUS = {"na_fila": "⏳ na fila", "rodando": "▶ rodando", "concluido": "✔ concluído",
          "erro": "✘ erro", "cancelado": "■ cancelado"}

# a mesma fila padrão do notebook (comparação justa: base congelada da A100)
FILA_PADRAO = [
    dict(nome="base_r50_it1000", modelo="R50", it2=1000, lr2=0.0025, aumentos=True),
    dict(nome="r50_it3000", modelo="R50", it2=3000, lr2=0.0025, aumentos=True),
    dict(nome="r50_it5000", modelo="R50", it2=5000, lr2=0.0025, aumentos=True),
    dict(nome="r50_lr0.00125_it1000", modelo="R50", it2=1000, lr2=0.00125, aumentos=True),
    dict(nome="r50_lr0.005_it1000", modelo="R50", it2=1000, lr2=0.005, aumentos=True),
    dict(nome="r50_sem_aumentos_it1000", modelo="R50", it2=1000, lr2=0.0025, aumentos=False),
    dict(nome="r101_it1000", modelo="R101", it2=1000, lr2=0.0025, aumentos=True),
    dict(nome="x101_it1000", modelo="X101", it2=1000, lr2=0.0025, aumentos=True),
    dict(nome="cascade_r50_it1000", modelo="CASCADE_R50", it2=1000, lr2=0.0025, aumentos=True),
]


def _fs(cred: Path):
    from firebase_admin import firestore
    return firestore.client(app=firebase_app(cred))


def _id(nome: str, gpu: str) -> str:
    limpo = re.sub(r"[^A-Za-z0-9_.-]+", "_", nome).strip("_") or "treino"
    return f"{limpo}__{gpu}__{datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]}"


def enfileirar(cred: Path, nome: str, gpu: str, modelo: str, it2: int, lr2: float, aumentos: bool,
               it1: int = 15000, etapa1: str = "auto", origem: str | None = None) -> str:
    from firebase_admin import firestore
    doc_id = _id(nome, gpu)
    _fs(cred).collection(COLECAO).document(doc_id).set(dict(
        nome=nome, gpu_desejada=gpu, status="na_fila", cancelar=False, criado_em=firestore.SERVER_TIMESTAMP,
        parametros=dict(modelo=modelo, etapa1=etapa1, it1=int(it1), it2=int(it2), lr2=float(lr2), aumentos=bool(aumentos)),
        origem=origem,
    ))
    return doc_id


def listar(cred: Path) -> list[dict]:
    docs = []
    for d in _fs(cred).collection(COLECAO).stream():
        j = d.to_dict() or {}
        j["id"] = d.id
        docs.append(j)
    return sorted(docs, key=lambda j: j.get("criado_em") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)


def cancelar(cred: Path, doc_id: str):
    ref = _fs(cred).collection(COLECAO).document(doc_id)
    if (ref.get().to_dict() or {}).get("status") == "na_fila":
        ref.update({"status": "cancelado", "cancelar": True})
    else:  # rodando: o notebook vê a flag no próximo aviso de progresso e para
        ref.update({"cancelar": True})


def rodar_novamente(cred: Path, job: dict, gpu: str | None = None) -> str:
    p = job.get("parametros") or {}
    return enfileirar(cred, job.get("nome", "treino"), gpu or job.get("gpu_desejada", "qualquer"), p.get("modelo", "R50"),
                      p.get("it2", 1000), p.get("lr2", 0.0025), p.get("aumentos", True), p.get("it1", 15000),
                      p.get("etapa1", "auto"), origem=job["id"])


def apagar(cred: Path, doc_id: str):
    _fs(cred).collection(COLECAO).document(doc_id).delete()


def baixar_relatorio(cred: Path, caminho_storage: str) -> dict:
    from firebase_admin import storage
    blob = storage.bucket(app=firebase_app(cred)).blob(caminho_storage)
    return json.loads(blob.download_as_bytes())
