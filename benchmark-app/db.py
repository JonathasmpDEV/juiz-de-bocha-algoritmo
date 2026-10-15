"""Banco SQLite local do benchmark: importa os relatorio.json gerados pelo notebook de experimentos."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS experimentos (
    id TEXT PRIMARY KEY, nome TEXT, grupos TEXT, gpu TEXT, gpu_nome TEXT, gpu_avaliacao TEXT,
    modelo TEXT, etapa1_modo TEXT, it1 INTEGER, it2 INTEGER, lr2 REAL, aumentos INTEGER,
    adaptado INTEGER, lote1 INTEGER, lote2 INTEGER, amp INTEGER, tf32 INTEGER,
    lr2_efetivo REAL, it2_efetivas INTEGER,
    inicio TEXT, fim TEXT, tempo_total_s REAL,
    etapa1_tempo_s REAL, etapa1_img_s REAL, etapa1_s_iter REAL, etapa1_data_time REAL,
    etapa2_tempo_s REAL, etapa2_img_s REAL, etapa2_s_iter REAL, etapa2_s_iter_p90 REAL, etapa2_data_time REAL,
    memoria_pico_mb REAL, parametros INTEGER, modelo_mb REAL,
    ap_bbox REAL, ap50_bbox REAL, ap75_bbox REAL, ap_segm REAL, ap50_segm REAL, ap75_segm REAL,
    decisao_082 REAL, acuracia_vencedora_082 REAL, acuracia_balim_082 REAL, inferencia_ms_082 REAL,
    juiz_tempo_s REAL, val_imagens INTEGER, train_imagens INTEGER, train_sha1 TEXT, val_sha1 TEXT,
    cuda TEXT, cudnn TEXT, torch TEXT, detectron2 TEXT, python TEXT, cpus INTEGER, ram_gb REAL,
    repo_commit TEXT, observacao TEXT, origem TEXT, importado_em TEXT, relatorio_json TEXT
);
CREATE TABLE IF NOT EXISTS coco (
    exp_id TEXT, etapa TEXT, tarefa TEXT, metrica TEXT, valor REAL,
    PRIMARY KEY (exp_id, etapa, tarefa, metrica)
);
CREATE TABLE IF NOT EXISTS juiz_limiar (
    exp_id TEXT, limiar REAL, fotos INTEGER, taxa_decisao REAL, acuracia_vencedora REAL, acuracia_balim REAL,
    fotos_com_referencia INTEGER, erro_contagem_medio REAL, bolas_media REAL,
    inferencia_ms_media REAL, inferencia_ms_p50 REAL, inferencia_ms_p90 REAL,
    PRIMARY KEY (exp_id, limiar)
);
CREATE TABLE IF NOT EXISTS por_foto (
    exp_id TEXT, foto TEXT, limiar REAL, largura INTEGER, altura INTEGER, inferencia_ms REAL,
    n_detectadas INTEGER, decidiu INTEGER, n_bolas INTEGER, dist_vencedora REAL, dist_segunda REAL,
    margem_px REAL, balim_diametro_px REAL, score_min REAL, score_medio REAL,
    ref_n_bolas INTEGER, ref_decidiu INTEGER, acerto_balim INTEGER, acerto_vencedora INTEGER,
    PRIMARY KEY (exp_id, foto, limiar)
);
CREATE TABLE IF NOT EXISTS curvas (
    exp_id TEXT, etapa TEXT, iteracao INTEGER, metrica TEXT, valor REAL,
    PRIMARY KEY (exp_id, etapa, iteracao, metrica)
);
CREATE TABLE IF NOT EXISTS ambiente (
    exp_id TEXT, chave TEXT, valor TEXT, PRIMARY KEY (exp_id, chave)
);
CREATE TABLE IF NOT EXISTS gpu_custos (
    gpu TEXT PRIMARY KEY, unidades_por_hora REAL, observacao TEXT
);
"""

TABELAS = ["experimentos", "coco", "juiz_limiar", "por_foto", "curvas", "ambiente", "gpu_custos"]
COLUNAS_JUIZ = ["fotos", "taxa_decisao", "acuracia_vencedora", "acuracia_balim", "fotos_com_referencia",
                "erro_contagem_medio", "bolas_media", "inferencia_ms_media", "inferencia_ms_p50", "inferencia_ms_p90"]
COLUNAS_FOTO = ["largura", "altura", "inferencia_ms", "n_detectadas", "decidiu", "n_bolas", "dist_vencedora",
                "dist_segunda", "margem_px", "balim_diametro_px", "score_min", "score_medio", "ref_n_bolas",
                "ref_decidiu", "acerto_balim", "acerto_vencedora"]


def conectar(caminho: Path) -> sqlite3.Connection:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(caminho, check_same_thread=False)
    con.executescript(SCHEMA)
    return con


def _g(d, *chaves, padrao=None):
    for c in chaves:
        if not isinstance(d, dict) or d.get(c) is None:
            return padrao
        d = d[c]
    return d


def _num(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def importar_relatorio(con: sqlite3.Connection, rel: dict, origem: str) -> str:
    exp_id = rel["id"]
    p, amb, perfil = rel.get("parametros") or {}, rel.get("ambiente") or {}, rel.get("perfil_gpu") or {}
    e1, e2, juiz, ds = rel.get("etapa1") or {}, rel.get("etapa2") or {}, rel.get("juiz") or {}, rel.get("dataset") or {}
    j082 = next((l for l in juiz.get("por_limiar", []) if abs(l["limiar"] - 0.82) < 1e-6), {})

    def coco(tarefa, m):
        return _g(e2, "coco", tarefa, m)

    linha = dict(
        id=exp_id, nome=rel.get("nome"), grupos=",".join(rel.get("grupos") or []), gpu=rel.get("gpu"),
        gpu_nome=amb.get("gpu_nome"), gpu_avaliacao=amb.get("gpu_avaliacao", rel.get("gpu")),
        modelo=p.get("modelo"), etapa1_modo=p.get("etapa1"), it1=p.get("it1"), it2=p.get("it2"), lr2=p.get("lr2"),
        aumentos=_num(p.get("aumentos")), adaptado=_num(perfil.get("adaptado")), lote1=perfil.get("lote1"),
        lote2=perfil.get("lote2"), amp=_num(perfil.get("amp")), tf32=_num(perfil.get("tf32")),
        lr2_efetivo=e2.get("lr_efetivo"), it2_efetivas=e2.get("iteracoes_efetivas"),
        inicio=rel.get("inicio"), fim=rel.get("fim"), tempo_total_s=rel.get("tempo_total_s"),
        etapa1_tempo_s=e1.get("tempo_s"), etapa1_img_s=_g(e1, "velocidade", "imagens_por_s"),
        etapa1_s_iter=_g(e1, "velocidade", "s_iter_mediana"), etapa1_data_time=_g(e1, "velocidade", "data_time_media"),
        etapa2_tempo_s=e2.get("tempo_s"), etapa2_img_s=_g(e2, "velocidade", "imagens_por_s"),
        etapa2_s_iter=_g(e2, "velocidade", "s_iter_mediana"), etapa2_s_iter_p90=_g(e2, "velocidade", "s_iter_p90"),
        etapa2_data_time=_g(e2, "velocidade", "data_time_media"),
        memoria_pico_mb=e2.get("memoria_pico_mb") or e1.get("memoria_pico_mb"),
        parametros=e2.get("parametros") or e1.get("parametros"), modelo_mb=e2.get("modelo_mb"),
        ap_bbox=coco("bbox", "AP"), ap50_bbox=coco("bbox", "AP50"), ap75_bbox=coco("bbox", "AP75"),
        ap_segm=coco("segm", "AP"), ap50_segm=coco("segm", "AP50"), ap75_segm=coco("segm", "AP75"),
        decisao_082=j082.get("taxa_decisao"), acuracia_vencedora_082=j082.get("acuracia_vencedora"),
        acuracia_balim_082=j082.get("acuracia_balim"), inferencia_ms_082=j082.get("inferencia_ms_media"),
        juiz_tempo_s=juiz.get("tempo_s"), val_imagens=ds.get("val_imagens"), train_imagens=ds.get("train_imagens"),
        train_sha1=ds.get("train_sha1"), val_sha1=ds.get("val_sha1"),
        cuda=str(amb.get("cuda")), cudnn=str(amb.get("cudnn")), torch=amb.get("torch"),
        detectron2=amb.get("detectron2"), python=amb.get("python"), cpus=amb.get("cpus"), ram_gb=amb.get("ram_gb"),
        repo_commit=amb.get("repo_commit"), observacao=rel.get("observacao"), origem=origem,
        importado_em=datetime.now().isoformat(timespec="seconds"), relatorio_json=json.dumps(rel, ensure_ascii=False),
    )
    with con:
        apagar_experimento(con, exp_id, commit=False)
        con.execute(f"INSERT INTO experimentos ({','.join(linha)}) VALUES ({','.join('?' * len(linha))})",
                    list(linha.values()))
        for etapa, dados in (("etapa1", e1), ("etapa2", e2)):
            for tarefa, metricas in (dados.get("coco") or {}).items():
                con.executemany("INSERT INTO coco VALUES (?,?,?,?,?)",
                                [(exp_id, etapa, tarefa, m, _num(v)) for m, v in metricas.items()])
            pontos = []
            for c in dados.get("curvas") or []:
                it = c.get("iteration")
                pontos += [(exp_id, etapa, it, m, _num(v)) for m, v in c.items() if m != "iteration"]
            con.executemany("INSERT OR REPLACE INTO curvas VALUES (?,?,?,?,?)", pontos)
        con.executemany(
            f"INSERT INTO juiz_limiar VALUES (?,?,{','.join('?' * len(COLUNAS_JUIZ))})",
            [(exp_id, l["limiar"], *[_num(l.get(c)) for c in COLUNAS_JUIZ]) for l in juiz.get("por_limiar", [])])
        con.executemany(
            f"INSERT OR REPLACE INTO por_foto VALUES (?,?,?,{','.join('?' * len(COLUNAS_FOTO))})",
            [(exp_id, f["foto"], f["limiar"], *[_num(f.get(c)) for c in COLUNAS_FOTO]) for f in juiz.get("por_foto", [])])
        planos = {**{f"ambiente.{k}": v for k, v in amb.items()}, **{f"perfil_gpu.{k}": v for k, v in perfil.items()},
                  **{f"dataset.{k}": v for k, v in ds.items()}, **{f"parametros.{k}": v for k, v in p.items()}}
        con.executemany("INSERT INTO ambiente VALUES (?,?,?)", [(exp_id, k, str(v)) for k, v in planos.items()])
    return exp_id


def apagar_experimento(con: sqlite3.Connection, exp_id: str, commit: bool = True):
    for t in TABELAS[:-1]:
        coluna = "id" if t == "experimentos" else "exp_id"
        con.execute(f"DELETE FROM {t} WHERE {coluna} = ?", (exp_id,))
    if commit:
        con.commit()


def importar_pasta(con: sqlite3.Connection, pasta: Path) -> list[tuple[str, str]]:
    """Importa todo relatorio.json encontrado na pasta (recursivo). Retorna [(arquivo, id ou erro)]."""
    resultado = []
    for arq in sorted(pasta.rglob("relatorio.json")):
        try:
            rel = json.loads(arq.read_text(encoding="utf-8"))
            resultado.append((str(arq.relative_to(pasta)), importar_relatorio(con, rel, origem=str(arq))))
        except Exception as e:  # relatório corrompido não derruba a importação dos outros
            resultado.append((str(arq.relative_to(pasta)), f"ERRO: {e}"))
    return resultado
