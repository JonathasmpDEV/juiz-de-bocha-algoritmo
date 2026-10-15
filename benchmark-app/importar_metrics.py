"""Gera relatorio.json a partir dos metrics.json de um treino feito ANTES da fila de experimentos.
Só tem o que o Detectron2 grava no treino (curvas, velocidade, avaliação COCO final); sem avaliação do juiz por foto.

Uso: uv run python importar_metrics.py <pasta com etapa1/metrics.json e etapa2/metrics.json> <id> <gpu> [observação]"""
import json
import statistics
import sys
from pathlib import Path

LOTE = {"etapa1": 2, "etapa2": 3}  # hiperparâmetros do treino original


def ler(caminho: Path):
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()] if caminho.exists() else []


def etapa(linhas, lote):
    treino = [l for l in linhas if "time" in l]
    tempos = sorted(l["time"] for l in treino)
    aval = [l for l in linhas if any(k.startswith("segm/") for k in l)]
    coco = {}
    if aval:
        for k, v in aval[-1].items():
            if "/" in k and k.split("/")[0] in ("bbox", "segm"):
                tarefa, m = k.split("/", 1)
                coco.setdefault(tarefa, {})[m] = v
    mediana = statistics.median(tempos) if tempos else None
    periodo = (treino[1]["iteration"] - treino[0]["iteration"]) if len(treino) > 1 else 20
    return dict(
        modo="treinar", iteracoes=max((l["iteration"] for l in treino), default=-1) + 1, lote=lote, amp=False,
        tempo_s=sum(l["time"] for l in treino) * periodo if treino else None,
        velocidade=dict(lote=lote, imagens_por_s=lote / mediana if mediana else None, s_iter_mediana=mediana,
                        s_iter_media=statistics.mean(tempos) if tempos else None,
                        s_iter_p90=tempos[int(0.9 * (len(tempos) - 1))] if tempos else None,
                        data_time_media=statistics.mean(l["data_time"] for l in treino) if treino else None,
                        amostras=len(tempos)),
        coco=coco,
        curvas=[{k: v for k, v in l.items() if isinstance(v, (int, float)) and "/AP" not in k} for l in treino],
    )


def main():
    pasta, exp_id, gpu = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    obs = sys.argv[4] if len(sys.argv) > 4 else ""
    e1, e2 = etapa(ler(pasta / "etapa1" / "metrics.json"), 2), etapa(ler(pasta / "etapa2" / "metrics.json"), 3)
    e2.update(lr=0.0025, aumentos=True)
    rel = dict(
        id=exp_id, nome=exp_id.split("__")[0], grupos=["importado"], gpu=gpu,
        parametros=dict(modelo="R50", etapa1="treinar", it1=e1["iteracoes"], it2=e2["iteracoes"],
                        lr2=0.0025, aumentos=True),
        ambiente=dict(gpu=gpu), perfil_gpu=dict(adaptado=False, lote1=2, lote2=3, amp=False, tf32=False),
        etapa1=e1, etapa2=e2, juiz=None,
        tempo_total_s=sum(t for t in (e1["tempo_s"], e2["tempo_s"]) if t),
        observacao=obs or "Treino feito antes da fila; dados só do metrics.json (sem avaliação do juiz por foto). "
                          "AP da etapa 2 medido na validação do próprio treino.",
    )
    destino = Path(__file__).parent / "relatorios" / exp_id
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "relatorio.json").write_text(json.dumps(rel, ensure_ascii=False), encoding="utf-8")
    print(f"{exp_id}: etapa1 {e1['velocidade']['imagens_por_s']:.2f} img/s, ~{e1['tempo_s'] / 60:.0f} min | "
          f"etapa2 {e2['velocidade']['imagens_por_s']:.2f} img/s, ~{e2['tempo_s'] / 60:.1f} min | "
          f"AP segm {e2['coco'].get('segm', {}).get('AP', 0):.1f} -> {destino}")


if __name__ == "__main__":
    main()
