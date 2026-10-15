"""Gera relatórios FICTÍCIOS (marcados como exemplo) para testar o app sem rodar treinos.
Uso: uv run python gerar_exemplo.py <pasta_destino>"""
import json
import random
import sys
from pathlib import Path

destino = Path(sys.argv[1] if len(sys.argv) > 1 else "exemplo")
random.seed(7)
fotos = [f"foto_{i:02d}.jpg" for i in range(12)]
limiares = [0.5, 0.6, 0.7, 0.82, 0.9]
casos = [("base_r50_it1000", "A100", "R50", 1000, 0.0025, True, 8, 9.5),
         ("base_r50_it1000", "T4", "R50", 1000, 0.0025, True, 3, 2.1),
         ("r50_it3000", "A100", "R50", 3000, 0.0025, True, 8, 9.4),
         ("x101_it1000", "A100", "X101", 1000, 0.0025, True, 8, 5.2)]

for nome, gpu, modelo, it2, lr2, aum, lote, img_s in casos:
    exp_id = f"{nome}__{gpu}"
    qualidade = 0.6 + 0.1 * (it2 > 1000) + 0.15 * (modelo == "X101") + random.uniform(-0.05, 0.05)
    por_foto = []
    for f in fotos:
        ms = 1000 / img_s * random.uniform(0.8, 1.2) / 4
        for lim in limiares:
            decidiu = random.random() < qualidade + 0.3 - lim * 0.3
            acerto = decidiu and random.random() < qualidade
            por_foto.append(dict(foto=f, limiar=lim, largura=1600, altura=1200, inferencia_ms=ms,
                                 n_detectadas=random.randint(2, 7), decidiu=decidiu, n_bolas=random.randint(1, 6),
                                 dist_vencedora=random.uniform(20, 200), dist_segunda=random.uniform(200, 400),
                                 margem_px=random.uniform(5, 150), balim_diametro_px=random.uniform(30, 60),
                                 score_min=lim, score_medio=min(0.99, lim + 0.08), ref_n_bolas=5, ref_decidiu=True,
                                 acerto_balim=decidiu and random.random() < 0.95, acerto_vencedora=acerto if decidiu else None))
    resumo = []
    for lim in limiares:
        l = [p for p in por_foto if p["limiar"] == lim]
        resumo.append(dict(limiar=lim, fotos=len(l), taxa_decisao=sum(p["decidiu"] for p in l) / len(l),
                           acuracia_vencedora=sum(bool(p["acerto_vencedora"]) for p in l) / len(l),
                           acuracia_balim=sum(bool(p["acerto_balim"]) for p in l) / len(l), fotos_com_referencia=len(l),
                           erro_contagem_medio=1.2, bolas_media=4.1, inferencia_ms_media=l[0]["inferencia_ms"],
                           inferencia_ms_p50=l[0]["inferencia_ms"], inferencia_ms_p90=l[0]["inferencia_ms"] * 1.2))
    it_ef = round(it2 * 3 / lote)
    curvas = [dict(iteration=i, total_loss=1.8 * (0.995 ** (i / (it_ef / 600))) + random.uniform(0, 0.05),
                   loss_mask=0.5 * (0.996 ** (i / (it_ef / 600))), lr=lr2 * lote / 3, time=lote / img_s, data_time=0.01)
              for i in range(19, it_ef, 20)]
    rel = dict(id=exp_id, nome=nome, grupos=["exemplo"], gpu=gpu,
               parametros=dict(modelo=modelo, etapa1="reusar", it1=0, it2=it2, lr2=lr2, aumentos=aum),
               inicio="2026-10-06T10:00:00", fim="2026-10-06T10:20:00", tempo_total_s=it_ef * lote / img_s + 120,
               ambiente=dict(gpu=gpu, gpu_nome=f"{gpu} (exemplo)", torch="2.x", cuda="12.x", detectron2="0.6", cpus=12),
               perfil_gpu=dict(adaptado=True, lote1=lote, lote2=lote, amp=True, tf32=gpu != "T4"),
               dataset=dict(val_imagens=len(fotos), train_imagens=48, val_sha1="exemplo", train_sha1="exemplo"),
               etapa1=dict(modo="reusar"),
               etapa2=dict(iteracoes=it2, lr=lr2, aumentos=aum, tempo_s=it_ef * lote / img_s, lote=lote,
                           lr_efetivo=lr2 * lote / 3, iteracoes_efetivas=it_ef, memoria_pico_mb=lote * 1500.0,
                           parametros=44_000_000 if modelo == "R50" else 107_000_000, modelo_mb=335.0 if modelo == "R50" else 820.0,
                           velocidade=dict(lote=lote, imagens_por_s=img_s, s_iter_mediana=lote / img_s, s_iter_p90=lote / img_s * 1.2, data_time_media=0.01),
                           coco=dict(bbox=dict(AP=60 * qualidade, AP50=85 * qualidade, AP75=65 * qualidade),
                                     segm=dict(AP=58 * qualidade, AP50=84 * qualidade, AP75=63 * qualidade)),
                           curvas=curvas),
               juiz=dict(tempo_s=30.0, por_limiar=resumo, por_foto=por_foto),
               observacao="DADOS DE EXEMPLO — fictícios, só para testar o app")
    pasta = destino / exp_id
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "relatorio.json").write_text(json.dumps(rel, ensure_ascii=False), encoding="utf-8")
    print("gerado", pasta)
