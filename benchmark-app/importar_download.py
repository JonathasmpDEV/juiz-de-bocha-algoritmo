"""Importa relatório(s) baixados do Drive pelo conector (arquivo JSON {content: base64, ...}) para relatorios/ e o banco.
Uso: uv run python importar_download.py <arquivo_baixado> [<outro> ...]"""
import base64
import json
import sys
from pathlib import Path

import db

RAIZ = Path(__file__).parent
con = db.conectar(RAIZ / "dados" / "benchmark.db")
for caminho in sys.argv[1:]:
    baixado = json.loads(Path(caminho).read_text(encoding="utf-8"))
    rel = json.loads(base64.b64decode(baixado["content"]))
    destino = RAIZ / "relatorios" / rel["id"] / "relatorio.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(rel, ensure_ascii=False), encoding="utf-8")
    db.importar_relatorio(con, rel, origem=f"drive:{baixado.get('id')}")
    j = next((l for l in (rel.get("juiz") or {}).get("por_limiar", []) if abs(l["limiar"] - 0.82) < 1e-6), {})
    e2 = rel.get("etapa2") or {}
    print(f"{rel['id']}: importado | GPU {rel.get('gpu')} | lote {e2.get('lote')} | "
          f"{(e2.get('velocidade') or {}).get('imagens_por_s', 0):.1f} img/s | "
          f"AP caixa {(e2.get('coco') or {}).get('bbox', {}).get('AP', 0):.1f} | "
          f"AP máscara {(e2.get('coco') or {}).get('segm', {}).get('AP', 0):.1f} | "
          f"decide {j.get('taxa_decisao', 0):.0%} | acerta vencedora {j.get('acuracia_vencedora') or 0:.0%}")
