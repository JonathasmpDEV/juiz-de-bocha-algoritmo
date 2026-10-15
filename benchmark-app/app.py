"""Juiz de Bocha — benchmark de treinos. Rode com:  uv run streamlit run app.py"""
import io
import json
import os
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import controle
import db
import envios

RAIZ = Path(__file__).parent
# .env opcional (veja .env.example); variáveis já definidas no ambiente têm prioridade
if (RAIZ / ".env").exists():
    for _linha in (RAIZ / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in _linha and not _linha.lstrip().startswith("#"):
            _k, _v = _linha.split("=", 1)
            if _v.strip():
                os.environ.setdefault(_k.strip(), _v.strip().strip('"'))
PASTA_RELATORIOS = Path(os.environ.get("BENCH_RELATORIOS", RAIZ / "relatorios"))
BANCO = Path(os.environ.get("BENCH_DB", RAIZ / "dados" / "benchmark.db"))

def curto(p: Path) -> str:
    try:
        return str(p.relative_to(RAIZ))
    except ValueError:
        return str(p)


st.set_page_config(page_title="Benchmark Juiz de Bocha", page_icon="🎯", layout="wide")

# ---------- paleta (dataviz/references/palette.md, validada) ----------
CAT_CLARO = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CAT_ESCURO = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
STATUS = {"acertou": "#0ca30c", "errou": "#d03b3b", "sem decisão": "#898781"}


def tema_escuro() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


ESCURO = tema_escuro()
CAT = CAT_ESCURO if ESCURO else CAT_CLARO
TINTA = {"grade": "#2c2c2a" if ESCURO else "#e1e0d9", "eixo": "#383835" if ESCURO else "#c3c2b7", "muda": "#898781"}


def estilo(fig: go.Figure, altura=380) -> go.Figure:
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="",
                      legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
                      font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif'),
                      hoverlabel=dict(font_size=13), bargap=0.25)
    fig.update_xaxes(gridcolor=TINTA["grade"], linecolor=TINTA["eixo"], zeroline=False)
    fig.update_yaxes(gridcolor=TINTA["grade"], linecolor=TINTA["eixo"], zeroline=False)
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    fig.update_traces(line_width=2, selector=dict(type="scatter"))
    return fig


def mostrar(fig, altura=380):
    st.plotly_chart(estilo(fig, altura), use_container_width=True, theme="streamlit")


@st.cache_resource
def conexao() -> sqlite3.Connection:
    c = db.conectar(BANCO)
    envios.preparar(c)
    return c


con = conexao()


def ler(sql, params=()):
    return pd.read_sql_query(sql, con, params=params)


def cores_por(chaves):
    """Cor segue a entidade (ordem fixa e estável), nunca a posição no filtro."""
    return {k: CAT[i % len(CAT)] for i, k in enumerate(sorted(set(chaves)))}


ROTULOS = {
    "id": "Experimento", "gpu": "GPU", "modelo": "Modelo", "it2": "Iterações etapa 2", "lr2": "LR etapa 2",
    "aumentos": "Aumentos", "lote2": "Lote", "amp": "AMP", "etapa2_img_s": "Imagens/s (etapa 2)",
    "etapa1_img_s": "Imagens/s (etapa 1)", "etapa2_tempo_s": "Tempo etapa 2 (s)", "tempo_total_s": "Tempo total (s)",
    "ap_bbox": "AP caixa", "ap_segm": "AP máscara", "ap50_segm": "AP50 máscara",
    "decisao_082": "Decide (0.82)", "acuracia_vencedora_082": "Acerta vencedora (0.82)",
    "acuracia_balim_082": "Acerta balim (0.82)", "inferencia_ms_082": "Inferência (ms)",
    "memoria_pico_mb": "Memória pico (MB)",
}

# ---------- navegação: duas áreas separadas ----------
AREAS = {
    "🧪 Testes de modelos": ["Controle de treinos", "Visão geral", "Comparar experimentos", "GPUs e custo", "Por foto",
                            "Curvas de treino", "Importar", "Exportar"],
    "📷 Envios dos usuários": ["Usuários", "Envios por usuário"],
}
st.sidebar.title("🎯 Juiz de Bocha")
_area_pedida = st.query_params.get("a")
area = st.sidebar.radio("Área", list(AREAS), index=list(AREAS).index(_area_pedida) if _area_pedida in AREAS else 0)
PAGINAS = AREAS[area]
_pedida = st.query_params.get("p")
st.sidebar.divider()
pagina = st.sidebar.radio("Página", PAGINAS, index=PAGINAS.index(_pedida) if _pedida in PAGINAS else 0,
                          label_visibility="collapsed")
st.query_params["a"], st.query_params["p"] = area, pagina

if area == "📷 Envios dos usuários":
    CRED = envios.caminho_credencial(RAIZ)
    usuarios = ler("""SELECT u.uid, u.email, u.nome, u.criado_em, COUNT(e.doc_id) AS envios,
                             MIN(e.data) AS primeiro_envio, MAX(e.data) AS ultimo_envio
                      FROM usuarios u LEFT JOIN envios e ON e.uid = u.uid GROUP BY u.uid
                      UNION ALL
                      SELECT e.uid, NULL, NULL, NULL, COUNT(*), MIN(e.data), MAX(e.data) FROM envios e
                      WHERE e.uid IS NOT NULL AND e.uid NOT IN (SELECT uid FROM usuarios) GROUP BY e.uid
                      ORDER BY envios DESC""")
    todos_envios = ler("SELECT * FROM envios")
    st.sidebar.caption(f"{len(usuarios)} usuário(s) · {len(todos_envios)} envio(s) na cópia local")

    with st.sidebar.expander("Sincronizar com o Firestore", expanded=todos_envios.empty):
        if not CRED.exists():
            st.warning("Falta a chave do Firebase Admin.")
            st.markdown(f"1. Firebase Console → projeto **juizdebocha** → ⚙ Configurações → **Contas de serviço**\n"
                        f"2. **Gerar nova chave privada**\n3. Salve o arquivo como:\n\n`{CRED}`\n\n"
                        "A chave dá acesso total ao projeto: não envie para ninguém nem para o Git.")
        elif st.button("Sincronizar agora", type="primary", use_container_width=True):
            with st.spinner("Lendo users e Photos do Firestore…"):
                try:
                    r = envios.sincronizar(con, CRED)
                    st.success(f"{r['usuarios']} usuários e {r['envios']} envios copiados"
                               + (f" ({r['sem_usuario']} envios sem usuário)" if r["sem_usuario"] else ""))
                    st.rerun()
                except Exception as e:
                    st.error(f"Falha ao ler o Firestore: {e}")

    if todos_envios.empty:
        st.title("Envios dos usuários")
        st.info("Nenhum envio na cópia local ainda. Use **Sincronizar com o Firestore** na barra lateral.")
        st.stop()
    todos_envios["data"] = pd.to_datetime(todos_envios.data, errors="coerce", utc=True)

    if pagina == "Usuários":
        st.title("Usuários")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Usuários", len(usuarios))
        k2.metric("Envios", len(todos_envios))
        recentes = todos_envios[todos_envios.data >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)]
        k3.metric("Envios nos últimos 30 dias", len(recentes))
        k4.metric("Usuários ativos (30 dias)", recentes.uid.nunique())
        por_mes = todos_envios.dropna(subset=["data"]).assign(mes=lambda d: d.data.dt.strftime("%Y-%m"))
        por_mes = por_mes.groupby("mes", as_index=False).size()
        fig = px.bar(por_mes, x="mes", y="size", title="Envios por mês", labels={"mes": "", "size": "Envios"},
                     color_discrete_sequence=[CAT[0]])
        mostrar(fig, altura=320)
        st.subheader("Por usuário")
        st.caption("Clique no UID para abrir os envios dessa pessoa.")
        tabela = usuarios.assign(abrir="?a=" + area.replace(" ", "+") + "&p=Envios+por+usu%C3%A1rio&uid=" + usuarios.uid)
        st.dataframe(tabela, hide_index=True, use_container_width=True, column_config={
            "uid": "UID", "email": "E-mail", "nome": "Nome", "criado_em": "Conta criada em", "envios": "Envios",
            "primeiro_envio": "Primeiro envio", "ultimo_envio": "Último envio",
            "abrir": st.column_config.LinkColumn("Abrir", display_text="ver envios")})
        st.download_button("⬇ usuarios.csv", usuarios.to_csv(index=False).encode("utf-8-sig"), "usuarios.csv", "text/csv")

    elif pagina == "Envios por usuário":
        st.title("Envios por usuário")
        rotulo = {r.uid: f"{r.email or r.nome or 'sem e-mail'} — {r.uid} ({r.envios})" for r in usuarios.itertuples()}
        uids = list(rotulo)
        _uid = st.query_params.get("uid")
        uid = st.selectbox("Usuário (UID)", uids, index=uids.index(_uid) if _uid in uids else 0,
                           format_func=rotulo.get)
        st.query_params["uid"] = uid
        dele = todos_envios[todos_envios.uid == uid].sort_values("data", ascending=False)
        info = usuarios[usuarios.uid == uid].iloc[0]
        k1, k2, k3 = st.columns(3)
        k1.metric("Envios", len(dele))
        k2.metric("Primeiro envio", str(info.primeiro_envio or "—")[:10])
        k3.metric("Último envio", str(info.ultimo_envio or "—")[:10])
        st.caption(f"UID `{uid}` · e-mail {info.email or '—'} · conta criada em {str(info.criado_em or '—')[:10]}")

        por_pagina = 24
        paginas = max(1, -(-len(dele) // por_pagina))
        pg = st.number_input("Página da galeria", 1, paginas, 1) if paginas > 1 else 1
        bloco = dele.iloc[(pg - 1) * por_pagina: pg * por_pagina]
        cols = st.columns(4)
        for i, r in enumerate(bloco.itertuples()):
            with cols[i % 4]:
                img = r.thumb or r.url
                if img:
                    st.image(img, use_container_width=True)
                legenda = " · ".join(x for x in (r.data.strftime("%d/%m/%Y %H:%M") if pd.notna(r.data) else None,
                                                 r.local) if x)
                st.caption(legenda or r.doc_id)
                if r.url:
                    st.markdown(f"[resultado do juiz]({r.url})")

        if st.toggle("Mostrar localização dos envios no mapa", value=False,
                     help="Localização é dado pessoal — mostre só quando precisar"):
            pontos = dele.dropna(subset=["latitude", "longitude"])
            if pontos.empty:
                st.info("Esses envios não têm localização.")
            else:
                st.map(pontos.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])

        st.subheader("Tabela")
        colunas = ["doc_id", "foto_id", "data", "local", "numero_foto_usuario", "url", "thumb", "ref"]
        st.dataframe(dele[colunas], hide_index=True, use_container_width=True,
                     column_config={"url": st.column_config.LinkColumn("Resultado"),
                                    "thumb": st.column_config.ImageColumn("Miniatura")})
        c1, c2 = st.columns(2)
        c1.download_button(f"⬇ envios de {uid}.csv", dele.drop(columns=["dados_json"]).to_csv(index=False).encode("utf-8-sig"),
                           f"envios_{uid}.csv", "text/csv", use_container_width=True)
        c2.download_button("⬇ todos os envios.csv", todos_envios.drop(columns=["dados_json"]).to_csv(index=False).encode("utf-8-sig"),
                           "envios.csv", "text/csv", use_container_width=True)
    st.stop()

# ====================== área: testes de modelos ======================
NOTEBOOK_URL = "https://colab.research.google.com/github/JonathasmpDEV/juiz-de-bocha-algoritmo/blob/main/colab/JuizDeBocha_Experimentos.ipynb"
CRED = envios.caminho_credencial(RAIZ)

if pagina == "Controle de treinos":
    st.title("Controle de treinos")
    if not CRED.exists():
        st.warning("Para controlar os treinos o app precisa da chave do Firebase Admin (projeto **juizdebocha**).")
        st.markdown(f"Firebase Console → ⚙ Configurações → **Contas de serviço** → **Gerar nova chave privada** e salve como "
                    f"`{CRED}`. No Colab, cadastre o **conteúdo** do mesmo arquivo no secret `FIREBASE_ADMIN_JSON`.")
        st.stop()

    st.info(f"**Como executar:** crie os treinos abaixo e escolha a GPU. Depois abra o [notebook executor]({NOTEBOOK_URL}) "
            "no Colab **com essa GPU** (*Ambiente de execução → Alterar tipo*) e use *Executar tudo*. "
            "O Colab não permite que o app escolha a GPU ou ligue a sessão sozinho; o resto (progresso, cancelar, "
            "relatório) volta para cá automaticamente.")

    rotulo_modelo = lambda m: controle.MODELOS.get(m, m)  # noqa: E731
    with st.expander("➕ Novo treino", expanded=True):
        with st.form("novo"):
            c1, c2, c3 = st.columns(3)
            nome = c1.text_input("Nome", "meu_treino")
            modelo = c2.selectbox("Modelo", list(controle.MODELOS), format_func=rotulo_modelo)
            gpu = c3.selectbox("GPU", controle.GPUS, index=1, help="O treino só roda num Colab aberto com essa GPU")
            c4, c5, c6 = st.columns(3)
            it2 = c4.number_input("Iterações (etapa 2, bocha)", 100, 50000, 1000, step=500)
            lr2 = c5.select_slider("Taxa de aprendizado", [0.000625, 0.00125, 0.0025, 0.005, 0.01], value=0.0025)
            repeticoes = c6.number_input("Repetições", 1, 5, 1, help="Rodar mais de uma vez mede a variação entre treinos")
            c7, c8, c9 = st.columns(3)
            aumentos = c7.toggle("Aumentos de dados", value=True)
            etapa1 = c8.selectbox("Etapa 1 (COCO bolas)", ["auto", "treinar"],
                                  help="auto: reaproveita a etapa 1 já treinada desse modelo (R50 usa a da A100); "
                                       "treinar: treina de novo")
            it1 = c9.number_input("Iterações etapa 1 (se treinar)", 1000, 100000, 15000, step=1000)
            if st.form_submit_button("Colocar na fila", type="primary"):
                for r in range(int(repeticoes)):
                    controle.enfileirar(CRED, nome if repeticoes == 1 else f"{nome}_r{r + 1}", gpu, modelo, it2, lr2,
                                        aumentos, it1, etapa1)
                st.success(f"{int(repeticoes)} treino(s) na fila da {gpu}.")
    with st.expander("📋 Fila padrão do benchmark (iterações, LR, aumentos e os 4 modelos)"):
        st.dataframe(pd.DataFrame(controle.FILA_PADRAO).assign(modelo=lambda d: d.modelo.map(rotulo_modelo)),
                     hide_index=True, use_container_width=True)
        g = st.selectbox("Rodar a fila padrão na GPU", controle.GPUS[1:], key="gpu_padrao")
        if st.button(f"Colocar os {len(controle.FILA_PADRAO)} treinos na fila da {g}"):
            for e in controle.FILA_PADRAO:
                controle.enfileirar(CRED, e["nome"], g, e["modelo"], e["it2"], e["lr2"], e["aumentos"])
            st.success("Fila padrão criada.")

    @st.fragment(run_every=15)
    def painel_fila():
        try:
            jobs = controle.listar(CRED)
        except Exception as e:
            st.error(f"Não consegui ler a fila no Firestore: {e}")
            return
        # importa sozinho os relatórios dos treinos concluídos
        no_banco = set(ler("SELECT id FROM experimentos").id)
        for j in jobs:
            if j.get("status") == "concluido" and j.get("relatorio") and j["id"] not in no_banco:
                try:
                    db.importar_relatorio(con, controle.baixar_relatorio(CRED, j["relatorio"]), origem=f"firestore:{j['id']}")
                    st.toast(f"Relatório importado: {j['nome']} ({j.get('gpu')})")
                except Exception as e:
                    st.warning(f"Falha ao importar {j['id']}: {e}")
        st.subheader("Fila e histórico")
        if not jobs:
            st.caption("Nenhum treino criado ainda.")
            return
        linhas = []
        for j in jobs:
            p, pr, r = j.get("parametros") or {}, j.get("progresso") or {}, j.get("resumo") or {}
            total = pr.get("total") or 0
            linhas.append(dict(
                id=j["id"], status=controle.STATUS.get(j.get("status"), j.get("status")), nome=j.get("nome"),
                modelo=p.get("modelo"), gpu_pedida=j.get("gpu_desejada"), gpu=j.get("gpu"),
                etapa=pr.get("etapa"), progresso=(pr.get("iteracao", 0) / total) if total else None,
                loss=pr.get("loss"), it2=p.get("it2"), lr2=p.get("lr2"), aumentos=p.get("aumentos"),
                acerta_vencedora=r.get("acerta_vencedora"), ap_segm=r.get("ap_segm"),
                criado=j.get("criado_em"), concluido=j.get("concluido_em"), cancelar=j.get("cancelar"),
            ))
        df = pd.DataFrame(linhas)
        st.dataframe(df, hide_index=True, use_container_width=True, column_config={
            "id": None, "status": "Status", "nome": "Nome", "modelo": "Modelo", "gpu_pedida": "GPU pedida",
            "gpu": "Rodou na", "etapa": "Etapa",
            "progresso": st.column_config.ProgressColumn("Progresso", format="percent", min_value=0, max_value=1),
            "loss": st.column_config.NumberColumn("Loss", format="%.3f"), "it2": "Iterações", "lr2": "LR",
            "aumentos": st.column_config.CheckboxColumn("Aumentos"),
            "acerta_vencedora": st.column_config.NumberColumn("Acerta vencedora", format="percent"),
            "ap_segm": st.column_config.NumberColumn("AP máscara", format="%.1f"),
            "criado": st.column_config.DatetimeColumn("Criado", format="DD/MM HH:mm"),
            "concluido": st.column_config.DatetimeColumn("Terminou", format="DD/MM HH:mm"),
            "cancelar": st.column_config.CheckboxColumn("Cancelamento pedido")})
        st.caption("Atualiza sozinho a cada 15 s. Relatórios de treinos concluídos entram no banco automaticamente.")

        escolhido = st.selectbox("Treino", df.id, format_func=lambda i: f"{df.set_index('id').loc[i, 'nome']} — {i}")
        job = next(j for j in jobs if j["id"] == escolhido)
        a1, a2, a3, a4 = st.columns(4)
        if job.get("status") in ("na_fila", "rodando") and a1.button("■ Cancelar", use_container_width=True):
            controle.cancelar(CRED, escolhido)
            st.rerun(scope="fragment")
        nova_gpu = a2.selectbox("Rodar de novo na", controle.GPUS, index=controle.GPUS.index(job.get("gpu_desejada", "qualquer")),
                                label_visibility="collapsed")
        if a3.button("↻ Rodar novamente", use_container_width=True):
            controle.rodar_novamente(CRED, job, nova_gpu)
            st.rerun(scope="fragment")
        if job.get("status") not in ("rodando",) and a4.button("🗑 Apagar da fila", use_container_width=True):
            controle.apagar(CRED, escolhido)
            st.rerun(scope="fragment")
        if job.get("erro"):
            st.code(job["erro"], language="text")

    painel_fila()
    st.stop()

exps = ler("SELECT * FROM experimentos ORDER BY gpu, nome")
st.sidebar.caption(f"{len(exps)} experimento(s) no banco\n\n`{curto(BANCO)}`")

if exps.empty and pagina not in ("Importar", "Exportar"):
    st.title("Benchmark de treinos — Juiz de Bocha")
    st.info("O banco ainda está vazio. Vá em **Importar** e carregue os `relatorio.json` gerados pelo notebook "
            "`JuizDeBocha_Experimentos` (ficam em `00-Juiz de Bocha/experimentos/` no Drive).")
    st.stop()

# filtros globais numa linha, acima dos gráficos
if pagina not in ("Importar", "Exportar"):
    f1, f2, f3 = st.columns(3)
    gpus = f1.multiselect("GPU", sorted(exps.gpu.dropna().unique()), default=sorted(exps.gpu.dropna().unique()))
    modelos = f2.multiselect("Modelo", sorted(exps.modelo.dropna().unique()), default=sorted(exps.modelo.dropna().unique()))
    limiar = f3.select_slider("Limiar de confiança do juiz", options=[0.5, 0.6, 0.7, 0.82, 0.9], value=0.82)
    exps = exps[exps.gpu.isin(gpus) & exps.modelo.isin(modelos)]
    juiz = ler("SELECT * FROM juiz_limiar")
    juiz = juiz[juiz.exp_id.isin(exps.id)]
    no_limiar = juiz[(juiz.limiar - limiar).abs() < 1e-6].set_index("exp_id")
    exps = exps.assign(
        decide=exps.id.map(no_limiar.taxa_decisao), acerta_vencedora=exps.id.map(no_limiar.acuracia_vencedora),
        acerta_balim=exps.id.map(no_limiar.acuracia_balim), inferencia_ms=exps.id.map(no_limiar.inferencia_ms_media))
    COR_GPU = cores_por(ler("SELECT DISTINCT gpu FROM experimentos").gpu.dropna())
    COR_EXP = cores_por(ler("SELECT id FROM experimentos").id)

# ====================================================================================
if pagina == "Visão geral":
    st.title("Visão geral")
    com_juiz = exps.dropna(subset=["acerta_vencedora"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Experimentos", len(exps), help="Com os filtros acima")
    if not com_juiz.empty:
        melhor = com_juiz.sort_values(["acerta_vencedora", "decide"], ascending=False).iloc[0]
        k2.metric(f"Melhor acerto da vencedora ({limiar})", f"{melhor.acerta_vencedora:.0%}", melhor.id, delta_color="off")
        k3.metric(f"Maior taxa de decisão ({limiar})", f"{com_juiz.decide.max():.0%}",
                  com_juiz.loc[com_juiz.decide.idxmax(), "id"], delta_color="off")
    rapidos = exps.dropna(subset=["etapa2_img_s"])
    if not rapidos.empty:
        r = rapidos.loc[rapidos.etapa2_img_s.idxmax()]
        k4.metric("Treino mais rápido", f"{r.etapa2_img_s:.1f} img/s", r.id, delta_color="off")

    st.subheader("Ranking")
    cols = ["id", "gpu", "modelo", "it2", "lr2", "aumentos", "lote2", "acerta_vencedora", "acerta_balim", "decide",
            "ap_segm", "ap_bbox", "etapa2_img_s", "tempo_total_s", "inferencia_ms", "memoria_pico_mb"]
    tabela = exps[cols].sort_values(["acerta_vencedora", "ap_segm"], ascending=False)
    st.dataframe(tabela, hide_index=True, use_container_width=True, column_config={
        "id": "Experimento", "gpu": "GPU", "modelo": "Modelo", "it2": "It. etapa 2", "lr2": st.column_config.NumberColumn("LR", format="%.5f"),
        "aumentos": st.column_config.CheckboxColumn("Aumentos"), "lote2": "Lote",
        "acerta_vencedora": st.column_config.ProgressColumn(f"Acerta vencedora ({limiar})", format="percent", min_value=0, max_value=1),
        "acerta_balim": st.column_config.ProgressColumn(f"Acerta balim ({limiar})", format="percent", min_value=0, max_value=1),
        "decide": st.column_config.ProgressColumn(f"Decide ({limiar})", format="percent", min_value=0, max_value=1),
        "ap_segm": st.column_config.NumberColumn("AP máscara", format="%.1f"),
        "ap_bbox": st.column_config.NumberColumn("AP caixa", format="%.1f"),
        "etapa2_img_s": st.column_config.NumberColumn("Img/s etapa 2", format="%.1f"),
        "tempo_total_s": st.column_config.NumberColumn("Tempo total", format="%.0f s"),
        "inferencia_ms": st.column_config.NumberColumn("Inferência", format="%.0f ms"),
        "memoria_pico_mb": st.column_config.NumberColumn("Memória pico", format="%.0f MB"),
    })
    st.caption("Acerto = a bola apontada pelo modelo cai dentro da bola correspondente nas marcações revisadas "
               "(validação congelada). AP = métrica COCO (0–100) na mesma validação.")

# ====================================================================================
elif pagina == "Comparar experimentos":
    st.title("Comparar experimentos")
    escolhidos = st.multiselect("Experimentos", list(exps.id), default=list(exps.id)[:8], max_selections=8,
                                help="Até 8, para as cores continuarem distinguíveis")
    sel = exps[exps.id.isin(escolhidos)]
    if sel.empty:
        st.stop()
    sem_juiz = sel[sel.acerta_vencedora.isna()].id.tolist()
    if sem_juiz:
        st.info("Sem avaliação do juiz por foto (treinos importados só do metrics.json): "
                + ", ".join(sem_juiz) + ". Esses aparecem só nas métricas COCO, velocidade e curvas.")
    # métricas COCO: AP geral e AP75 (critério rigoroso), caixa × máscara
    coco_cols = {"ap_bbox": ("AP", "Caixa"), "ap_segm": ("AP", "Máscara"),
                 "ap75_bbox": ("AP75", "Caixa"), "ap75_segm": ("AP75", "Máscara")}
    d = sel.melt(id_vars="id", value_vars=list(coco_cols), var_name="col", value_name="valor").dropna()
    d["métrica"], d["tarefa"] = zip(*d.col.map(coco_cols)) if not d.empty else ([], [])
    c1, c2 = st.columns(2)
    for col, metrica in ((c1, "AP"), (c2, "AP75")):
        with col:
            dm = d[d["métrica"] == metrica]
            fig = px.bar(dm, x="id", y="valor", color="tarefa", barmode="group",
                         title=f"{metrica} COCO na validação (0–100)", labels={"id": "", "valor": metrica, "tarefa": ""},
                         color_discrete_map={"Caixa": CAT[0], "Máscara": CAT[1]})
            fig.update_yaxes(range=[0, 100])
            mostrar(fig)
    st.caption("AP75 exige que a caixa/máscara prevista coincida ≥ 75% com a marcada. Caixa muito abaixo da máscara "
               "indica caixas \"folgadas\" (efeito da rotação no treino original).")

    j = juiz[juiz.exp_id.isin(escolhidos)]
    com_juiz = sel.dropna(subset=["acerta_vencedora"]).sort_values("acerta_vencedora")
    if com_juiz.empty:
        st.info("Nenhum experimento selecionado tem avaliação do juiz por foto ainda — ela vem dos relatórios "
                "gerados pela fila de experimentos no Colab.")
    else:
        c3, c4, c5 = st.columns(3)
        with c3:
            fig = px.bar(com_juiz, x="acerta_vencedora", y="id", orientation="h",
                         title=f"Acerto da vencedora (limiar {limiar})", labels={"acerta_vencedora": "Acerto", "id": ""},
                         color_discrete_sequence=[CAT[0]], text=com_juiz.acerta_vencedora.map("{:.0%}".format))
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            mostrar(fig)
        for col, metrica, titulo in ((c4, "acuracia_vencedora", "Acerto da vencedora × limiar"),
                                     (c5, "taxa_decisao", "Taxa de decisão × limiar")):
            with col:
                fig = px.line(j, x="limiar", y=metrica, color="exp_id", markers=True, title=titulo,
                              labels={"limiar": "Limiar de confiança", metrica: "", "exp_id": ""},
                              color_discrete_map=COR_EXP)
                fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
                fig.update_traces(marker_size=8)
                mostrar(fig)

    st.subheader("Parâmetros lado a lado")
    campos = ["gpu", "modelo", "etapa1_modo", "it1", "it2", "lr2", "aumentos", "adaptado", "lote2", "lr2_efetivo",
              "it2_efetivas", "amp", "tf32", "tempo_total_s", "etapa2_tempo_s", "etapa2_img_s", "etapa2_s_iter",
              "memoria_pico_mb", "parametros", "modelo_mb", "ap_bbox", "ap50_bbox", "ap75_bbox", "ap_segm",
              "ap50_segm", "ap75_segm", "decide", "acerta_vencedora", "acerta_balim", "inferencia_ms",
              "torch", "cuda", "detectron2", "cpus", "ram_gb", "val_sha1"]
    st.dataframe(sel.set_index("id")[campos].T.astype(str), use_container_width=True)

# ====================================================================================
elif pagina == "GPUs e custo":
    st.title("GPUs e custo")
    st.caption("Velocidade em **imagens por segundo** (comparável mesmo com lotes diferentes no modo reativo).")
    vel = exps.melt(id_vars=["id", "gpu", "nome"], value_vars=["etapa1_img_s", "etapa2_img_s"],
                    var_name="etapa", value_name="img_s").dropna()
    vel["etapa"] = vel.etapa.map({"etapa1_img_s": "Etapa 1 (COCO)", "etapa2_img_s": "Etapa 2 (bocha)"})
    vel = vel.groupby(["gpu", "etapa"], as_index=False).img_s.median()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(vel, x="gpu", y="img_s", color="etapa", barmode="group", title="Velocidade de treino (mediana)",
                     labels={"gpu": "", "img_s": "Imagens/s"},
                     color_discrete_map={"Etapa 1 (COCO)": CAT[0], "Etapa 2 (bocha)": CAT[1]})
        mostrar(fig)
    with c2:
        inf = exps.dropna(subset=["inferencia_ms"]).groupby("gpu", as_index=False).inferencia_ms.median()
        if inf.empty:
            st.markdown("**Tempo de inferência por foto**")
            st.info("Ainda sem medição de inferência — vem da avaliação do juiz na fila de experimentos.")
        else:
            fig = px.bar(inf, x="gpu", y="inferencia_ms", title="Tempo de inferência por foto (mediana)",
                         labels={"gpu": "", "inferencia_ms": "ms por foto"}, color_discrete_sequence=[CAT[0]])
            mostrar(fig)

    st.subheader("Custo estimado")
    custos = ler("SELECT * FROM gpu_custos")
    todas = sorted(set(exps.gpu.dropna()) | set(custos.gpu))
    custos = pd.DataFrame({"gpu": todas}).merge(custos, how="left")
    st.caption("Preencha as unidades de computação do Colab por hora de cada GPU (veja no seu plano). Fica salvo no banco.")
    editado = st.data_editor(custos, hide_index=True, disabled=["gpu"], key="custos", column_config={
        "gpu": "GPU", "unidades_por_hora": st.column_config.NumberColumn("Unidades/hora", min_value=0.0, step=0.1),
        "observacao": "Observação"})
    if st.button("Salvar custos"):
        with con:
            con.execute("DELETE FROM gpu_custos")
            con.executemany("INSERT INTO gpu_custos VALUES (?,?,?)",
                            editado.where(pd.notna(editado), None).values.tolist())
        st.success("Custos salvos.")
    custo = exps.merge(editado[["gpu", "unidades_por_hora"]], on="gpu", how="left")
    custo["horas"] = custo.tempo_total_s / 3600
    custo["unidades"] = custo.horas * custo.unidades_por_hora
    st.dataframe(custo[["id", "gpu", "horas", "unidades_por_hora", "unidades", "acerta_vencedora"]],
                 hide_index=True, use_container_width=True, column_config={
                     "horas": st.column_config.NumberColumn("Horas", format="%.2f"),
                     "unidades": st.column_config.NumberColumn("Unidades gastas", format="%.2f"),
                     "acerta_vencedora": st.column_config.NumberColumn(f"Acerto ({limiar})", format="percent")})

# ====================================================================================
elif pagina == "Por foto":
    st.title("Resultado por foto")
    escolhidos = st.multiselect("Experimentos", list(exps.id), default=list(exps.id)[:8])
    fotos = ler("SELECT * FROM por_foto WHERE abs(limiar - ?) < 1e-6", (limiar,))
    fotos = fotos[fotos.exp_id.isin(escolhidos) & (fotos.ref_decidiu == 1)]
    if fotos.empty:
        st.info("Sem dados por foto para essa seleção.")
        st.stop()
    fotos["resultado"] = fotos.apply(lambda r: "sem decisão" if r.decidiu == 0
                                     else ("acertou" if r.acerto_vencedora == 1 else "errou"), axis=1)
    codigo = {"errou": 0, "sem decisão": 1, "acertou": 2}
    simbolo = {"errou": "✘", "sem decisão": "–", "acertou": "✔"}
    matriz = fotos.pivot(index="foto", columns="exp_id", values="resultado")
    fig = go.Figure(go.Heatmap(
        z=matriz.replace(codigo).values, x=list(matriz.columns), y=list(matriz.index),
        text=matriz.replace(simbolo).values, texttemplate="%{text}", xgap=2, ygap=2, showscale=False,
        colorscale=[[0, STATUS["errou"]], [0.5, STATUS["sem decisão"]], [1, STATUS["acertou"]]], zmin=0, zmax=2,
        customdata=matriz.values, hovertemplate="%{y}<br>%{x}<br><b>%{customdata}</b><extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    mostrar(fig, altura=max(380, 22 * len(matriz) + 120))
    st.caption("✔ acertou a vencedora · ✘ apontou outra bola · – não decidiu. Só fotos em que as marcações "
               "revisadas permitem decidir. Fotos que todos erram costumam ser problema da foto ou da marcação.")
    dificeis = (matriz != "acertou").sum(axis=1).sort_values(ascending=False)
    st.dataframe(dificeis.rename("experimentos que não acertaram").reset_index(), hide_index=True)
    st.subheader("Detalhe")
    st.dataframe(fotos.drop(columns=["resultado"]), hide_index=True, use_container_width=True)

# ====================================================================================
elif pagina == "Curvas de treino":
    st.title("Curvas de treino")
    curvas = ler("SELECT * FROM curvas")
    curvas = curvas[curvas.exp_id.isin(exps.id)]
    c1, c2, c3 = st.columns(3)
    etapa = c1.radio("Etapa", ["etapa2", "etapa1"], horizontal=True, format_func=lambda e: e.replace("etapa", "Etapa "))
    metricas = sorted(curvas[curvas.etapa == etapa].metrica.unique())
    metrica = c2.selectbox("Métrica", metricas, index=metricas.index("total_loss") if "total_loss" in metricas else 0)
    eixo_x = c3.radio("Eixo X", ["Iteração", "Imagens vistas"], horizontal=True,
                      help="Imagens vistas = iteração × lote; compara treinos com lotes diferentes")
    escolhidos = st.multiselect("Experimentos", sorted(curvas[curvas.etapa == etapa].exp_id.unique()),
                                default=sorted(curvas[curvas.etapa == etapa].exp_id.unique())[:8], max_selections=8)
    d = curvas[(curvas.etapa == etapa) & (curvas.metrica == metrica) & curvas.exp_id.isin(escolhidos)].copy()
    if d.empty:
        st.info("Sem curvas para essa seleção.")
        st.stop()
    lote = exps.set_index("id")["lote2" if etapa == "etapa2" else "lote1"].fillna(3 if etapa == "etapa2" else 2)
    d["x"] = d.iteracao * (d.exp_id.map(lote) if eixo_x == "Imagens vistas" else 1)
    fig = px.line(d.sort_values("x"), x="x", y="valor", color="exp_id", title=metrica,
                  labels={"x": eixo_x, "valor": "", "exp_id": ""}, color_discrete_map=COR_EXP)
    if "loss" in metrica:
        fig.update_yaxes(type="log")
    mostrar(fig, altura=460)

# ====================================================================================
elif pagina == "Importar":
    st.title("Importar relatórios")
    st.markdown(f"Coloque as pastas de `00-Juiz de Bocha/experimentos/` (com os `relatorio.json`) em "
                f"`{curto(PASTA_RELATORIOS)}/` e clique em importar, ou envie os arquivos abaixo.")
    PASTA_RELATORIOS.mkdir(exist_ok=True)
    if st.button("Importar da pasta relatorios/", type="primary"):
        res = db.importar_pasta(con, PASTA_RELATORIOS)
        if not res:
            st.warning("Nenhum relatorio.json encontrado.")
        for arq, r in res:
            (st.error if r.startswith("ERRO") else st.success)(f"{arq} → {r}")
    enviados = st.file_uploader("Ou envie relatorio.json", type="json", accept_multiple_files=True)
    if enviados and st.button("Importar arquivos enviados"):
        for f in enviados:
            try:
                st.success(f"{f.name} → {db.importar_relatorio(con, json.load(f), origem=f'upload:{f.name}')}")
            except Exception as e:
                st.error(f"{f.name}: {e}")
    st.subheader("No banco")
    lista = ler("SELECT id, gpu, modelo, origem, importado_em FROM experimentos ORDER BY importado_em DESC")
    st.dataframe(lista, hide_index=True, use_container_width=True)
    if not lista.empty:
        apagar = st.selectbox("Remover experimento do banco", [""] + list(lista.id))
        if apagar and st.button(f"Remover {apagar}"):
            db.apagar_experimento(con, apagar)
            st.rerun()

# ====================================================================================
elif pagina == "Exportar":
    st.title("Exportar dados")
    tabelas = {t: ler(f"SELECT * FROM {t}") for t in db.TABELAS}
    tabelas["experimentos"] = tabelas["experimentos"].drop(columns=["relatorio_json"])
    st.caption("Tudo que foi coletado, em CSV (UTF-8 com BOM, abre direto no Excel).")
    cols = st.columns(4)
    for i, (nome, df) in enumerate(tabelas.items()):
        cols[i % 4].download_button(f"{nome}.csv ({len(df)} linhas)", df.to_csv(index=False).encode("utf-8-sig"),
                                    file_name=f"{nome}.csv", mime="text/csv", use_container_width=True)
    pacote = io.BytesIO()
    with zipfile.ZipFile(pacote, "w", zipfile.ZIP_DEFLATED) as z:
        for nome, df in tabelas.items():
            z.writestr(f"{nome}.csv", df.to_csv(index=False).encode("utf-8-sig"))
    planilha = io.BytesIO()
    with pd.ExcelWriter(planilha, engine="openpyxl") as w:
        for nome, df in tabelas.items():
            df.to_excel(w, sheet_name=nome[:31], index=False)
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ Tudo em ZIP (CSVs)", pacote.getvalue(), "benchmark_juiz_de_bocha.zip", "application/zip",
                       type="primary", use_container_width=True)
    c2.download_button("⬇ Planilha Excel (todas as abas)", planilha.getvalue(), "benchmark_juiz_de_bocha.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    c3.download_button("⬇ Banco SQLite", BANCO.read_bytes() if BANCO.exists() else b"", "benchmark.db",
                       use_container_width=True)
    if st.button("Salvar CSVs também na pasta exportacoes/"):
        destino = RAIZ / "exportacoes"
        destino.mkdir(exist_ok=True)
        for nome, df in tabelas.items():
            df.to_csv(destino / f"{nome}.csv", index=False, encoding="utf-8-sig")
        st.success(f"Salvo em {destino}")
