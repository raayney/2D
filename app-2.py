import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import io
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def watt_sample(rng, a=0.988, b=2.249):
    g = 1.0 + (b * a / 4.0)
    x_max = a * (g + np.sqrt(g**2 - 1.0))
    while True:
        u1, u2 = rng.random(), rng.random()
        x = -a * np.log(u1)
        y = -a * np.log(u2)
        if y >= (x - x_max)**2 / (a * x_max):
            return x

def creer_grille(taille, fraction_uranium, rng):
    return rng.random((taille, taille)) < fraction_uranium

def deplacer(i, j, di, dj, taille):
    ni, nj = i + di, j + dj
    if not (0 <= ni < taille and 0 <= nj < taille):
        return None
    return (ni, nj)

def classifier_trajectoire(trajectoire, seuil=20):
    if trajectoire[-1] == 0:
        return "extinction"
    if np.max(trajectoire) >= seuil:
        return "croissance"
    return "quasi-critique"

def keff_empirique(trajectoire):
    ratios = [trajectoire[t] / trajectoire[t-1]
              for t in range(1, len(trajectoire)) if trajectoire[t-1] > 0]
    return float(np.mean(ratios)) if ratios else 0.0

def direction_aleatoire(rng):
    return DIRECTIONS[rng.integers(4)]

def direction_opposee(di, dj):
    return (-di, -dj)

def creer_neutrons_prompts(rng, n_born, i, j, wa, wb):
    return [(i, j, *direction_aleatoire(rng), float(watt_sample(rng, wa, wb)))
            for _ in range(int(n_born))]

def creer_fragments(rng, i, j, tke_total, p_min, p_max):
    di1, dj1 = direction_aleatoire(rng)
    di2, dj2 = direction_opposee(di1, dj1)
    e = max(float(tke_total) / 2.0, 0.0)
    return [
        (i, j, di1, dj1, e, int(rng.integers(p_min, p_max + 1))),
        (i, j, di2, dj2, e, int(rng.integers(p_min, p_max + 1))),
    ]

def creer_betas(rng, i, j, energie_totale, p_min, p_max):
    n = max(1, int(rng.poisson(6)))
    poids = rng.dirichlet(np.ones(n))
    return [(i, j, *direction_aleatoire(rng), float(energie_totale * w),
             int(rng.integers(p_min, p_max + 1))) for w in poids]

def identifier_paroi(i, j, di, dj, taille):
    ni, nj = i + di, j + dj
    if ni < 0:          return "haut",    nj
    if ni >= taille:    return "bas",     nj
    if nj < 0:          return "gauche",  ni
    if nj >= taille:    return "droite",  ni
    return None, None

def interaction_U235(ni, nj, di, dj, rng, p_fiss, p_abs, k,
                     wa, wb, tke_moy, tke_sig, beta_E, fp_min, fp_max, bp_min, bp_max):
    r = rng.random()
    if r < p_fiss:
        n_born = int(rng.poisson(k))
        tke = float(max(rng.normal(tke_moy, tke_sig), 0.0))
        return "fission", \
               creer_neutrons_prompts(rng, n_born, ni, nj, wa, wb), \
               creer_fragments(rng, ni, nj, tke, fp_min, fp_max), \
               creer_betas(rng, ni, nj, beta_E, bp_min, bp_max)
    if r < p_fiss + p_abs:
        return "absorption", [], [], []
    di2, dj2 = direction_aleatoire(rng)
    return "diffusion", [(ni, nj, di2, dj2)], [], []

def avancer_chargees(particules, taille, rng, carte, profil, progressif):
    restantes, q_c, q_p = [], 0.0, 0.0
    for (i, j, di, dj, E, steps) in particules:
        if steps <= 0 or E <= 0.0:
            continue
        pos = deplacer(i, j, di, dj, taille)
        if pos is None:
            cote, idx = identifier_paroi(i, j, di, dj, taille)
            q_p += E
            if cote:
                profil[cote][idx] += E
            continue
        ni, nj = pos
        if progressif:
            dE = min(E / steps, E)
            carte[ni, nj] += dE
            q_c += dE
            E_new = E - dE
        else:
            E_new = E
        steps_new = steps - 1
        if steps_new <= 0 or E_new <= 0.0:
            if not progressif and E_new > 0.0:
                carte[ni, nj] += E_new
                q_c += E_new
            continue
        restantes.append((ni, nj, di, dj, E_new, steps_new))
    return restantes, q_c, q_p

def etat_suivant(neutrons, fragments, betas, grille, rng, params):
    T = grille.shape[0]
    carte = np.zeros((T, T), dtype=float)
    profil = {c: np.zeros(T) for c in ("haut", "bas", "gauche", "droite")}
    new_n, new_f, new_b = [], [], []
    q_cn, q_pn = 0.0, 0.0

    for (i, j, di, dj, E) in neutrons:
        pos = deplacer(i, j, di, dj, T)
        if pos is None:
            cote, idx = identifier_paroi(i, j, di, dj, T)
            q_pn += E
            if cote:
                profil[cote][idx] += E
            continue
        ni, nj = pos
        if grille[ni, nj]:
            ev, n_, f_, b_ = interaction_U235(
                ni, nj, di, dj, rng,
                params["p_fiss"], params["p_abs"], params["k"],
                params["wa"], params["wb"],
                params["tke_moy"], params["tke_sig"], params["beta_E"],
                params["fp_min"], params["fp_max"], params["bp_min"], params["bp_max"])
            if ev == "fission":
                new_n.extend(n_); new_f.extend(f_); new_b.extend(b_)
            elif ev == "absorption":
                carte[ni, nj] += E; q_cn += E
            else:
                di2, dj2 = n_[0][2], n_[0][3]
                new_n.append((ni, nj, di2, dj2, E))
        else:
            new_n.append((ni, nj, di, dj, E))

    f2, q_cf, q_pf = avancer_chargees(fragments + new_f, T, rng, carte, profil, params["progressif"])
    b2, q_cb, q_pb = avancer_chargees(betas + new_b, T, rng, carte, profil, params["progressif"])

    compteurs = dict(
        q_coeur_neutrons=q_cn, q_parois_neutrons=q_pn,
        q_coeur_fragments=q_cf, q_parois_fragments=q_pf,
        q_coeur_betas=q_cb, q_parois_betas=q_pb,
    )
    return new_n, f2, b2, compteurs, carte, profil


def simuler(taille, frac_u, n_pas, seed, params, max_n=600):
    rng = np.random.default_rng(seed)
    grille = creer_grille(taille, frac_u, rng)
    centre = taille // 2
    di, dj = direction_aleatoire(rng)
    E0 = float(watt_sample(rng, params["wa"], params["wb"]))

    neutrons = [(centre, centre, di, dj, E0)]
    fragments, betas = [], []
    totaux = Counter()
    carte_acc = np.zeros((taille, taille))
    profil_acc = {c: np.zeros(taille) for c in ("haut", "bas", "gauche", "droite")}

    traj_n = [1]
    traj_c = [0]
    hist_n = []
    hist_c = []

    def snapshot():
        gn = np.zeros((taille, taille), dtype=np.float32)
        gc = np.zeros((taille, taille), dtype=np.float32)
        for p in neutrons:
            gn[p[0], p[1]] += 1
        for p in fragments + betas:
            gc[p[0], p[1]] += 1
        hist_n.append(gn)
        hist_c.append(gc)

    snapshot()

    for _ in range(n_pas):
        if not neutrons and not fragments and not betas:
            break
        if len(neutrons) > max_n:
            idx = rng.choice(len(neutrons), max_n, replace=False)
            neutrons = [neutrons[i] for i in idx]

        neutrons, fragments, betas, comp, carte_step, profil_step = etat_suivant(
            neutrons, fragments, betas, grille, rng, params)

        totaux.update(comp)
        carte_acc += carte_step
        for c in profil_acc:
            profil_acc[c] += profil_step[c]

        traj_n.append(len(neutrons))
        traj_c.append(len(fragments) + len(betas))
        snapshot()

    return dict(
        traj_n=np.array(traj_n),
        traj_c=np.array(traj_c),
        grille=grille,
        totaux=dict(totaux),
        carte=carte_acc,
        profil=profil_acc,
        hist_n=hist_n,
        hist_c=hist_c,
    )


#animation

def make_gif(res, max_frames=60, fps_ms=120):
    hist_n = res["hist_n"]
    hist_c = res["hist_c"]
    grille = res["grille"]
    traj_n = res["traj_n"]

    n_steps = len(hist_n)
    step = max(1, n_steps // max_frames)
    indices = list(range(0, n_steps, step))
    if indices[-1] != n_steps - 1:
        indices.append(n_steps - 1)

    vmax_n = max(traj_n.max(), 1)

    frames = []
    for idx in indices:
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")

        # U-235 lattice (subtle)
        ax.imshow(grille.astype(float), cmap="Blues", alpha=0.12, vmin=0, vmax=1,
                  interpolation="nearest")

        # Charged particles
        gc = hist_c[idx]
        if gc.max() > 0:
            ax.imshow(np.log1p(gc), cmap="YlOrBr", alpha=0.55,
                      vmin=0, vmax=np.log1p(gc.max() + 1), interpolation="nearest")

        # Neutrons
        gn = hist_n[idx]
        if gn.max() > 0:
            ax.imshow(np.log1p(gn), cmap="hot", alpha=0.85,
                      vmin=0, vmax=np.log1p(vmax_n + 1), interpolation="nearest")

        ax.set_title(f"Pas {idx}  —  {int(traj_n[idx])} neutrons",
                     color="white", fontsize=9, pad=4)
        ax.axis("off")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="#0e1117")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    gif_buf = io.BytesIO()
    frames[0].save(
        gif_buf, format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=fps_ms,
        loop=0,
    )
    gif_buf.seek(0)
    return gif_buf


# UI Stremalit

st.set_page_config(
    page_title="Simulateur Nucléaire",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.title("Simulateur — Réaction en Chaîne Nucléaire 2D")

# paramètres du sidebar
with st.sidebar:
    st.header("Paramètres")

    st.subheader("Grille")
    taille     = st.slider("Taille de la grille", 10, 100, 50, step=5,
                            help="Nombre de cases par côté (N×N)")
    fraction_u = st.slider("Fraction U-235", 0.0, 1.0, 0.80, step=0.05,
                            help="Probabilité qu'une case contienne de l'U-235")

    st.subheader("Interactions")
    p_fiss = st.slider("P(fission)",    0.01, 0.50, 0.10, step=0.01)
    p_abs  = st.slider("P(absorption)", 0.01, 0.60, 0.30, step=0.01)
    p_diff = max(0.0, 1.0 - p_fiss - p_abs)
    st.caption(f"→ P(diffusion) = **{p_diff:.2f}**")
    if p_fiss + p_abs > 1.0:
        st.warning("⚠️ P(fission) + P(absorption) > 1 !")

    k_n = st.slider("K neutrons / fission", 1.0, 5.0, 2.43, step=0.01)

    st.subheader("Simulation")
    n_pas = st.slider("Nombre de pas",         10, 300, 100, step=10)
    seed  = st.slider("Seed aléatoire",          0, 200,  42, step=1)

    st.subheader("Spectre de Watt (neutrons)")
    wa = st.slider("a", 0.5, 2.0, 0.988, step=0.001, format="%.3f")
    wb = st.slider("b", 1.0, 4.0, 2.249, step=0.001, format="%.3f")

    st.subheader("Fragments de fission")
    tke_moy   = st.slider("TKE moyenne (MeV)",  100.0, 220.0, 168.0, step=1.0)
    tke_sig   = st.slider("TKE écart-type (MeV)",  1.0,  30.0,   8.0, step=0.5)
    fp_min    = st.slider("Portée min fragments",    1,    5,     1)
    fp_max    = st.slider("Portée max fragments",    1,   10,     2)

    st.subheader("Électrons bêta")
    beta_E    = st.slider("Énergie bêta/fission (MeV)", 1.0, 20.0,  8.0, step=0.5)
    bp_min    = st.slider("Portée min bêta",  1, 10,  5)
    bp_max    = st.slider("Portée max bêta",  5, 30, 10)

    progressif = st.checkbox("Dépôt d'énergie progressif", value=True)

    st.divider()
    run_btn = st.button("🚀 Simuler", type="primary", use_container_width=True)


#main
if run_btn:
    params = dict(
        p_fiss=p_fiss, p_abs=p_abs, k=k_n,
        wa=wa, wb=wb,
        tke_moy=tke_moy, tke_sig=tke_sig,
        beta_E=beta_E,
        fp_min=fp_min, fp_max=fp_max,
        bp_min=bp_min, bp_max=bp_max,
        progressif=progressif,
    )

    with st.spinner("Simulation en cours…"):
        res = simuler(taille, fraction_u, n_pas, seed, params)

    traj_n = res["traj_n"]
    totaux = res["totaux"]

    # KPIs
    keff   = keff_empirique(traj_n)
    classe = classifier_trajectoire(traj_n)
    q_coeur  = sum(totaux.get(k, 0) for k in
                   ("q_coeur_neutrons", "q_coeur_fragments", "q_coeur_betas"))
    q_parois = sum(totaux.get(k, 0) for k in
                   ("q_parois_neutrons", "q_parois_fragments", "q_parois_betas"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("k_eff empirique", f"{keff:.3f}",
              delta="≈ critique" if abs(keff - 1) < 0.05 else None)
    c2.metric("Régime", classe)
    c3.metric("Max neutrons", int(traj_n.max()))
    c4.metric("Chaleur cœur",  f"{q_coeur:.1f} MeV")
    c5.metric("Chaleur parois", f"{q_parois:.1f} MeV")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Animation", "Carte thermique", "Parois", "Bilan énergétique"])

    # TAB 1: Animation
    with tab1:
        col_anim, col_traj = st.columns([1, 1], gap="large")

        with col_anim:
            st.markdown("**Évolution spatiale des particules**")
            st.caption("🔴 Neutrons (chaud) · 🟠 Particules chargées · 🔵 U-235")
            with st.spinner("Génération du GIF…"):
                gif = make_gif(res, max_frames=60)
            st.image(gif, use_column_width=True)

        with col_traj:
            st.markdown("**Populations au fil du temps**")
            fig_t = go.Figure()
            t_ax = list(range(len(traj_n)))
            fig_t.add_trace(go.Scatter(
                x=t_ax, y=traj_n.tolist(), name="Neutrons",
                line=dict(color="#ff4b4b", width=2), fill="tozeroy",
                fillcolor="rgba(255,75,75,0.10)"))
            fig_t.add_trace(go.Scatter(
                x=t_ax, y=res["traj_c"].tolist(), name="Chargées (frag+β)",
                line=dict(color="#ff8c00", width=2, dash="dot")))
            fig_t.update_layout(
                xaxis_title="Pas de temps", yaxis_title="Nb de particules",
                template="plotly_dark", height=420,
                legend=dict(orientation="h", y=1.08),
                margin=dict(t=20))
            st.plotly_chart(fig_t, use_container_width=True)

    # TAB 2 : Heatmap
    with tab2:
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.markdown("**Dépôt énergétique cumulé dans le cœur**")
            fig_h = px.imshow(
                res["carte"],
                color_continuous_scale="inferno",
                labels={"color": "Énergie (MeV)"},
                aspect="equal",
            )
            fig_h.update_layout(template="plotly_dark", height=440,
                                 margin=dict(t=10, b=10))
            st.plotly_chart(fig_h, use_container_width=True)

        with col_h2:
            st.markdown("**Grille U-235**")
            fig_g = px.imshow(
                res["grille"].astype(int),
                color_continuous_scale="Blues",
                labels={"color": "U-235"},
                aspect="equal",
                binary_string=True,
            )
            fig_g.update_layout(template="plotly_dark", height=440,
                                  margin=dict(t=10, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

    # TAB 3 : Données aux Parois
    with tab3:
        pp = res["profil"]
        colors_w = {"haut": "#e63946", "bas": "#2a9d8f",
                    "gauche": "#e9c46a", "droite": "#f4a261"}
        fig_w = go.Figure()
        for cote, col in colors_w.items():
            fig_w.add_trace(go.Bar(
                x=list(range(taille)), y=pp[cote].tolist(),
                name=f"Paroi {cote}",
                marker_color=col, opacity=0.85))
        fig_w.update_layout(
            barmode="overlay",
            xaxis_title="Indice le long de la paroi",
            yaxis_title="Énergie déposée (MeV)",
            template="plotly_dark", height=420,
            legend=dict(orientation="h", y=1.06),
            bargap=0.0)
        st.plotly_chart(fig_w, use_container_width=True)

        df_w = pd.DataFrame(
            {"Paroi": list(colors_w.keys()),
             "Total (MeV)": [float(pp[c].sum()) for c in colors_w],
             "Max local (MeV)": [float(pp[c].max()) for c in colors_w],
             "Position max": [int(pp[c].argmax()) for c in colors_w]})
        st.dataframe(df_w, use_container_width=True, hide_index=True)

    # Tab 4 : Bilan
    with tab4:
        lignes = [
            ("Cœur  — neutrons",   totaux.get("q_coeur_neutrons",   0.0)),
            ("Cœur  — fragments",  totaux.get("q_coeur_fragments",  0.0)),
            ("Cœur  — bêtas",      totaux.get("q_coeur_betas",      0.0)),
            ("Parois — neutrons",  totaux.get("q_parois_neutrons",  0.0)),
            ("Parois — fragments", totaux.get("q_parois_fragments", 0.0)),
            ("Parois — bêtas",     totaux.get("q_parois_betas",     0.0)),
        ]
        df_e = pd.DataFrame(lignes, columns=["Canal", "Énergie (MeV)"])
        total = max(df_e["Énergie (MeV)"].sum(), 1e-12)
        df_e["Fraction (%)"] = (df_e["Énergie (MeV)"] / total * 100).round(2)

        col_df, col_pie = st.columns([1, 1])
        with col_df:
            st.dataframe(df_e, use_container_width=True, hide_index=True)

            eta_c, eta_t = 0.94, 0.33
            e_elec = q_coeur * eta_c * eta_t
            st.metric("Énergie électrique estimée",
                       f"{e_elec:.2f} MeV",
                       help=f"η_collecte={eta_c}, η_thermique={eta_t}")

        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=["Cœur (neutrons)", "Cœur (fragments)", "Cœur (bêtas)",
                        "Parois (neutrons)", "Parois (fragments)", "Parois (bêtas)"],
                values=[totaux.get(k, 0) for k in (
                    "q_coeur_neutrons", "q_coeur_fragments", "q_coeur_betas",
                    "q_parois_neutrons", "q_parois_fragments", "q_parois_betas")],
                marker_colors=["#e07b39","#c1440e","#f4a261",
                                "#3a7ebf","#1b4f8a","#6baed6"],
                hole=0.42,
            ))
            fig_pie.update_layout(
                title="Répartition de l'énergie",
                template="plotly_dark", height=380,
                legend=dict(font_size=11))
            st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("Réglez les paramètres dans la barre latérale, puis cliquez sur **Simuler**.")
    st.markdown("""
    ### Que peut-on explorer ?

    | Paramètre | Effet attendu |
    |---|---|
    | **Fraction U-235 ↑** | Plus de fissions, croissance plus probable |
    | **P(fission) ↑** | k_eff monte vers 1, puis dépasse la criticité |
    | **K neutrons/fission ↑** | Multiplication plus agressive |
    | **Portée bêta ↑** | Plus d'énergie atteint les parois |
    | **TKE moyenne ↑** | Fragments plus énergétiques, chaleur cœur ↑ |
    | **Seed** | Explore la variabilité stochastique |
    """)