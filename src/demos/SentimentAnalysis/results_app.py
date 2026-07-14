from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
SUMMARY  = THIS_DIR / "results" / "summary.json"
FIGS_DIR = THIS_DIR / "figures"

st.set_page_config(
    page_title="Shapiq · Sentiment Results",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.mono { font-family: monospace; font-size: 0.85rem; }
.pill {
    display: inline-block; padding: 1px 8px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 600; margin-right: 4px;
}
.pill-pos  { background: #d1fae5; color: #065f46; }
.pill-neg  { background: #fee2e2; color: #991b1b; }
.finding {
    background: #f8fafc; border-left: 3px solid #6366f1;
    border-radius: 0 6px 6px 0; padding: 10px 14px;
    margin-bottom: 10px; font-size: 0.9rem; line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> list[dict]:
    if not SUMMARY.exists():
        return []
    return json.loads(SUMMARY.read_text(encoding="utf-8"))

def fig_path(run_key: str, kind: str) -> Path | None:
    p = FIGS_DIR / f"{run_key}_{kind}.png"
    return p if p.exists() else None

PHENOMENA = ["negative", "positive", "negation", "sarcasm"]
LANGS     = ["en", "fr", "de"]
FLAGS     = {"en": "🇬🇧", "fr": "🇫🇷", "de": "🇩🇪"}
EXPECTED  = {"negative": "NEGATIVE", "positive": "POSITIVE",
             "negation": "POSITIVE", "sarcasm": "NEGATIVE"}
PH_ICONS  = {"negative": "👎", "positive": "👍", "negation": "🔄", "sarcasm": "😏"}

def label_badge(label: str, correct: bool) -> str:
    icon = "✓" if correct else "✗"
    css  = "pill-pos" if correct else "pill-neg"
    return f'<span class="pill {css}">{label} {icon}</span>'

def is_correct(run: dict) -> bool:
    return run["label"] == EXPECTED.get(run["phenomenon"], "?")

def top_pairs(run: dict, n: int = 5) -> list[tuple[str, str, float]]:
    d = run.get("sii_exact") or run.get("sii_approx") or {}
    order2 = d.get("order2", {})
    ranked = sorted(order2.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
    result = []
    for key, val in ranked:
        parts = key.split("__", 1)
        if len(parts) == 2:
            result.append((parts[0], parts[1], val))
    return result

def top_svs(run: dict, n: int = 8) -> list[tuple[str, float]]:
    d = run.get("sv_exact") or run.get("sv_approx") or {}
    order1 = d.get("order1", {})
    return sorted(order1.items(), key=lambda x: abs(x[1]), reverse=True)[:n]

# ── Sidebar ────────────────────────────────────────────────────────────────────
data = load_data()

with st.sidebar:
    st.markdown("## 🔬 Results Explorer")
    st.caption("EN · FR · DE · 24 sentences")
    if not data:
        st.error(f"`summary.json` not found at:\n`{SUMMARY}`")
        st.stop()
    st.caption(
        f"**{len(data)}** runs · "
        f"**{sum(1 for r in data if r['model_type']=='encoder')}** encoder · "
        f"**{sum(1 for r in data if r['model_type']=='decoder')}** decoder"
    )
    st.divider()
    page = st.radio(
        "Section",
        ["📖 Story", "📊 Accuracy", "🔁 Negation", "😏 Sarcasm", "🔍 Explorer"],
        label_visibility="collapsed",
    )

# ════════════════════════════════════════════════════════════════════════════════
# STORY
# ════════════════════════════════════════════════════════════════════════════════
if page == "📖 Story":
    st.markdown("## What did we find?")
    st.caption("24 sentences × 4 phenomena × EN/FR/DE × 2 models — "
               "XLM-RoBERTa (encoder) + Qwen 3.5-4B (decoder)")
    st.divider()
    st.markdown("""
<div class="finding">
    <b>Finding 1  Encoder predictions were more consistent</b><br>
    XLM-RoBERTa handled clear positive and negative sentiment more reliably,
    while Qwen sometimes classified obviously negative sentences as positive.
</div>

<div class="finding">
    <b>Finding 2  Interactions explain negation better</b><br>
    Expressions such as <i>“not bad,” “pas mauvais,”</i> and
    <i>“nicht schlecht”</i> become positive through the interaction between
    the negation and the negative adjective.
</div>

<div class="finding">
    <b>Finding 3  Sarcasm was the most difficult phenomenon</b><br>
    Both models were often influenced by positive surface words and missed
    the intended negative meaning, although the interaction values still
    revealed conflicting word combinations.
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# ACCURACY
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Accuracy":
    st.markdown("## Label accuracy")
    st.caption("For negation, expected label is POSITIVE. For sarcasm, expected label is NEGATIVE (model should understand the flip).")
    st.divider()

    col_enc, col_dec = st.columns(2)
    for col, mtype, mname, mid in [
        (col_enc, "encoder", "XLM-RoBERTa", "cardiffnlp/twitter-xlm-roberta-base-sentiment"),
        (col_dec, "decoder", "Qwen 3.5-4B",  "Qwen/Qwen3.5-4B"),
    ]:
        with col:
            st.markdown(f"#### {mname}")
            st.caption(f"`{mid}`")
            runs = [r for r in data if r["model_type"] == mtype]
            total_c = sum(1 for r in runs if is_correct(r))
            st.metric("Overall", f"{total_c}/{len(runs)}")
            st.divider()
            for ph in PHENOMENA:
                ph_runs = [r for r in runs if r["phenomenon"] == ph]
                c, n    = sum(1 for r in ph_runs if is_correct(r)), len(ph_runs)
                st.markdown(f"**{PH_ICONS[ph]} {ph.capitalize()}**")
                st.progress(c / n if n else 0, text=f"{c}/{n}")
                wrong = [r["text"] for r in ph_runs if not is_correct(r)]
                if wrong:
                    with st.expander(f"  {len(wrong)} wrong label(s)"):
                        for t in wrong:
                            st.caption(f"• {t}")
                st.markdown("")

# ════════════════════════════════════════════════════════════════════════════════
# NEGATION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔁 Negation":
    st.markdown("## Negation — k-SII captures what SV misses")
    st.caption("The negation word alone has modest SV. Its k-SII with the sentiment word is "
               "consistently the dominant pair across EN, FR, and DE in both models.")
    st.divider()

    neg_runs = [r for r in data if r["phenomenon"] == "negation"]
    grouped: dict[str, dict] = {}
    for r in neg_runs:
        rid = r["run_id"]
        if rid not in grouped:
            grouped[rid] = {"text": r["text"], "lang": r["lang"], "topic": r["topic"]}
        grouped[rid][r["model_type"]] = r

    for rid, g in grouped.items():
        enc_r = g.get("encoder")
        dec_r = g.get("decoder")
        enc_pairs = top_pairs(enc_r) if enc_r else []
        dec_pairs = top_pairs(dec_r) if dec_r else []
        enc_top   = enc_pairs[0] if enc_pairs else None
        dec_top   = dec_pairs[0] if dec_pairs else None

        with st.container(border=True):
            st.markdown(f"{FLAGS.get(g['lang'],'')} **[{g['lang'].upper()}]** {g['text']}")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("**Encoder** (XLM-RoBERTa)")
                if enc_r and enc_top:
                    sign = "+" if enc_top[2] > 0 else ""
                    ok = "✓" if is_correct(enc_r) else "✗"
                    st.markdown(
                        f"k-SII(`{enc_top[0]}` + `{enc_top[1]}`) = "
                        f"**{sign}{enc_top[2]:.3f}** · label `{enc_r['label']}` {ok}"
                    )
                    st.progress(min(abs(enc_top[2]) / 3.5, 1.0))
            with c2:
                st.caption("**Decoder** (Qwen 3.5-4B)")
                if dec_r and dec_top:
                    sign = "+" if dec_top[2] > 0 else ""
                    ok = "✓" if is_correct(dec_r) else "✗"
                    st.markdown(
                        f"k-SII(`{dec_top[0]}` + `{dec_top[1]}`) = "
                        f"**{sign}{dec_top[2]:.3f}** · label `{dec_r['label']}` {ok}"
                    )
                    st.progress(min(abs(dec_top[2]) / 3.5, 1.0))

            with st.expander("📊 Show plots"):
                tabs = st.tabs(["Encoder", "Decoder"])
                for tab, mtype, run in zip(tabs, ["encoder", "decoder"], [enc_r, dec_r]):
                    with tab:
                        if run is None:
                            st.caption("No data")
                            continue
                        rk = run["run_key"]
                        fc1, fc2, fc3 = st.columns(3)
                        for col, kind, cap in [
                            (fc1, "sentence", "Sentence (SV)"),
                            (fc2, "network",  "Network (k-SII)"),
                            (fc3, "heatmap",  "Heatmap (k-SII)"),
                        ]:
                            fp = fig_path(rk, kind)
                            with col:
                                if fp:
                                    st.image(str(fp), caption=cap, use_container_width=True)
                                else:
                                    st.caption(f"_{cap} not found_")

# ════════════════════════════════════════════════════════════════════════════════
# SARCASM
# ════════════════════════════════════════════════════════════════════════════════
elif page == "😏 Sarcasm":
    st.markdown("## Sarcasm — spread interactions, near-zero accuracy")
    st.caption("Both models fail 5/6 times. The one correct case is an accident of "
               "literal negative vocabulary, not genuine sarcasm detection.")
    st.divider()

    sar_runs = [r for r in data if r["phenomenon"] == "sarcasm"]
    enc_c = sum(1 for r in sar_runs if r["model_type"] == "encoder" and is_correct(r))
    dec_c = sum(1 for r in sar_runs if r["model_type"] == "decoder" and is_correct(r))

    m1, m2, m3 = st.columns(3)
    m1.metric("Encoder accuracy", f"{enc_c}/6")
    m2.metric("Decoder accuracy", f"{dec_c}/6")
    m3.metric("Only correct case", "FR film (both)")
    st.divider()

    st.markdown("#### 🔍 The French outlier — accidentally correct")
    fr_enc = next((r for r in sar_runs if r["lang"] == "fr"
                   and r["topic"] == "film" and r["model_type"] == "encoder"), None)
    if fr_enc:
        st.markdown(f"> *{fr_enc['text']}*")
        svs = top_svs(fr_enc, n=3)
        st.caption("Top SVs (encoder): " + " · ".join(f"`{w}` = {v:+.3f}" for w, v in svs))
        st.caption("ennuyeux dominates negatively — model reads literal content, not irony.")

    st.divider()
    st.markdown("#### All sarcasm results")

    for lang in LANGS:
        st.markdown(f"**{FLAGS[lang]} {lang.upper()}**")
        for r in [x for x in sar_runs if x["lang"] == lang]:
            correct = is_correct(r)
            badge = label_badge(r["label"], correct)
            mtype = "ENC" if r["model_type"] == "encoder" else "DEC"
            pairs = top_pairs(r, n=3)
            pairs_str = " · ".join(f"`{p[0]}+{p[1]}`: {p[2]:+.2f}" for p in pairs) if pairs else "—"
            st.markdown(
                f"{badge} **[{mtype}]** {r['text']}<br>"
                f"<span class='mono'>top k-SII: {pairs_str}</span>",
                unsafe_allow_html=True,
            )

        with st.expander(f"📊 Plots — sarcasm {lang.upper()}"):
            for topic in ["film", "service"]:
                r = next((x for x in sar_runs if x["lang"] == lang
                          and x["topic"] == topic
                          and x["model_type"] == "encoder"), None)
                if r:
                    st.caption(f"**{topic}** (encoder)")
                    fc1, fc2, fc3 = st.columns(3)
                    for col, kind in [(fc1, "sentence"), (fc2, "network"), (fc3, "heatmap")]:
                        fp = fig_path(r["run_key"], kind)
                        with col:
                            if fp:
                                st.image(str(fp), use_container_width=True)
                            else:
                                st.caption("_not found_")
        st.markdown("")

# ════════════════════════════════════════════════════════════════════════════════
# EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explorer":
    st.markdown("## Explorer")
    st.caption("Select any sentence to see its SV, k-SII values, and shapiq plots.")
    st.divider()

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        ph_sel = st.selectbox("Phenomenon", ["All"] + PHENOMENA,
                              format_func=lambda x: f"{PH_ICONS.get(x,'')} {x}" if x != "All" else x)
    with fc2:
        lang_sel = st.selectbox("Language", ["All"] + [f"{FLAGS[l]} {l.upper()}" for l in LANGS])
    with fc3:
        model_sel = st.selectbox("Model", ["encoder", "decoder"])

    lang_val = lang_sel.split()[-1].lower() if lang_sel != "All" else "All"
    runs = [r for r in data if r["model_type"] == model_sel]
    if ph_sel   != "All": runs = [r for r in runs if r["phenomenon"] == ph_sel]
    if lang_val != "All": runs = [r for r in runs if r["lang"] == lang_val]

    if not runs:
        st.info("No results match this filter.")
        st.stop()

    options = {f"{FLAGS.get(r['lang'],'')} [{r['lang'].upper()}] {r['text']}": r for r in runs}
    selected_key = st.selectbox(f"{len(runs)} sentence(s)", list(options.keys()),
                                label_visibility="visible")
    r  = options[selected_key]
    rk = r["run_key"]
    correct = is_correct(r)

    st.divider()

    hc1, hc2, hc3, hc4, hc5 = st.columns([3, 1, 1, 1, 1])
    hc1.markdown(f"**{r['text']}**")
    hc1.caption(f"{FLAGS.get(r['lang'],'')} {r['lang'].upper()} · {r['phenomenon']} · {r['topic']} · {r['model_type']}")
    hc2.metric("Label",    r["label"])
    hc3.metric("Expected", EXPECTED.get(r["phenomenon"], "?"))
    hc4.metric("Correct?", "✓" if correct else "✗")
    hc5.metric("n words",  r["n_players"])

    st.markdown("#### Shapiq plots")
    fp_s = fig_path(rk, "sentence")
    fp_n = fig_path(rk, "network")
    fp_h = fig_path(rk, "heatmap")

    if not any([fp_s, fp_n, fp_h]):
        st.warning(f"No figures found for `{rk}`. Run `build_diagrams.py` first.")
    else:
        pc1, pc2, pc3 = st.columns(3)
        if fp_s: pc1.image(str(fp_s), caption="Sentence (SV)",       use_container_width=True)
        if fp_n: pc2.image(str(fp_n), caption="Network (k-SII)",     use_container_width=True)
        if fp_h: pc3.image(str(fp_h), caption="Heatmap (k-SII)",     use_container_width=True)

    st.divider()
    nc1, nc2 = st.columns(2)

    with nc1:
        st.markdown("#### Top Shapley Values")
        svs = top_svs(r, n=10)
        if svs:
            max_sv = max(abs(v) for _, v in svs)
            for word, val in svs:
                icon = "🟢" if val > 0 else "🔴"
                st.markdown(f"{icon} `{word}` &nbsp; `{val:+.4f}`", unsafe_allow_html=True)
                st.progress(abs(val) / max_sv if max_sv else 0)
        else:
            st.caption("No SV data.")

    with nc2:
        st.markdown("#### Top k-SII pairs")
        pairs = top_pairs(r, n=8)
        if pairs:
            max_p = max(abs(v) for _, _, v in pairs)
            for w1, w2, val in pairs:
                icon = "🟢" if val > 0 else "🔵"
                kind = "synergy" if val > 0 else "redundancy"
                st.markdown(f"{icon} `{w1}` + `{w2}` &nbsp; `{val:+.4f}` _{kind}_",
                            unsafe_allow_html=True)
                st.progress(abs(val) / max_p if max_p else 0)
        else:
            st.caption("No k-SII data.")

    st.divider()
    with st.expander("Technical details"):
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("v(full)",      f"{r['v_full']:+.4f}")
        sc2.metric("v(∅) baseline", f"{r['v_empty']:+.4f}")
        sc3.metric("Exact skipped", "yes" if r.get("exact_skipped") else "no")
        n = r.get("n_players", 0)
        if n > 20:
            sc4.metric("Budget", str(r["budget"]))
            st.caption("Budget = 512: approximation used only for n > 20 players).")
                
       # if r.get("timing"):
        #    st.caption("Timing (s): " + " · ".join(f"{k}: {v:.2f}" for k, v in r["timing"].items()))
        #st.caption(f"`run_key`: `{rk}`")
        st.caption(f"Model: `{r['model_id']}`")
# ══════════════════════════════════════════════════════════════════════════
# Instant Analysis launcher
# ──────────────────────────────────────────────────────────────────────────
# Launches the dynamic app (app.py) as a second Streamlit server on
# LIVE_PORT and links to it. The link opens in a new browser tab, so the
# precomputed results stay open in this one.
#
# Behaviour:
#   - If the live app is already running  -> show only the link button.
#   - If not                              -> show a start button that
#     spawns it, then show the link.
#
# Note: local-demo pattern only. For a deployed version (e.g. HF Spaces),
# use Streamlit's native multipage layout instead of subprocess.
# ══════════════════════════════════════════════════════════════════════════

import socket
import subprocess
import sys

LIVE_APP  = THIS_DIR / "app.py"
LIVE_PORT = 8502  # results app: 8501 · live app: 8502


def is_port_in_use(port: int) -> bool:
    """Return True if a server is already listening on localhost:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0


def launch_live_app() -> None:
    """Spawn app.py as an independent Streamlit server on LIVE_PORT."""
    subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(LIVE_APP),
            "--server.port", str(LIVE_PORT),
            "--server.headless", "true",
        ],
        cwd=str(THIS_DIR),
    )


def render_instant_analysis_launcher() -> None:
    """Render the launcher section in the sidebar."""
    st.divider()
    st.markdown("### ⚡ Instant Analysis")
    st.caption("Live SV / k-SII computation on any sentence you type.")

    if is_port_in_use(LIVE_PORT):
        st.link_button(
            "Open Instant Analysis ↗",
            f"http://localhost:{LIVE_PORT}",
            use_container_width=True,
        )
    else:
        if st.button("🚀 Start Instant Analysis", use_container_width=True):
            launch_live_app()
            st.success("Starting — give it ~5 seconds, then click below.")
            st.link_button(
                "Open Instant Analysis ↗",
                f"http://localhost:{LIVE_PORT}",
                use_container_width=True,
            )


# Call inside your sidebar block:
with st.sidebar:
    render_instant_analysis_launcher()