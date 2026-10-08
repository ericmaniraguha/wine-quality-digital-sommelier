"""🍷 Digital Sommelier — Streamlit front-end for the wine quality model.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from wine_features import MODEL_FEATURES, RAW_FEATURES, engineer_features

MODEL_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Digital Sommelier", page_icon="🍷", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "wine_quality_model.joblib")
    card = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))
    ranges = json.loads((MODEL_DIR / "feature_ranges.json").read_text(encoding="utf-8"))
    return model, card, ranges


try:
    model, card, ranges = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Run sections 16–17 of the notebook first.")
    st.stop()

LABELS = {
    "fixed_acidity": ("Fixed acidity", "g(tartaric)/dm³"),
    "volatile_acidity": ("Volatile acidity", "g(acetic)/dm³ — high = vinegar fault"),
    "citric_acid": ("Citric acid", "g/dm³ — freshness"),
    "residual_sugar": ("Residual sugar", "g/dm³"),
    "chlorides": ("Chlorides", "g(NaCl)/dm³ — saltiness"),
    "free_sulfur_dioxide": ("Free SO₂", "mg/dm³ — active protection"),
    "total_sulfur_dioxide": ("Total SO₂", "mg/dm³"),
    "density": ("Density", "g/cm³"),
    "ph": ("pH", "acidity scale"),
    "sulphates": ("Sulphates", "g(K₂SO₄)/dm³"),
    "alcohol": ("Alcohol", "% vol"),
}

# ----------------------------------------------------------------- sidebar
st.sidebar.title("🍷 Lab measurements")
wine_type = st.sidebar.radio("Wine type", ["red", "white"], horizontal=True)

if st.sidebar.button("↻ Reset to typical " + wine_type):
    for f in RAW_FEATURES:
        st.session_state[f] = float(ranges[f][f"{wine_type}_median"])
    st.rerun()

values = {}
for f in RAW_FEATURES:
    label, help_text = LABELS[f]
    r = ranges[f]
    values[f] = st.sidebar.slider(
        label,
        min_value=float(r["min"]),
        max_value=float(r["max"]),
        value=float(st.session_state.get(f, r[f"{wine_type}_median"])),
        step=float(r["step"]),
        help=help_text,
        key=f,
    )

# ----------------------------------------------------------------- predict
sample = pd.DataFrame([{**values, "wine_type": wine_type}])
features = engineer_features(sample)[MODEL_FEATURES]
score = float(np.clip(model.predict(features)[0], 0, 10))

if score >= 7:
    band, colour, verdict = "Premium", "#2A9D8F", "Reserve-label candidate — send to the tasting panel."
elif score >= 6:
    band, colour, verdict = "Good", "#7FA650", "Solid commercial quality."
elif score >= 5:
    band, colour, verdict = "Average", "#D9A441", "Standard table wine — meets the bar, no more."
else:
    band, colour, verdict = "Below standard", "#C1483A", "Investigate for faults before bottling."

st.title("🍷 Digital Sommelier")
st.caption(
    f"Model: **{card['model_name']}** · test RMSE {card['test_metrics']['rmse']:.3f} · "
    f"R² {card['test_metrics']['r2']:.3f} · "
    f"{card['test_metrics']['within_half_point_pct']:.0f}% of bottles predicted within ±0.5 of the panel"
)

left, right = st.columns([1, 1.35])

with left:
    st.markdown(
        f"""
        <div style="background:{colour}18;border:2px solid {colour};border-radius:14px;
                    padding:26px;text-align:center">
          <div style="font-size:13px;letter-spacing:.12em;text-transform:uppercase;opacity:.75">
            predicted quality
          </div>
          <div style="font-size:66px;font-weight:800;line-height:1.05;color:{colour}">
            {score:.1f}<span style="font-size:24px;opacity:.6"> / 10</span>
          </div>
          <div style="font-size:19px;font-weight:700;color:{colour}">{band}</div>
          <div style="font-size:14px;opacity:.8;margin-top:8px">{verdict}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(score / 10, 1.0))
    st.caption(
        f"±{card['test_metrics']['mae']:.2f} typical error. Screening aid — "
        "it does not replace a sensory panel."
    )

with right:
    st.subheader("How this wine compares")
    rows = []
    for f in RAW_FEATURES:
        r = ranges[f]
        ref = r[f"{wine_type}_median"]
        delta = values[f] - ref
        pct = 100 * delta / ref if ref else 0.0
        rows.append({
            "Measurement": LABELS[f][0],
            "This wine": round(values[f], 3),
            f"Typical {wine_type}": round(ref, 3),
            "Difference": f"{pct:+.0f}%",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.divider()

# ----------------------------------------------------------------- guidance
st.subheader("Sommelier's notes")
notes = []
if values["volatile_acidity"] > 0.6:
    notes.append(("⚠️", "Volatile acidity above 0.6 g/dm³ — vinegar fault risk. "
                        "Check sanitation and oxygen exposure; this is the strongest negative driver."))
elif values["volatile_acidity"] < 0.3:
    notes.append(("✅", "Volatile acidity is clean — no acetic fault indicated."))

if values["alcohol"] < 10:
    notes.append(("⚠️", "Alcohol under 10% vol. Alcohol is the strongest positive correlate of "
                        "panel scores; it reflects fruit ripeness at harvest."))
elif values["alcohol"] > 12:
    notes.append(("✅", "Alcohol above 12% vol — in the band where premium scores concentrate."))

if values["free_sulfur_dioxide"] < 10:
    notes.append(("⚠️", "Free SO₂ below 10 mg/dm³ — under-protected against oxidation."))
elif values["free_sulfur_dioxide"] > 50:
    notes.append(("⚠️", "Free SO₂ above 50 mg/dm³ — tasters may perceive sulphur."))
else:
    notes.append(("✅", "Free SO₂ is in the protective 10–50 mg/dm³ range."))

if values["residual_sugar"] > 45:
    notes.append(("ℹ️", "Residual sugar above 45 g/dm³ — this is a sweet wine; the model has seen "
                        "few of these, so treat the estimate with extra caution."))

for icon, text in notes:
    st.markdown(f"{icon} {text}")

with st.expander("Model card · limitations · how to read this number"):
    st.write(f"**Trained on** {card['training_rows']:,} wines · "
             f"{len(card['feature_order'])} features")
    st.write("**Most influential measurements:** " + ", ".join(card["top_features"]))
    st.write("**Limitations**")
    for lim in card["known_limitations"]:
        st.write(f"- {lim}")
    st.json(card["test_metrics"])
