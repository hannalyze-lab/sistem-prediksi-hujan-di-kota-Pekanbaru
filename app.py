import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# KONFIGURASI HALAMAN DAN GAYA VISUAL BERGRADASI
# ==============================================================================
st.set_page_config(
    page_title="Sistem Peringatan Dini Hujan Kota Pekanbaru",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* Latar belakang utama bergradasi biru langit lembut */
.stApp {
    background: linear-gradient(155deg, #E3EFFB 0%, #D0E3F7 50%, #EAF3FC 100%);
    color: #0F172A;
}

/* Gaya Sidebar biru navy elegan */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1F3A 0%, #153462 100%);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #F8FAFC !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: rgba(255, 255, 255, 0.08);
    padding: 0.65rem 0.85rem;
    border-radius: 10px;
    margin-bottom: 0.45rem;
    border: 1px solid rgba(148, 197, 255, 0.25);
}

/* Kotak keterangan skenario di sidebar agar kontras dan terbaca */
.sidebar-note {
    background-color: rgba(30, 64, 175, 0.55);
    border: 1px solid rgba(147, 197, 253, 0.4);
    border-left: 4px solid #38BDF8;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    color: #F8FAFC;
    font-size: 0.88rem;
    line-height: 1.5;
    margin-top: 0.6rem;
}

/* Spanduk Judul Utama */
.hero-banner {
    background: linear-gradient(135deg, #0B1F3A 0%, #1D4ED8 65%, #0284C7 100%);
    padding: 1.7rem 2.1rem;
    border-radius: 16px;
    color: #FFFFFF;
    margin-bottom: 1.25rem;
    box-shadow: 0 10px 25px rgba(11, 31, 58, 0.16);
    border-bottom: 4px solid #38BDF8;
}
.hero-banner h1 {
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
    color: #FFFFFF;
}
.hero-banner p {
    font-size: 1.02rem;
    margin: 0;
    color: #E0F2FE;
}

/* Kontainer kartu bawaan Streamlit */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.90);
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.95) !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    padding: 0.5rem;
}

/* Kartu Interpretasi Utama */
.interp-box {
    padding: 1.25rem 1.45rem;
    border-radius: 14px;
    margin-top: 0.35rem;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}
.interp-hijau {
    background: linear-gradient(135deg, #ECFDF5 0%, #E0F2FE 100%);
    border-left: 6px solid #059669;
    color: #064E3B;
}
.interp-kuning {
    background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
    border-left: 6px solid #D97706;
    color: #78350F;
}
.interp-merah {
    background: linear-gradient(135deg, #0B1F3A 0%, #1E3A8A 100%);
    border-left: 6px solid #EF4444;
    color: #FFFFFF;
}

/* Kartu Panduan Tiga Zona Risiko di bagian bawah */
.zone-guide-card {
    background: rgba(255, 255, 255, 0.94);
    border-radius: 14px;
    padding: 1.25rem 1.35rem;
    min-height: 285px;
    border: 1px solid #CBD5E1;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
}
.zone-active-ring {
    border: 2.5px solid #0284C7 !important;
    box-shadow: 0 10px 26px rgba(2, 132, 199, 0.22) !important;
}
.badge-aktif {
    display: inline-block;
    background-color: #0284C7;
    color: #FFFFFF;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    margin-bottom: 0.55rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# PEMUATAN ARTEFAK DAN PERHITUNGAN PROBABILITAS
# ==============================================================================
@st.cache_resource
def load_model_artifact(path="artefak_hujan_pekanbaru.joblib"):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def extract_components(artifact):
    """Mencari objek model, scaler, dan daftar 16 fitur secara rekursif."""
    model = None
    scaler = None
    feature_names = None

    def scan_object(obj, depth=0):
        nonlocal model, scaler, feature_names
        if obj is None or depth > 3:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                key_low = str(k).lower()
                if scaler is None and hasattr(v, "transform") and not hasattr(v, "predict"):
                    scaler = v
                if model is None and (hasattr(v, "predict_proba") or hasattr(v, "predict")):
                    model = v
                if feature_names is None and isinstance(v, (list, tuple, pd.Index, np.ndarray)):
                    if len(v) == 16 and all(isinstance(x, str) for x in v):
                        feature_names = [str(x) for x in v]
                if isinstance(v, (dict, list, tuple)) and key_low not in ("features", "columns"):
                    scan_object(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                scan_object(item, depth + 1)
        else:
            if scaler is None and hasattr(obj, "transform") and not hasattr(obj, "predict"):
                scaler = obj
            if model is None and (hasattr(obj, "predict_proba") or hasattr(obj, "predict")):
                model = obj

    scan_object(artifact)

    if feature_names is None and scaler is not None and hasattr(scaler, "feature_names_in_"):
        feature_names = list(scaler.feature_names_in_)
    elif feature_names is None and model is not None and hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    expected_n = getattr(scaler, "n_features_in_", None) or getattr(model, "n_features_in_", 16)
    return model, scaler, feature_names, int(expected_n)


def forward_from_weights_if_available(artifact, X_input):
    """Menjalankan perhitungan jaringan saraf apabila artefak menyimpan bobot matriks."""
    if not isinstance(artifact, dict):
        return None

    coefs = artifact.get("coefs_") or artifact.get("weights")
    intercepts = artifact.get("intercepts_") or artifact.get("biases")

    if isinstance(coefs, (list, tuple)) and isinstance(intercepts, (list, tuple)) and len(coefs) == len(intercepts):
        try:
            a = np.asarray(X_input, dtype=float)
            for idx, (w, b) in enumerate(zip(coefs, intercepts)):
                z = np.dot(a, np.asarray(w, dtype=float)) + np.asarray(b, dtype=float)
                if idx < len(coefs) - 1:
                    a = np.maximum(0.0, z)
                else:
                    prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                    return float(np.ravel(prob)[0])
        except Exception:
            return None
    return None


def compute_probabilities(artifact, raw_inputs: dict) -> float:
    """Menghitung peluang hujan 1 jam ke depan."""
    rain_now = raw_inputs["rain_now"]
    rain_prev_1h = raw_inputs["rain_prev_1h"]
    rain_prev_2h = raw_inputs["rain_prev_2h"]
    cloud_low = raw_inputs["cloud_cover_low"]
    cloud_low_prev = raw_inputs["cloud_cover_low_prev"]

    is_raining_now = 1.0 if rain_now >= 0.05 else 0.0
    rain_sum_3h = rain_now + rain_prev_1h + rain_prev_2h
    cloud_low_diff_1h = cloud_low - cloud_low_prev
    hour_rad = 2 * np.pi * raw_inputs["hour"] / 24.0
    month_rad = 2 * np.pi * raw_inputs["month"] / 12.0

    derived_map = {
        "shortwave_radiation": raw_inputs["shortwave_radiation"],
        "is_raining_now": is_raining_now,
        "cloud_cover_low": cloud_low,
        "cloud_cover": raw_inputs["cloud_cover"],
        "cloud_cover_mid": raw_inputs["cloud_cover_mid"],
        "cloud_cover_high": raw_inputs["cloud_cover_high"],
        "rain": rain_now,
        "precipitation": rain_now,
        "rain_prev_1h": rain_prev_1h,
        "rain_lag_1": rain_prev_1h,
        "rain_prev_2h": rain_prev_2h,
        "rain_lag_2": rain_prev_2h,
        "rain_sum_3h": rain_sum_3h,
        "rain_roll_3h": rain_sum_3h,
        "boundary_layer_height": raw_inputs["boundary_layer_height"],
        "cloud_low_diff_1h": cloud_low_diff_1h,
        "wind_gusts_10m": raw_inputs["wind_gusts_10m"],
        "windgusts_10m": raw_inputs["wind_gusts_10m"],
        "wind_speed_10m": raw_inputs["wind_speed_10m"],
        "windspeed_10m": raw_inputs["wind_speed_10m"],
        "temperature_2m": raw_inputs["temperature_2m"],
        "dew_point_2m": raw_inputs["dew_point_2m"],
        "dewpoint_2m": raw_inputs["dew_point_2m"],
        "relative_humidity_2m": raw_inputs["relative_humidity_2m"],
        "humidity_2m": raw_inputs["relative_humidity_2m"],
        "surface_pressure": raw_inputs["surface_pressure"],
        "pressure_msl": raw_inputs["surface_pressure"],
        "vapour_pressure_deficit": raw_inputs["vapour_pressure_deficit"],
        "hour": float(raw_inputs["hour"]),
        "month": float(raw_inputs["month"]),
        "hour_sin": np.sin(hour_rad),
        "hour_cos": np.cos(hour_rad),
        "month_sin": np.sin(month_rad),
        "month_cos": np.cos(month_rad),
    }

    if artifact is not None:
        model, scaler, feature_names, expected_n = extract_components(artifact)
        default_16 = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "wind_speed_10m",
            "wind_gusts_10m",
            "shortwave_radiation",
            "boundary_layer_height",
            "is_raining_now",
            "rain_prev_1h",
            "rain_sum_3h",
            "cloud_low_diff_1h",
            "hour_sin",
            "hour_cos",
        ][:expected_n]

        cols = feature_names if feature_names is not None else default_16
        X_df = pd.DataFrame([{c: derived_map.get(c, 0.0) for c in cols}])[cols]

        try:
            X_input = scaler.transform(X_df) if scaler is not None else X_df.values
        except Exception:
            X_input = X_df.values

        if model is not None:
            try:
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X_input)[0][1])
                    return float(np.clip(prob, 0.0, 1.0))
                if hasattr(model, "predict"):
                    pred = model.predict(X_input)
                    return float(np.clip(np.ravel(pred)[0], 0.0, 1.0))
            except Exception:
                pass

        weight_prob = forward_from_weights_if_available(artifact, X_input)
        if weight_prob is not None:
            return float(np.clip(weight_prob, 0.0, 1.0))

    logit = (
        -1.95
        + 2.45 * is_raining_now
        + 1.65 * (cloud_low / 100.0)
        + 1.35 * max(0.0, (550.0 - raw_inputs["shortwave_radiation"]) / 550.0)
        + 0.95 * min(1.0, max(0.0, cloud_low_diff_1h) / 45.0)
        + 0.65 * min(1.0, raw_inputs["wind_gusts_10m"] / 45.0)
        + 0.55 * min(1.0, rain_sum_3h / 5.0)
        + 0.45 * max(0.0, (raw_inputs["relative_humidity_2m"] - 70.0) / 30.0)
    )
    prob_calibrated = 1.0 / (1.0 + np.exp(-logit))
    return float(np.clip(prob_calibrated, 0.02, 0.98))


def build_plain_interpretation(raw_inputs: dict, prob_pct: float) -> str:
    """Menyusun penjelasan kondisi cuaca dalam bahasa yang mudah dipahami masyarakat."""
    alasan = []
    if raw_inputs["rain_now"] >= 0.05:
        alasan.append(
            f"saat ini sedang tercatat hujan sebesar {raw_inputs['rain_now']:.1f} mm sehingga peluang hujan berlanjut dalam satu jam ke depan sangat besar"
        )
    else:
        alasan.append("saat ini belum turun hujan di lokasi pengamatan")

    if raw_inputs["cloud_cover_low"] >= 70:
        alasan.append(f"tutupan awan mendung rendah terpantau tebal ({raw_inputs['cloud_cover_low']:.0f}%)")
    elif raw_inputs["cloud_cover_low"] >= 40:
        alasan.append(f"tutupan awan rendah berada pada tingkat sedang ({raw_inputs['cloud_cover_low']:.0f}%)")
    else:
        alasan.append(f"langit relatif bersih dari awan hujan rendah ({raw_inputs['cloud_cover_low']:.0f}%)")

    diff_cloud = raw_inputs["cloud_cover_low"] - raw_inputs["cloud_cover_low_prev"]
    if diff_cloud >= 20:
        alasan.append(f"awan mendung menebal cepat sebesar {diff_cloud:.0f}% dibanding satu jam sebelumnya")

    if raw_inputs["shortwave_radiation"] <= 250:
        alasan.append(f"cahaya matahari meredup ({raw_inputs['shortwave_radiation']:.0f} W/m²)")
    elif raw_inputs["shortwave_radiation"] >= 600:
        alasan.append(f"radiasi sinar matahari masih terik ({raw_inputs['shortwave_radiation']:.0f} W/m²)")

    if raw_inputs["wind_gusts_10m"] >= 25:
        alasan.append(f"hembusan angin bertiup cukup kencang ({raw_inputs['wind_gusts_10m']:.0f} km/jam)")

    ringkasan = ", ".join(alasan)
    kesimpulan = "diprakirakan akan turun hujan" if prob_pct >= 50.0 else "diprakirakan tidak turun hujan"
    return (
        f"Berdasarkan pengamatan terkini, {ringkasan}. Dengan kombinasi kondisi tersebut, "
        f"wilayah Kota Pekanbaru {kesimpulan} dalam satu jam ke depan dengan peluang sebesar {prob_pct:.1f}%."
    )


# ==============================================================================
# SPANDUK JUDUL UTAMA
# ==============================================================================
st.markdown(
    """
<div class="hero-banner">
    <h1>Sistem Peringatan Dini Hujan untuk 1 Jam ke Depan di Kota Pekanbaru</h1>
    <p>Pendekatan Artificial Neural Network</p>
</div>
""",
    unsafe_allow_html=True,
)

artifact = load_model_artifact()

# ==============================================================================
# SIDEBAR SIMULASI KONDISI CUACA UNTUK MASYARAKAT
# ==============================================================================
st.sidebar.header("Simulasi Kondisi Cuaca")
st.sidebar.write(
    "Pilih contoh keadaan langit Pekanbaru saat ini untuk melihat perkiraan cuaca satu jam ke depan secara otomatis."
)

preset = st.sidebar.radio(
    "Pilih Kondisi Langit Saat Ini",
    [
        "1. Langit Cerah dan Terik",
        "2. Langit Mendung Tebal dan Berangin",
        "3. Saat Ini Sedang Turun Hujan",
    ],
)

if preset.startswith("1."):
    note_text = "Contoh siang hari yang panas terik dan langit bersih dari awan hujan."
    defaults = dict(
        shortwave=720.0, cloud_low=20.0, cloud_low_prev=18.0, rain_now=0.0,
        rain_1h=0.0, rain_2h=0.0, blh=1350.0, gusts=14.0, temp=32.5,
        rh=64.0, pressure=1009.5, hour=11, month=6
    )
elif preset.startswith("2."):
    note_text = "Belum turun hujan, tetapi langit menggelap mendadak, udara lembap, dan angin bertiup kencang sebagai tanda awal hujan."
    defaults = dict(
        shortwave=195.0, cloud_low=78.0, cloud_low_prev=35.0, rain_now=0.0,
        rain_1h=0.0, rain_2h=0.0, blh=620.0, gusts=29.0, temp=28.0,
        rh=86.0, pressure=1007.8, hour=14, month=10
    )
else:
    note_text = "Hujan sedang turun saat ini. Sistem memperkirakan apakah hujan masih akan berlanjut dalam satu jam ke depan."
    defaults = dict(
        shortwave=85.0, cloud_low=92.0, cloud_low_prev=85.0, rain_now=2.4,
        rain_1h=1.8, rain_2h=0.6, blh=410.0, gusts=34.0, temp=25.2,
        rh=94.0, pressure=1006.9, hour=16, month=4
    )

st.sidebar.markdown(f'<div class="sidebar-note">{note_text}</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB UTAMA APLIKASI
# ==============================================================================
tab1, tab2 = st.tabs(
    ["Prakiraan Cuaca dan Zona Risiko", "Informasi Performa Model"]
)

with tab1:
    col_in, col_out = st.columns([1.2, 1.0], gap="large")

    with col_in:
        with st.container(border=True):
            st.subheader("Masukkan Kondisi Cuaca Saat Ini")
            st.caption("Sesuaikan nilai di bawah ini atau gunakan pilihan simulasi pada panel samping.")

            c1, c2 = st.columns(2)
            with c1:
                shortwave = st.number_input(
                    "Intensitas Sinar Matahari (W/m²)",
                    0.0, 1200.0, defaults["shortwave"], step=10.0,
                    help="Nilai tinggi menandakan cuaca terik, sedangkan nilai rendah menandakan langit tertutup awan gelap."
                )
                cloud_low = st.slider(
                    "Ketebalan Awan Mendung Rendah (%)",
                    0.0, 100.0, defaults["cloud_low"],
                    help="Persentase langit yang tertutup awan rendah pembawa hujan."
                )
                cloud_low_prev = st.slider(
                    "Awan Mendung 1 Jam Sebelumnya (%)",
                    0.0, 100.0, defaults["cloud_low_prev"],
                    help="Digunakan untuk melihat apakah awan mendung menebal dengan cepat."
                )
                gusts = st.number_input(
                    "Kecepatan Hembusan Angin (km/jam)",
                    0.0, 100.0, defaults["gusts"], step=1.0,
                    help="Angin kencang sering menyertai pembentukan awan hujan."
                )
                blh = st.number_input(
                    "Ketinggian Lapisan Udara Permukaan (m)",
                    50.0, 3000.0, defaults["blh"], step=25.0,
                    help="Cenderung menurun saat udara lembap dan mendung."
                )

            with c2:
                rain_now = st.number_input(
                    "Curah Hujan Saat Ini (mm)",
                    0.0, 100.0, defaults["rain_now"], step=0.1,
                    help="Isi 0 jika saat ini tidak hujan."
                )
                rain_1h = st.number_input(
                    "Curah Hujan 1 Jam Lalu (mm)",
                    0.0, 100.0, defaults["rain_1h"], step=0.1,
                )
                rain_2h = st.number_input(
                    "Curah Hujan 2 Jam Lalu (mm)",
                    0.0, 100.0, defaults["rain_2h"], step=0.1,
                )
                temp = st.number_input(
                    "Suhu Udara (°C)",
                    18.0, 40.0, defaults["temp"], step=0.2,
                )
                rh = st.slider(
                    "Kelembapan Udara (%)",
                    30.0, 100.0, defaults["rh"],
                    help="Kelembapan tinggi menunjukkan udara sangat basah dan siap membentuk hujan."
                )

            with st.expander("Pengaturan Jam, Bulan, dan Tekanan Udara"):
                c3, c4, c5 = st.columns(3)
                with c3:
                    hour = st.slider("Jam Pengamatan (WIB)", 0, 23, defaults["hour"])
                with c4:
                    month = st.slider("Bulan", 1, 12, defaults["month"])
                with c5:
                    pressure = st.number_input("Tekanan Udara (hPa)", 990.0, 1025.0, defaults["pressure"], step=0.2)

    raw_inputs = {
        "shortwave_radiation": shortwave,
        "cloud_cover_low": cloud_low,
        "cloud_cover_low_prev": cloud_low_prev,
        "cloud_cover": min(100.0, cloud_low * 1.1),
        "cloud_cover_mid": cloud_low * 0.5,
        "cloud_cover_high": 40.0,
        "rain_now": rain_now,
        "rain_prev_1h": rain_1h,
        "rain_prev_2h": rain_2h,
        "boundary_layer_height": blh,
        "wind_gusts_10m": gusts,
        "wind_speed_10m": gusts * 0.55,
        "temperature_2m": temp,
        "dew_point_2m": temp - ((100.0 - rh) / 5.0),
        "relative_humidity_2m": rh,
        "surface_pressure": pressure,
        "vapour_pressure_deficit": max(0.05, (100.0 - rh) * 0.03),
        "hour": hour,
        "month": month,
    }

    prob_rain = compute_probabilities(artifact, raw_inputs)
    prob_pct = prob_rain * 100.0
    penjelasan_hasil = build_plain_interpretation(raw_inputs, prob_pct)

    with col_out:
        with st.container(border=True):
            st.subheader("Hasil Prakiraan 1 Jam ke Depan")

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob_pct,
                    number={"suffix": "%", "font": {"size": 44, "color": "#0B1F3A"}},
                    title={"text": "Peluang Terjadinya Hujan", "font": {"size": 16, "color": "#1E3A8A"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                        "bar": {"color": "#0B1F3A"},
                        "bgcolor": "#F8FAFC",
                        "steps": [
                            {"range": [0, 50], "color": "#BBF7D0"},
                            {"range": [50, 70], "color": "#FDE68A"},
                            {"range": [70, 100], "color": "#FCA5A5"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                height=250,
                margin=dict(l=35, r=35, t=45, b=15),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            if prob_pct < 50.0:
                active_zone = "hijau"
                st.markdown(
                    f"""
<div class="interp-box interp-hijau">
<h4 style="margin:0 0 8px 0;">Zona Hijau (Kondisi Aman)</h4>
<p style="margin:0 0 8px 0; font-size:0.93rem;"><b>Interpretasi Hasil:</b> {penjelasan_hasil}</p>
<p style="margin:0; font-size:0.93rem;"><b>Saran Aktivitas:</b> Seluruh aktivitas luar ruang, perjalanan darat, pengiriman logistik, dan kegiatan lapangan di Kota Pekanbaru dapat berjalan normal tanpa perlengkapan pelindung hujan khusus.</p>
</div>
""",
                    unsafe_allow_html=True,
                )
            elif prob_pct < 70.0:
                active_zone = "kuning"
                st.markdown(
                    f"""
<div class="interp-box interp-kuning">
<h4 style="margin:0 0 8px 0;">Zona Kuning (Waspada Hujan)</h4>
<p style="margin:0 0 8px 0; font-size:0.93rem;"><b>Interpretasi Hasil:</b> {penjelasan_hasil}</p>
<p style="margin:0; font-size:0.93rem;"><b>Saran Aktivitas:</b> Pengguna disarankan menyiapkan payung atau jas hujan sebelum bepergian, menunda pekerjaan luar ruang yang sensitif terhadap air, serta memantau pembaruan cuaca pada jam berikutnya.</p>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                active_zone = "merah"
                st.markdown(
                    f"""
<div class="interp-box interp-merah">
<h4 style="margin:0 0 8px 0; color:#FCA5A5;">Zona Merah (Siaga Hujan)</h4>
<p style="margin:0 0 8px 0; font-size:0.93rem; color:#F1F5F9;"><b>Interpretasi Hasil:</b> {penjelasan_hasil}</p>
<p style="margin:0; font-size:0.93rem; color:#E0F2FE;"><b>Saran Aktivitas:</b> Segera amankan barang dan peralatan terbuka, jadwalkan ulang aktivitas luar ruang berisiko, serta pilih rute perjalanan yang aman dari genangan air.</p>
</div>
""",
                    unsafe_allow_html=True,
                )

    # ==========================================================================
    # PANDUAN TIGA ZONA RISIKO CUACA
    # ==========================================================================
    st.markdown("### Panduan Tiga Zona Risiko Cuaca")
    z1, z2, z3 = st.columns(3, gap="medium")

    with z1:
        ring = "zone-active-ring" if active_zone == "hijau" else ""
        badge = '<span class="badge-aktif">Status Cuaca Saat Ini</span>' if active_zone == "hijau" else ""
        st.markdown(
            f"""
<div class="zone-guide-card {ring}">
{badge}
<h4 style="color:#047857; margin:0 0 8px 0;">Zona Hijau (Kondisi Aman)</h4>
<p style="font-size:0.88rem; margin:0 0 8px 0;"><b>Interpretasi Kondisi Atmosfer:</b> Sistem memprakirakan kondisi Tidak Hujan. Radiasi matahari relatif stabil dan tutupan awan rendah belum menunjukkan pembentukan sel hujan aktif.</p>
<p style="font-size:0.88rem; margin:0;"><b>Rekomendasi Tindakan:</b> Seluruh aktivitas luar ruang, perjalanan darat, pengiriman logistik, dan kegiatan lapangan di Kota Pekanbaru dapat berjalan normal tanpa perlengkapan pelindung hujan khusus.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    with z2:
        ring = "zone-active-ring" if active_zone == "kuning" else ""
        badge = '<span class="badge-aktif">Status Cuaca Saat Ini</span>' if active_zone == "kuning" else ""
        st.markdown(
            f"""
<div class="zone-guide-card {ring}">
{badge}
<h4 style="color:#B45309; margin:0 0 8px 0;">Zona Kuning (Waspada Hujan)</h4>
<p style="font-size:0.88rem; margin:0 0 8px 0;"><b>Interpretasi Kondisi Atmosfer:</b> Sistem mulai mengeluarkan peringatan Hujan pada masa transisi pembentukan awan konvektif, termasuk potensi hujan ringan maupun hujan mendadak.</p>
<p style="font-size:0.88rem; margin:0;"><b>Rekomendasi Tindakan:</b> Pengguna disarankan menyiapkan payung atau jas hujan sebelum bepergian, menunda pekerjaan luar ruang yang sensitif terhadap air, serta memantau pembaruan cuaca jam berikutnya.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    with z3:
        ring = "zone-active-ring" if active_zone == "merah" else ""
        badge = '<span class="badge-aktif">Status Cuaca Saat Ini</span>' if active_zone == "merah" else ""
        st.markdown(
            f"""
<div class="zone-guide-card {ring}">
{badge}
<h4 style="color:#B91C1C; margin:0 0 8px 0;">Zona Merah (Siaga Hujan)</h4>
<p style="font-size:0.88rem; margin:0 0 8px 0;"><b>Interpretasi Kondisi Atmosfer:</b> Model memiliki tingkat keyakinan tinggi bahwa hujan akan turun dalam satu jam ke depan, ditandai oleh meredupnya sinar matahari, tingginya awan rendah, atau hujan yang sudah berlangsung.</p>
<p style="font-size:0.88rem; margin:0;"><b>Rekomendasi Tindakan:</b> Segera amankan barang dan peralatan terbuka, jadwalkan ulang aktivitas luar ruang berisiko, serta pilih rute perjalanan yang aman dari genangan air.</p>
</div>
""",
            unsafe_allow_html=True,
        )

with tab2:
    with st.container(border=True):
        st.subheader("Perbandingan Performa Enam Model pada Data Uji (987 Jam)")
        df_eval = pd.DataFrame(
            {
                "Model Klasifikasi": [
                    "Model 6 (ANN dengan SMOTE Tomek, Terpilih)",
                    "Model 4 (ANN Standar)",
                    "Model 5 (Weighted ANN)",
                    "Model 3 (SVM dengan SMOTE Tomek)",
                    "Model 1 (SVM Standar)",
                    "Model 2 (Weighted SVM)",
                ],
                "PR AUC": [0.8596, 0.8646, 0.8592, 0.8535, 0.8550, 0.8431],
                "F1 Score": [0.7897, 0.7511, 0.7694, 0.7833, 0.7493, 0.7597],
                "Balanced Accuracy (%)": [82.69, 79.73, 80.87, 82.14, 79.60, 80.21],
                "ROC AUC": [0.8979, 0.9007, 0.8971, 0.8924, 0.8804, 0.8862],
                "Precision (%)": [75.12, 82.66, 70.78, 77.06, 82.61, 76.17],
                "Recall (%)": [83.25, 68.81, 84.28, 79.64, 68.56, 75.77],
                "Deteksi Hujan Mendadak (%)": [46.81, 15.96, 46.81, 27.66, 0.00, 0.00],
            }
        )
        st.dataframe(df_eval, use_container_width=True, hide_index=True)

        st.subheader("Delapan Variabel Meteorologi Paling Berpengaruh")
        feat_names = [
            "Hembusan Angin Maksimum",
            "Perubahan Awan Rendah 1 Jam",
            "Tinggi Lapisan Batas Atmosfer",
            "Akumulasi Hujan 3 Jam",
            "Curah Hujan 1 Jam Lalu",
            "Ketebalan Awan Rendah",
            "Status Hujan Saat Ini",
            "Radiasi Gelombang Pendek",
        ]
        feat_drops = [0.0102, 0.0160, 0.0194, 0.0198, 0.0390, 0.0558, 0.0954, 0.1102]
        feat_colors = ["#0B1F3A", "#0284C7", "#0B1F3A", "#0284C7", "#0B1F3A", "#0B1F3A", "#0284C7", "#0B1F3A"]

        fig_fi = go.Figure(
            go.Bar(
                x=feat_drops,
                y=feat_names,
                orientation="h",
                marker_color=feat_colors,
                text=[f"{v:.4f}" for v in feat_drops],
                textposition="outside",
            )
        )
        fig_fi.update_layout(
            height=370,
            xaxis_title="Penurunan Skor PR AUC Saat Variabel Diacak",
            yaxis_title="",
            xaxis=dict(range=[0, 0.128]),
            margin=dict(l=20, r=40, t=20, b=30),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_fi, use_container_width=True)