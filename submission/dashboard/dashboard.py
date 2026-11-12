from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ------------------------------------------------------------------ Konfigurasi halaman
st.set_page_config(page_title="Dashboard E-Commerce Olist", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

NAVY, BLUE, GREY, RED, GREEN = "#1B2A49", "#2E5EAA", "#C9CED6", "#D1495B", "#2A9D8F"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "font.size": 10.5, "axes.labelcolor": "#475467", "xtick.color": "#475467",
                     "ytick.color": "#475467", "figure.facecolor": "white", "axes.facecolor": "white",
                     "grid.color": "#EAECF0"})

DELAY_BINS = [-np.inf, -8, -1, 0, 3, 7, np.inf]
DELAY_LABELS = ["Lebih awal >7 hari", "Lebih awal 1-7 hari", "Hari estimasi",
                "Telat 1-3 hari", "Telat 4-7 hari", "Telat >7 hari"]
SEGMENT_ORDER = ["New High-Value", "Dormant High-Value", "Cooling Down",
                 "New Customers", "Lost / Low-Value", "Repeat Buyers"]
SEGMENT_ACTION = {
    "New High-Value": "Dorong pembelian kedua dalam 90 hari pertama",
    "Dormant High-Value": "Kampanye win-back dengan voucher",
    "Cooling Down": "Pengingat dan penawaran terbatas",
    "New Customers": "Rekomendasi produk otomatis (biaya rendah)",
    "Lost / Low-Value": "Pesan massal berbiaya rendah, tanpa insentif mahal",
    "Repeat Buyers": "Pertahankan lewat program loyalitas",
}

# ------------------------------------------------------------------ Gaya (CSS)
st.markdown("""
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1320px;}
footer {visibility: hidden;}

.hero {background: linear-gradient(120deg, #1B2A49 0%, #2E5EAA 100%); padding: 1.5rem 1.9rem;
       border-radius: 16px; margin-bottom: 1.1rem;}
.hero h1 {color: #FFFFFF; font-size: 1.75rem; margin: 0; padding: 0; line-height: 1.25;}
.hero p {color: #D6E2F7; margin: .45rem 0 0 0; font-size: .95rem;}
.hero .tag {display: inline-block; background: rgba(255,255,255,.16); color: #FFFFFF; border-radius: 999px;
            padding: .12rem .7rem; font-size: .75rem; margin: .7rem .4rem 0 0;}

.kpi {background: #FFFFFF; border: 1px solid #E4E7EC; border-left: 5px solid var(--accent, #2E5EAA);
      border-radius: 12px; padding: .85rem 1.05rem; box-shadow: 0 1px 3px rgba(16,24,40,.07); height: 100%;}
.kpi .label {font-size: .72rem; color: #667085; text-transform: uppercase; letter-spacing: .05em; font-weight: 600;}
.kpi .value {font-size: 1.6rem; font-weight: 700; color: #1B2A49; line-height: 1.35;}
.kpi .sub {font-size: .76rem; color: #667085;}

div[data-testid="stSlider"] {max-width: 460px;}
.insight {background: #EEF4FF; border-left: 4px solid #2E5EAA; padding: .75rem 1rem; border-radius: 8px;
          color: #1B2A49; margin: .3rem 0 .9rem 0; font-size: .93rem; line-height: 1.5;}
.insight.warn {background: #FDECEF; border-left-color: #D1495B;}

button[data-baseweb="tab"] {font-weight: 600; font-size: .98rem;}
h3 {padding-top: .3rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ Data & fungsi bantu
@st.cache_data
def load_data() -> pd.DataFrame:
    path = Path(__file__).parent / "main_data.csv"
    return pd.read_csv(path, parse_dates=["order_purchase_timestamp"])


def format_real(value: float) -> str:
    """Format nilai Real Brasil (R$) dengan pemisah ribuan."""
    return f"R$ {value:,.0f}"


def format_real_m(value: float) -> str:
    """Format nilai Real Brasil dalam juta."""
    return f"R$ {value / 1e6:,.2f} jt"


def kpi_card(col, label: str, value: str, sub: str, accent: str = BLUE) -> None:
    col.markdown(f'<div class="kpi" style="--accent:{accent}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def insight(text: str, kind: str = "info") -> None:
    cls = "insight warn" if kind == "warn" else "insight"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def show_figure(fig) -> None:
    with st.container(border=True):
        st.pyplot(fig)
    plt.close(fig)


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung skor RFM per pelanggan unik dan tentukan segmen berdasarkan aturan bisnis."""
    cust = (df.groupby("customer_unique_id")
            .agg(last_purchase=("order_purchase_timestamp", "max"), frequency=("order_id", "nunique"),
                 price_sum=("price", "sum"), freight_sum=("freight_value", "sum")))
    cust["monetary"] = cust["price_sum"] + cust["freight_sum"]
    snapshot = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    cust["recency"] = (snapshot - cust["last_purchase"]).dt.days
    cust["R_score"] = pd.qcut(cust["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    cust["M_score"] = pd.qcut(cust["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def segment(r):
        if r["frequency"] >= 2:
            return "Repeat Buyers"
        if r["R_score"] >= 4:
            return "New High-Value" if r["M_score"] >= 4 else "New Customers"
        if r["R_score"] == 3:
            return "Cooling Down"
        return "Dormant High-Value" if r["M_score"] >= 4 else "Lost / Low-Value"

    cust["segment"] = cust.apply(segment, axis=1)
    return cust


main_df = load_data()

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Filter Data")
    min_date = main_df["order_purchase_timestamp"].min().date()
    max_date = main_df["order_purchase_timestamp"].max().date()
    date_range = st.date_input("Rentang tanggal pembelian", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)
    all_states = sorted(main_df["customer_state"].unique())
    all_toggle = st.checkbox("Tampilkan semua state", value=True)
    if all_toggle:
        states = all_states
    else:
        states = st.multiselect("Pilih state pelanggan", all_states, default=["SP", "RJ", "MG"])
    st.divider()
    with st.expander("ℹ️ Tentang dashboard", expanded=False):
        st.markdown("- **Produk & Revenue**: kategori dengan revenue terbesar dan tren bulanan\n"
                    "- **Pengiriman & Kepuasan**: dampak keterlambatan pada review\n"
                    "- **Segmentasi RFM**: kelompok pelanggan dan prioritas retensi\n"
                    "- **Peta Geospasial**: sebaran pelanggan dan late rate per state")
    st.caption("Data: pesanan berstatus *delivered*, Jan 2017 - Agu 2018 (Olist, Brasil).")

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
else:
    start = end = date_range[0] if isinstance(date_range, (tuple, list)) else date_range

mask = (main_df["order_purchase_timestamp"].dt.date.between(start, end)) & (main_df["customer_state"].isin(states))
df = main_df[mask]

# ------------------------------------------------------------------ Hero
st.markdown(f"""
<div class="hero">
  <h1>📊 Dashboard Analisis E-Commerce Olist</h1>
  <p>Kategori produk, keterlambatan pengiriman, dan segmentasi pelanggan dalam satu tampilan interaktif.</p>
  <span class="tag">📅 {start:%d %b %Y} - {end:%d %b %Y}</span><span class="tag">📍 {len(states)} dari {len(all_states)} state</span>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("Tidak ada data pada filter yang dipilih. Ubah rentang tanggal atau state di sidebar.")
    st.stop()

order_df = df.drop_duplicates("order_id")
n_orders = order_df["order_id"].nunique()
n_customers = df["customer_unique_id"].nunique()
repeat_pct = (order_df.groupby("customer_unique_id").size() >= 2).mean() * 100
late_pct = order_df["is_late"].mean() * 100

# ------------------------------------------------------------------ KPI
k1, k2, k3, k4, k5 = st.columns(5)
kpi_card(k1, "Total Revenue", format_real_m(df["price"].sum()), f"≈ {format_real(df['price'].sum() / n_orders)} per pesanan", BLUE)
kpi_card(k2, "Jumlah Pesanan", f"{n_orders:,}", f"{len(df):,} item terjual", NAVY)
kpi_card(k3, "Pelanggan Unik", f"{n_customers:,}", f"{repeat_pct:.1f}% berbelanja ulang", GREEN)
kpi_card(k4, "Pesanan Terlambat", f"{late_pct:.1f}%", "melewati tanggal estimasi", RED)
kpi_card(k5, "Rata-rata Review", f"{order_df['review_score'].mean():.2f} / 5", "dari ulasan pelanggan", "#F4A261")
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["🛍️ Produk & Revenue", "🚚 Pengiriman & Kepuasan",
                                  "👥 Segmentasi RFM", "🗺️ Peta Geospasial"])

# ------------------------------------------------------------------ Tab 1: Produk & Revenue
with tab1:
    st.markdown("### Kategori produk dengan revenue terbesar")
    top_n = st.slider("Jumlah kategori yang ditampilkan", 5, 15, 10)  # lebar dibatasi via CSS

    cat = df.groupby("product_category")["price"].sum().sort_values(ascending=False)
    top = cat.head(top_n)
    share = top / cat.sum() * 100
    monthly = df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["price"].sum() / 1e6
    monthly.index = monthly.index.to_timestamp()
    peak = monthly.idxmax()

    top3 = ", ".join(f"<b>{c}</b>" for c in top.index[:3])
    insight(f"Tiga kategori teratas ({top3}) menyumbang <b>{share.head(3).sum():.1f}%</b> dari total revenue. "
            f"Revenue bulanan tertinggi terjadi pada <b>{peak:%B %Y}</b> sebesar <b>R$ {monthly.max():.2f} juta</b>.")

    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(7.2, 5.3))
        plot = top.iloc[::-1] / 1e6
        ax.barh(plot.index, plot.values, color=[BLUE if i >= len(plot) - 3 else GREY for i in range(len(plot))])
        for y, (v, s) in enumerate(zip(plot.values, share.iloc[::-1].values)):
            ax.text(v + plot.max() * 0.012, y, f"R$ {v:.2f} jt ({s:.1f}%)", va="center", fontsize=8.5, color="#344054")
        ax.set_xlim(0, plot.max() * 1.42)
        ax.set_xlabel("Revenue (juta R$)")
        ax.set_ylabel("")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Top {top_n} kategori = {share.sum():.0f}% dari total revenue", loc="left")
        fig.tight_layout()
        show_figure(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(7.2, 5.3))
        ax.fill_between(monthly.index, monthly.values, color=BLUE, alpha=0.12)
        ax.plot(monthly.index, monthly.values, color=BLUE, lw=2.4, marker="o", ms=4)
        ax.scatter([peak], [monthly.max()], color=RED, zorder=5, s=55)
        ax.annotate(f"Puncak: {peak:%b %Y}", xy=(peak, monthly.max()), xytext=(-105, 2),
                    textcoords="offset points", fontsize=9, color=RED, fontweight="bold")
        ax.set_ylim(0, monthly.max() * 1.15)
        ax.set_ylabel("Revenue (juta R$)")
        ax.set_title("Tren revenue bulanan", loc="left")
        fig.autofmt_xdate()
        fig.tight_layout()
        show_figure(fig)

    with st.expander("📋 Lihat tabel revenue per kategori"):
        table = pd.DataFrame({"Revenue (R$)": cat.round(0), "Kontribusi (%)": (cat / cat.sum() * 100).round(2)})
        st.dataframe(table.head(top_n))

# ------------------------------------------------------------------ Tab 2: Pengiriman & Kepuasan
with tab2:
    st.markdown("### Keterlambatan pengiriman vs kepuasan pelanggan")
    rev_df = order_df.dropna(subset=["review_score"])
    on_time = rev_df.loc[rev_df["is_late"] == 0, "review_score"].mean()
    late = rev_df.loc[rev_df["is_late"] == 1, "review_score"].mean()

    min_orders = st.slider("Minimal jumlah pesanan per state (untuk peringkat late rate)", 50, 1000, 300, step=50)
    late_state = (order_df.groupby("customer_state").agg(orders=("order_id", "nunique"), late_rate=("is_late", "mean")))
    late_state["late_rate"] *= 100
    late_state = late_state[late_state["orders"] >= min_orders].sort_values("late_rate", ascending=False).head(10)

    if pd.notna(on_time) and pd.notna(late):
        worst = f" State dengan late rate tertinggi: <b>{late_state.index[0]}</b> ({late_state['late_rate'].iloc[0]:.1f}%)." \
            if not late_state.empty else ""
        insight(f"Pesanan terlambat mendapat rata-rata review <b>{late:.2f}</b>, jauh di bawah pesanan tepat waktu "
                f"(<b>{on_time:.2f}</b>): selisih <b>{on_time - late:.2f} poin</b>.{worst}", "warn")

    m1, m2, m3 = st.columns(3)
    kpi_card(m1, "Review tepat waktu", f"{on_time:.2f}" if pd.notna(on_time) else "-", "rata-rata skor 1-5", BLUE)
    kpi_card(m2, "Review terlambat", f"{late:.2f}" if pd.notna(late) else "-", "rata-rata skor 1-5", RED)
    kpi_card(m3, "Selisih skor", f"{on_time - late:.2f} poin" if pd.notna(on_time) and pd.notna(late) else "-",
             "tepat waktu dikurangi terlambat", NAVY)
    st.write("")

    col_a, col_b = st.columns(2)
    with col_a:
        grp = rev_df.assign(delay_group=pd.cut(rev_df["delay_days"], bins=DELAY_BINS, labels=DELAY_LABELS))
        ds = grp.groupby("delay_group", observed=True)["review_score"].agg(avg="mean", n="count").reset_index()
        labels = ds["delay_group"].astype(str)
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        ax.bar(labels, ds["avg"], color=[RED if "Telat" in l else BLUE for l in labels])
        for x, (v, n) in enumerate(zip(ds["avg"], ds["n"])):
            ax.text(x, v + 0.06, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
            ax.text(x, 0.12, f"n={n:,}", ha="center", color="white", fontsize=7.5)
        ax.set_ylim(0, 5)
        ax.set_ylabel("Rata-rata review score")
        ax.set_xlabel("")
        ax.grid(axis="x", visible=False)
        ax.set_title("Skor anjlok begitu pesanan melewati estimasi", loc="left")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.tight_layout()
        show_figure(fig)
    with col_b:
        if late_state.empty:
            st.info("Tidak ada state yang memenuhi minimal jumlah pesanan.")
        else:
            overall = order_df["is_late"].mean() * 100
            plot = late_state.iloc[::-1]
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            ax.barh(plot.index, plot["late_rate"], color=[RED if i >= len(plot) - 3 else GREY for i in range(len(plot))])
            for y, (v, n) in enumerate(zip(plot["late_rate"], plot["orders"])):
                ax.text(v + plot["late_rate"].max() * 0.01, y, f"{v:.1f}% (n={n:,})", va="center", fontsize=8.5, color="#344054")
            ax.axvline(overall, color="black", ls="--", lw=1)
            ax.set_xlim(0, max(plot["late_rate"].max(), overall) * 1.35)
            ax.set_xlabel(f"Pesanan terlambat (%)   |   garis putus-putus = rata-rata {overall:.1f}%")
            ax.set_ylabel("State")
            ax.grid(axis="y", visible=False)
            ax.set_title("10 state dengan late rate tertinggi", loc="left")
            fig.tight_layout()
            show_figure(fig)

# ------------------------------------------------------------------ Tab 3: RFM
with tab3:
    st.markdown("### Segmentasi pelanggan berbasis RFM")
    st.caption("Recency = hari sejak pembelian terakhir · Frequency = jumlah pesanan · Monetary = total belanja (harga + ongkir). "
               "Dihitung ulang mengikuti filter di sidebar.")
    if n_customers < 50:
        st.info("Data terlalu sedikit untuk segmentasi RFM. Perluas rentang tanggal atau pilih lebih banyak state.")
    else:
        cust = compute_rfm(df)
        seg = (cust.groupby("segment")
               .agg(customers=("monetary", "size"), revenue=("monetary", "sum"),
                    avg_recency=("recency", "mean"), avg_monetary=("monetary", "mean")))
        seg["pct_customers"] = seg["customers"] / seg["customers"].sum() * 100
        seg["pct_revenue"] = seg["revenue"] / seg["revenue"].sum() * 100
        seg = seg.reindex([s for s in SEGMENT_ORDER if s in seg.index])

        best = seg["pct_revenue"].idxmax()
        insight(f"Segmen <b>{best}</b> menyumbang revenue terbesar (<b>{seg.loc[best, 'pct_revenue']:.1f}%</b> dari "
                f"{seg.loc[best, 'pct_customers']:.1f}% pelanggan). Hanya <b>{(cust['frequency'] >= 2).mean() * 100:.1f}%</b> "
                f"pelanggan yang berbelanja lebih dari sekali.")

        r1, r2, r3 = st.columns(3)
        kpi_card(r1, "Pelanggan berbelanja ulang", f"{(cust['frequency'] >= 2).mean() * 100:.1f}%", "memiliki 2+ pesanan", GREEN)
        kpi_card(r2, "Rata-rata recency", f"{cust['recency'].mean():.0f} hari", "sejak pembelian terakhir", NAVY)
        kpi_card(r3, "Rata-rata belanja", format_real(cust["monetary"].mean()), "per pelanggan (harga + ongkir)", BLUE)
        st.write("")

        fig, ax = plt.subplots(figsize=(11, 5))
        y = np.arange(len(seg))
        h = 0.38
        ax.barh(y + h / 2, seg["pct_customers"], h, color=GREY, label="% pelanggan")
        ax.barh(y - h / 2, seg["pct_revenue"], h, color=BLUE, label="% revenue")
        for i, (pc, pr, avg) in enumerate(zip(seg["pct_customers"], seg["pct_revenue"], seg["avg_monetary"])):
            ax.text(pc + 0.4, i + h / 2, f"{pc:.1f}%", va="center", fontsize=8.5, color="#344054")
            ax.text(pr + 0.4, i - h / 2, f"{pr:.1f}%  (rata-rata R$ {avg:,.0f})", va="center", fontsize=8.5, color=BLUE)
        ax.set_yticks(y)
        ax.set_yticklabels(seg.index)
        ax.invert_yaxis()
        ax.set_xlim(0, max(seg["pct_revenue"].max(), seg["pct_customers"].max()) * 1.5)
        ax.set_xlabel("Persentase dari total (%)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", frameon=False)
        ax.set_title("Porsi pelanggan vs porsi revenue tiap segmen", loc="left")
        fig.tight_layout()
        show_figure(fig)

        table = pd.DataFrame({
            "Pelanggan": seg["customers"].map("{:,.0f}".format),
            "% Pelanggan": seg["pct_customers"].map("{:.1f}%".format),
            "% Revenue": seg["pct_revenue"].map("{:.1f}%".format),
            "Rata-rata belanja": seg["avg_monetary"].map("R$ {:,.0f}".format),
            "Rata-rata recency": seg["avg_recency"].map("{:.0f} hari".format),
            "Aksi yang disarankan": [SEGMENT_ACTION[s] for s in seg.index],
        }, index=seg.index)
        table.index.name = "Segmen"
        st.markdown("#### Ringkasan segmen dan aksi yang disarankan")
        st.table(table)

# ------------------------------------------------------------------ Tab 4: Geospasial
with tab4:
    st.markdown("### Sebaran pelanggan dan tingkat keterlambatan")
    geo = order_df.dropna(subset=["lat", "lng"])
    if geo.empty:
        st.info("Tidak ada koordinat pada filter yang dipilih.")
    else:
        min_geo = st.slider("Minimal jumlah pesanan per state (peta kanan)", 0, 1000, 300, step=50, key="geo_min")
        state_geo = (order_df.groupby("customer_state")
                     .agg(lat=("lat", "mean"), lng=("lng", "mean"), orders=("order_id", "nunique"),
                          late_rate=("is_late", "mean")))
        state_geo["late_rate"] *= 100
        state_geo = state_geo[(state_geo["orders"] >= min_geo)].dropna(subset=["lat", "lng"])

        top_states = order_df["customer_state"].value_counts(normalize=True).head(3)
        txt = (f"Tiga state terpadat (<b>{', '.join(top_states.index)}</b>) menampung "
               f"<b>{top_states.sum() * 100:.0f}%</b> pesanan.")
        if not state_geo.empty:
            hi = state_geo["late_rate"].idxmax()
            txt += f" Late rate tertinggi pada state berpesanan ≥ {min_geo}: <b>{hi}</b> ({state_geo.loc[hi, 'late_rate']:.1f}%)."
        insight(txt)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
        ax = axes[0]
        hb = ax.hexbin(geo["lng"], geo["lat"], gridsize=60, bins="log", cmap="Blues", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Jumlah pesanan (skala log)", shrink=0.7)
        ax.set_title("Kepadatan pelanggan", loc="left")

        ax = axes[1]
        ax.scatter(geo["lng"], geo["lat"], s=1, color="#E4E7EC", alpha=0.4)
        if not state_geo.empty:
            sc = ax.scatter(state_geo["lng"], state_geo["lat"], s=state_geo["orders"] / 25 + 25, c=state_geo["late_rate"],
                            cmap="Reds", edgecolor="black", linewidth=0.6, vmin=0, vmax=max(state_geo["late_rate"].max(), 1))
            fig.colorbar(sc, ax=ax, label="Pesanan terlambat (%)", shrink=0.7)
            for s_, r in state_geo.iterrows():
                if r["late_rate"] >= 12 or r["orders"] > 10000:
                    ax.annotate(s_, (r["lng"], r["lat"]), xytext=(5, 5), textcoords="offset points",
                                fontsize=9, fontweight="bold")
        ax.set_title("Late rate per state (ukuran lingkaran = jumlah pesanan)", loc="left")
        for ax in axes:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal")
        fig.tight_layout()
        show_figure(fig)

st.divider()
st.caption("Sumber data: Brazilian E-Commerce Public Dataset by Olist · Analisis oleh Adil Daisuke .")
