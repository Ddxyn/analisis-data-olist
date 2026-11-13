from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Analisis Penjualan Olist", layout="wide")

# Satu warna aksen (sama dengan warna aksen bawaan Streamlit), sisanya abu-abu dan hitam
INK, GREY, ACCENT, LIGHT = "#262730", "#C9CCD3", "#E0474C", "#EEF0F3"

sns.set_theme(style="white")
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
    "axes.labelcolor": "#555B66", "xtick.color": "#555B66", "ytick.color": "#555B66",
    "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.8, "axes.axisbelow": True,
})

DELAY_BINS = [-np.inf, -8, -1, 0, 1, 3, 7, np.inf]
DELAY_LABELS = ["Awal >7 hari", "Awal 1-7 hari", "Hari estimasi", "Telat 1 hari", "Telat 2-3 hari", "Telat 4-7 hari", "Telat >7 hari"]
SEGMENT_ORDER = ["New High-Value", "Dormant High-Value", "Cooling Down", "New Customers", "Lost / Low-Value", "Repeat Buyers"]
SEGMENT_ACTION = {
    "New High-Value": "Dorong pembelian kedua dalam 90 hari pertama",
    "Dormant High-Value": "Kampanye win-back dengan voucher",
    "Cooling Down": "Pengingat dan penawaran terbatas",
    "New Customers": "Rekomendasi produk otomatis (biaya rendah)",
    "Lost / Low-Value": "Pesan massal berbiaya rendah, tanpa insentif mahal",
    "Repeat Buyers": "Pertahankan lewat program loyalitas",
}

# Penyesuaian kecil saja: ruang atas, footer, dan lebar slider
st.markdown("""
<style>
.block-container {padding-top: 3rem; max-width: 1250px;}
footer {visibility: hidden;}
div[data-testid="stSlider"] {max-width: 420px;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "main_data.csv", parse_dates=["order_purchase_timestamp"])


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
    st.header("Filter")
    min_date = main_df["order_purchase_timestamp"].min().date()
    max_date = main_df["order_purchase_timestamp"].max().date()
    date_range = st.date_input("Tanggal pembelian", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    all_states = sorted(main_df["customer_state"].unique())
    all_toggle = st.checkbox("Semua state", value=True)
    states = all_states if all_toggle else st.multiselect("State pelanggan", all_states, default=["SP", "RJ", "MG"])
    st.divider()
    st.caption("Hanya pesanan berstatus delivered, Januari 2017 sampai Agustus 2018. "
               "Nilai dalam Real Brasil (R$). Sumber: Olist Brazilian E-Commerce Public Dataset.")

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
else:
    start = end = date_range[0] if isinstance(date_range, (tuple, list)) else date_range

df = main_df[main_df["order_purchase_timestamp"].dt.date.between(start, end) & main_df["customer_state"].isin(states)]

# ------------------------------------------------------------------ Judul
st.title("Analisis Penjualan E-Commerce Olist")
st.caption(f"Periode {start:%d %b %Y} sampai {end:%d %b %Y}, {len(states)} dari {len(all_states)} state.")

if df.empty:
    st.warning("Tidak ada data pada filter yang dipilih.")
    st.stop()

order_df = df.drop_duplicates("order_id")
n_orders = order_df["order_id"].nunique()
n_customers = df["customer_unique_id"].nunique()
repeat_pct = (order_df.groupby("customer_unique_id").size() >= 2).mean() * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total revenue", f"R$ {df['price'].sum() / 1e6:,.2f} jt")
c2.metric("Pesanan", f"{n_orders:,}")
c3.metric("Pelanggan unik", f"{n_customers:,}", help=f"{repeat_pct:.1f}% di antaranya berbelanja lebih dari sekali.")
c4.metric("Pesanan terlambat", f"{order_df['is_late'].mean() * 100:.1f}%", help="Tiba setelah tanggal estimasi.")
c5.metric("Rata-rata review", f"{order_df['review_score'].mean():.2f} dari 5")

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["Produk", "Pengiriman", "Pelanggan (RFM)", "Wilayah"])

# ------------------------------------------------------------------ Produk
with tab1:
    st.subheader("Kategori produk apa yang menyumbang revenue terbesar?")
    top_n = st.slider("Jumlah kategori", 5, 15, 10)

    cat = df.groupby("product_category")["price"].sum().sort_values(ascending=False)
    top = cat.head(top_n)
    share = top / cat.sum() * 100
    monthly = df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["price"].sum() / 1e6
    monthly.index = monthly.index.to_timestamp()
    peak = monthly.idxmax()

    col_a, col_b = st.columns([1.15, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot = top.iloc[::-1] / 1e6
        ax.barh(plot.index, plot.values, color=[INK if i >= len(plot) - 3 else GREY for i in range(len(plot))], height=0.7)
        for y, (v, s) in enumerate(zip(plot.values, share.iloc[::-1].values)):
            ax.text(v + plot.max() * 0.015, y, f"{v:.2f} jt  ({s:.1f}%)", va="center", fontsize=8.5, color="#555B66")
        ax.set_xlim(0, plot.max() * 1.35)
        ax.set_xlabel("Revenue (juta R$)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Revenue per kategori, {top_n} teratas")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(monthly.index, monthly.values, color=INK, lw=1.8)
        ax.scatter([peak], [monthly.max()], color=ACCENT, zorder=5, s=36)
        ax.annotate(f"{peak:%b %Y}", xy=(peak, monthly.max()), xytext=(-48, 2), textcoords="offset points",
                    fontsize=9, color=ACCENT)
        ax.set_ylim(0, monthly.max() * 1.15)
        ax.set_ylabel("Revenue (juta R$)")
        ax.set_title("Revenue per bulan")
        ax.grid(axis="x", visible=False)
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    top3 = ", ".join(top.index[:3])
    st.markdown(f"**Temuan.** {top_n} kategori teratas menyumbang **{share.sum():.1f}%** dari total revenue. "
                f"Tiga yang terbesar ({top3}) menyumbang {share.head(3).sum():.1f}%. "
                f"Revenue bulanan tertinggi terjadi pada {peak:%B %Y}, sebesar R$ {monthly.max():.2f} juta.")
    with st.expander("Tabel revenue per kategori"):
        st.dataframe(pd.DataFrame({"Revenue (R$)": cat.round(0), "Kontribusi (%)": (cat / cat.sum() * 100).round(2)}).head(top_n))

# ------------------------------------------------------------------ Pengiriman
with tab2:
    st.subheader("Seberapa besar keterlambatan pengiriman memengaruhi review pelanggan?")
    min_orders = st.slider("Minimal pesanan per state", 50, 1000, 300, step=50)

    rev_df = order_df.dropna(subset=["review_score"])
    on_time = rev_df.loc[rev_df["is_late"] == 0, "review_score"].mean()
    late = rev_df.loc[rev_df["is_late"] == 1, "review_score"].mean()
    overall = order_df["is_late"].mean() * 100
    late_state = (order_df.groupby("customer_state").agg(orders=("order_id", "nunique"), late_rate=("is_late", "mean")))
    late_state["late_rate"] *= 100
    late_state = late_state[late_state["orders"] >= min_orders].sort_values("late_rate", ascending=False).head(10)

    col_a, col_b = st.columns(2)
    with col_a:
        grp = rev_df.assign(delay_group=pd.cut(rev_df["delay_days"], bins=DELAY_BINS, labels=DELAY_LABELS))
        ds = grp.groupby("delay_group", observed=True)["review_score"].agg(avg="mean", n="count").reset_index()
        labels = ds["delay_group"].astype(str)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(labels, ds["avg"], color=[ACCENT if "Telat" in l else GREY for l in labels], width=0.7)
        for x, v in enumerate(ds["avg"]):
            ax.text(x, v + 0.07, f"{v:.2f}", ha="center", fontsize=9, color=INK)
        ax.set_ylim(0, 5)
        ax.set_ylabel("Rata-rata review (1-5)")
        ax.grid(axis="x", visible=False)
        ax.set_title("Review menurut selisih tiba dan estimasi")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        if late_state.empty:
            st.info("Tidak ada state yang memenuhi minimal jumlah pesanan.")
        else:
            plot = late_state.iloc[::-1]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(plot.index, plot["late_rate"], color=[ACCENT if i >= len(plot) - 3 else GREY for i in range(len(plot))], height=0.7)
            for y, (v, n) in enumerate(zip(plot["late_rate"], plot["orders"])):
                ax.text(v + plot["late_rate"].max() * 0.015, y, f"{v:.1f}%  (n={n:,})", va="center", fontsize=8.5, color="#555B66")
            ax.axvline(overall, color=INK, ls="--", lw=0.9)
            ax.set_xlim(0, max(plot["late_rate"].max(), overall) * 1.35)
            ax.set_xlabel(f"Pesanan terlambat (%). Garis putus-putus: rata-rata {overall:.1f}%")
            ax.grid(axis="y", visible=False)
            ax.set_title("State dengan keterlambatan tertinggi")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if pd.notna(on_time) and pd.notna(late):
        worst = (f" State dengan keterlambatan tertinggi adalah {late_state.index[0]} ({late_state['late_rate'].iloc[0]:.1f}%)."
                 if not late_state.empty else "")
        by_bin = dict(zip(labels, ds["avg"]))
        drop = (f" Skor turun dari {by_bin['Hari estimasi']:.2f} pada hari estimasi menjadi {by_bin['Telat 1 hari']:.2f} "
                f"pada keterlambatan 1 hari." if {"Hari estimasi", "Telat 1 hari"} <= set(by_bin) else "")
        st.markdown(f"**Temuan.** Pesanan terlambat rata-rata mendapat review **{late:.2f}**, sedangkan pesanan tepat waktu "
                    f"**{on_time:.2f}** (selisih {on_time - late:.2f} poin).{drop}{worst}")

# ------------------------------------------------------------------ RFM
with tab3:
    st.subheader("Kelompok pelanggan mana yang paling layak diprioritaskan untuk retensi?")
    st.caption("Recency: hari sejak pembelian terakhir. Frequency: jumlah pesanan. Monetary: total belanja (harga + ongkir). "
               "Dihitung ulang mengikuti filter.")
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

        fig, ax = plt.subplots(figsize=(10, 4.6))
        y = np.arange(len(seg))
        h = 0.38
        ax.barh(y - h / 2, seg["pct_customers"], h, color=GREY, label="% pelanggan")
        ax.barh(y + h / 2, seg["pct_revenue"], h, color=INK, label="% revenue")
        for i, (pc, pr) in enumerate(zip(seg["pct_customers"], seg["pct_revenue"])):
            ax.text(pc + 0.4, i - h / 2, f"{pc:.1f}%", va="center", fontsize=8.5, color="#555B66")
            ax.text(pr + 0.4, i + h / 2, f"{pr:.1f}%", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(seg.index)
        ax.invert_yaxis()
        ax.set_xlim(0, max(seg["pct_revenue"].max(), seg["pct_customers"].max()) * 1.2)
        ax.set_xlabel("Persentase dari total (%)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", frameon=False)
        ax.set_title("Porsi pelanggan dan porsi revenue per segmen")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        best = seg["pct_revenue"].idxmax()
        st.markdown(f"**Temuan.** Hanya **{(cust['frequency'] >= 2).mean() * 100:.1f}%** pelanggan yang berbelanja lebih dari sekali. "
                    f"Segmen {best} menyumbang revenue terbesar ({seg.loc[best, 'pct_revenue']:.1f}% dari "
                    f"{seg.loc[best, 'pct_customers']:.1f}% pelanggan).")

        table = pd.DataFrame({
            "Pelanggan": seg["customers"].map("{:,.0f}".format),
            "% Pelanggan": seg["pct_customers"].map("{:.1f}".format),
            "% Revenue": seg["pct_revenue"].map("{:.1f}".format),
            "Rata-rata belanja": seg["avg_monetary"].map("R$ {:,.0f}".format),
            "Rata-rata recency": seg["avg_recency"].map("{:.0f} hari".format),
            "Aksi yang disarankan": [SEGMENT_ACTION[s] for s in seg.index],
        }, index=seg.index)
        table.index.name = "Segmen"
        st.table(table)

# ------------------------------------------------------------------ Wilayah
with tab4:
    st.subheader("Di mana pelanggan terkonsentrasi dan di mana pengiriman sering terlambat?")
    geo = order_df.dropna(subset=["lat", "lng"])
    if geo.empty:
        st.info("Tidak ada koordinat pada filter yang dipilih.")
    else:
        min_geo = st.slider("Minimal pesanan per state (peta kanan)", 0, 1000, 300, step=50, key="geo_min")
        state_geo = (order_df.groupby("customer_state")
                     .agg(lat=("lat", "mean"), lng=("lng", "mean"), orders=("order_id", "nunique"),
                          late_rate=("is_late", "mean")))
        state_geo["late_rate"] *= 100
        state_geo = state_geo[state_geo["orders"] >= min_geo].dropna(subset=["lat", "lng"])

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        ax = axes[0]
        hb = ax.hexbin(geo["lng"], geo["lat"], gridsize=60, bins="log", cmap="Greys", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Jumlah pesanan (skala log)", shrink=0.7)
        ax.set_title("Kepadatan pelanggan")

        ax = axes[1]
        ax.scatter(geo["lng"], geo["lat"], s=1, color=LIGHT, alpha=0.6)
        if not state_geo.empty:
            sc = ax.scatter(state_geo["lng"], state_geo["lat"], s=state_geo["orders"] / 25 + 25, c=state_geo["late_rate"],
                            cmap="Reds", edgecolor=INK, linewidth=0.5, vmin=0, vmax=max(state_geo["late_rate"].max(), 1))
            fig.colorbar(sc, ax=ax, label="Pesanan terlambat (%)", shrink=0.7)
            for s_, r in state_geo.iterrows():
                if r["late_rate"] >= 12 or r["orders"] > 10000:
                    ax.annotate(s_, (r["lng"], r["lat"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.set_title("Keterlambatan per state (ukuran lingkaran = jumlah pesanan)")
        for ax in axes:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal")
            ax.grid(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        top_states = order_df["customer_state"].value_counts(normalize=True).head(3)
        txt = f"**Temuan.** Tiga state terpadat ({', '.join(top_states.index)}) menampung **{top_states.sum() * 100:.0f}%** pesanan."
        if not state_geo.empty:
            hi = state_geo["late_rate"].idxmax()
            txt += f" Keterlambatan tertinggi pada state dengan minimal {min_geo} pesanan ada di {hi} ({state_geo.loc[hi, 'late_rate']:.1f}%)."
        st.markdown(txt)

st.divider()
st.caption("Sumber data: Brazilian E-Commerce Public Dataset by Olist. Analisis oleh Adil Daisuke.")
