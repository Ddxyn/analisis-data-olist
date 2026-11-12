from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ------------------------------------------------------------------ Konfigurasi
st.set_page_config(page_title="Dashboard E-Commerce Olist", page_icon="📊", layout="wide")

sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12})
BLUE, GREY, RED = "#2E5EAA", "#C9CED6", "#D1495B"

DELAY_BINS = [-np.inf, -8, -1, 0, 3, 7, np.inf]
DELAY_LABELS = ["Lebih awal >7 hari", "Lebih awal 1-7 hari", "Hari estimasi",
                "Telat 1-3 hari", "Telat 4-7 hari", "Telat >7 hari"]
SEGMENT_ORDER = ["New High-Value", "Dormant High-Value", "Cooling Down",
                 "New Customers", "Lost / Low-Value", "Repeat Buyers"]


# ------------------------------------------------------------------ Data
@st.cache_data
def load_data() -> pd.DataFrame:
    path = Path(__file__).parent / "main_data.csv"
    return pd.read_csv(path, parse_dates=["order_purchase_timestamp"])


def format_real(value: float) -> str:
    """Format angka mata uang Real Brasil (R$) dengan pemisah ribuan."""
    return f"R$ {value:,.0f}"


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

# ------------------------------------------------------------------ Sidebar (filter)
st.sidebar.header("Filter Data")
min_date = main_df["order_purchase_timestamp"].min().date()
max_date = main_df["order_purchase_timestamp"].max().date()
date_range = st.sidebar.date_input("Rentang tanggal pembelian", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
all_states = sorted(main_df["customer_state"].unique())
states = st.sidebar.multiselect("State pelanggan", all_states, default=all_states)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
else:
    start = end = date_range[0] if isinstance(date_range, (tuple, list)) else date_range

mask = (main_df["order_purchase_timestamp"].dt.date.between(start, end)) & (main_df["customer_state"].isin(states))
df = main_df[mask]
st.sidebar.caption("Data: pesanan berstatus *delivered*, Jan 2017 - Agu 2018.")

# ------------------------------------------------------------------ Header
st.title("📊 Dashboard Analisis E-Commerce (Olist)")
st.caption("Menjawab tiga pertanyaan: kategori produk berkontribusi revenue terbesar, dampak keterlambatan pengiriman "
           "terhadap kepuasan pelanggan, dan segmentasi pelanggan berbasis RFM.")

if df.empty:
    st.warning("Tidak ada data pada filter yang dipilih. Ubah rentang tanggal atau state.")
    st.stop()

order_df = df.drop_duplicates("order_id")

# ------------------------------------------------------------------ KPI
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue Produk", format_real(df["price"].sum()))
c2.metric("Jumlah Pesanan", f"{order_df['order_id'].nunique():,}")
c3.metric("Pelanggan Unik", f"{df['customer_unique_id'].nunique():,}")
c4.metric("Pesanan Terlambat", f"{order_df['is_late'].mean() * 100:.1f}%")
c5.metric("Rata-rata Review", f"{order_df['review_score'].mean():.2f} / 5")

tab1, tab2, tab3, tab4 = st.tabs(["🛍️ Produk & Revenue", "🚚 Pengiriman & Kepuasan",
                                  "👥 Segmentasi RFM", "🗺️ Peta Geospasial"])

# ------------------------------------------------------------------ Tab 1: Produk & Revenue
with tab1:
    st.subheader("Kategori produk dengan revenue terbesar")
    top_n = st.slider("Jumlah kategori teratas", 5, 15, 10)

    cat = (df.groupby("product_category")["price"].sum().sort_values(ascending=False))
    top = cat.head(top_n)
    share = top / cat.sum() * 100

    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(7, 5.2))
        plot = top.iloc[::-1] / 1e6
        ax.barh(plot.index, plot.values, color=[BLUE if i >= len(plot) - 3 else GREY for i in range(len(plot))])
        for y, (v, s) in enumerate(zip(plot.values, share.iloc[::-1].values)):
            ax.text(v + plot.max() * 0.01, y, f"R$ {v:.2f} jt ({s:.1f}%)", va="center", fontsize=8.5)
        ax.set_xlim(0, plot.max() * 1.4)
        ax.set_xlabel("Revenue (juta R$)")
        ax.set_ylabel("")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Top {top_n} kategori = {share.sum():.0f}% dari total revenue", loc="left")
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        monthly = (df.groupby(df["order_purchase_timestamp"].dt.to_period("M"))["price"].sum() / 1e6)
        monthly.index = monthly.index.to_timestamp()
        fig, ax = plt.subplots(figsize=(7, 5.2))
        ax.plot(monthly.index, monthly.values, color=BLUE, lw=2.2, marker="o", ms=4)
        peak = monthly.idxmax()
        ax.annotate(f"Puncak: {peak:%b %Y}", xy=(peak, monthly.max()), xytext=(10, -25),
                    textcoords="offset points", arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9)
        ax.set_ylim(0, None)
        ax.set_ylabel("Revenue (juta R$)")
        ax.set_title("Revenue bulanan", loc="left")
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Lihat tabel revenue per kategori"):
        table = pd.DataFrame({"Revenue (R$)": cat.round(0), "Kontribusi (%)": (cat / cat.sum() * 100).round(2)})
        st.dataframe(table.head(top_n))

# ------------------------------------------------------------------ Tab 2: Pengiriman & Kepuasan
with tab2:
    st.subheader("Keterlambatan pengiriman vs review score")
    rev_df = order_df.dropna(subset=["review_score"])
    on_time = rev_df.loc[rev_df["is_late"] == 0, "review_score"].mean()
    late = rev_df.loc[rev_df["is_late"] == 1, "review_score"].mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Review pesanan tepat waktu", f"{on_time:.2f}" if pd.notna(on_time) else "-")
    m2.metric("Review pesanan terlambat", f"{late:.2f}" if pd.notna(late) else "-")
    m3.metric("Selisih", f"{on_time - late:.2f} poin" if pd.notna(on_time) and pd.notna(late) else "-")

    col_a, col_b = st.columns(2)
    with col_a:
        grp = rev_df.assign(delay_group=pd.cut(rev_df["delay_days"], bins=DELAY_BINS, labels=DELAY_LABELS))
        ds = grp.groupby("delay_group", observed=True)["review_score"].agg(avg="mean", n="count").reset_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(ds["delay_group"].astype(str), ds["avg"],
               color=[RED if "Telat" in l else BLUE for l in ds["delay_group"].astype(str)])
        for x, (v, n) in enumerate(zip(ds["avg"], ds["n"])):
            ax.text(x, v + 0.06, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
            ax.text(x, 0.12, f"n={n:,}", ha="center", color="white", fontsize=7.5)
        ax.set_ylim(0, 5)
        ax.set_ylabel("Rata-rata review score")
        ax.set_xlabel("")
        ax.set_title("Skor anjlok begitu pesanan melewati estimasi", loc="left")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        min_orders = st.slider("Minimal jumlah pesanan per state", 50, 1000, 300, step=50)
        late_state = (order_df.groupby("customer_state")
                      .agg(orders=("order_id", "nunique"), late_rate=("is_late", "mean")))
        late_state["late_rate"] *= 100
        late_state = late_state[late_state["orders"] >= min_orders].sort_values("late_rate", ascending=False).head(10)
        overall = order_df["is_late"].mean() * 100
        if late_state.empty:
            st.info("Tidak ada state yang memenuhi minimal jumlah pesanan.")
        else:
            plot = late_state.iloc[::-1]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(plot.index, plot["late_rate"], color=[RED if i >= len(plot) - 3 else GREY for i in range(len(plot))])
            for y, (v, n) in enumerate(zip(plot["late_rate"], plot["orders"])):
                ax.text(v + plot["late_rate"].max() * 0.01, y, f"{v:.1f}% (n={n:,})", va="center", fontsize=8.5)
            ax.axvline(overall, color="black", ls="--", lw=1)
            ax.set_xlim(0, max(plot["late_rate"].max(), overall) * 1.35)
            ax.set_xlabel(f"Pesanan terlambat (%)  |  garis putus-putus = rata-rata {overall:.1f}%")
            ax.set_ylabel("State")
            ax.grid(axis="y", visible=False)
            ax.set_title("10 state dengan late rate tertinggi", loc="left")
            st.pyplot(fig)
            plt.close(fig)

# ------------------------------------------------------------------ Tab 3: RFM
with tab3:
    st.subheader("Segmentasi pelanggan berbasis RFM")
    st.caption("Recency = hari sejak pembelian terakhir, Frequency = jumlah pesanan, Monetary = total belanja (harga + ongkir). "
               "Dihitung ulang sesuai filter pada sidebar.")
    if df["customer_unique_id"].nunique() < 50:
        st.info("Data terlalu sedikit untuk segmentasi RFM. Perluas rentang tanggal atau pilih lebih banyak state.")
    else:
        cust = compute_rfm(df)
        seg = (cust.groupby("segment")
               .agg(customers=("monetary", "size"), revenue=("monetary", "sum"),
                    avg_recency=("recency", "mean"), avg_monetary=("monetary", "mean")))
        seg["pct_customers"] = seg["customers"] / seg["customers"].sum() * 100
        seg["pct_revenue"] = seg["revenue"] / seg["revenue"].sum() * 100
        seg = seg.reindex([s for s in SEGMENT_ORDER if s in seg.index])

        k1, k2, k3 = st.columns(3)
        k1.metric("Pelanggan berbelanja ulang", f"{(cust['frequency'] >= 2).mean() * 100:.1f}%")
        k2.metric("Rata-rata recency", f"{cust['recency'].mean():.0f} hari")
        k3.metric("Rata-rata belanja per pelanggan", format_real(cust["monetary"].mean()))

        fig, ax = plt.subplots(figsize=(10, 5))
        y = np.arange(len(seg))
        h = 0.38
        ax.barh(y + h / 2, seg["pct_customers"], h, color=GREY, label="% pelanggan")
        ax.barh(y - h / 2, seg["pct_revenue"], h, color=BLUE, label="% revenue")
        for i, (pc, pr, avg) in enumerate(zip(seg["pct_customers"], seg["pct_revenue"], seg["avg_monetary"])):
            ax.text(pc + 0.4, i + h / 2, f"{pc:.1f}%", va="center", fontsize=8.5)
            ax.text(pr + 0.4, i - h / 2, f"{pr:.1f}% (rata-rata R$ {avg:,.0f})", va="center", fontsize=8.5, color=BLUE)
        ax.set_yticks(y)
        ax.set_yticklabels(seg.index)
        ax.invert_yaxis()
        ax.set_xlim(0, max(seg["pct_revenue"].max(), seg["pct_customers"].max()) * 1.5)
        ax.set_xlabel("Persentase dari total (%)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", frameon=False)
        ax.set_title("Perbandingan porsi pelanggan dan porsi revenue tiap segmen", loc="left")
        st.pyplot(fig)
        plt.close(fig)

        table = seg.rename(columns={"customers": "Pelanggan", "revenue": "Revenue (R$)", "avg_recency": "Rata-rata recency (hari)",
                                    "avg_monetary": "Rata-rata belanja (R$)", "pct_customers": "% Pelanggan",
                                    "pct_revenue": "% Revenue"}).round(1)
        st.dataframe(table)

# ------------------------------------------------------------------ Tab 4: Geospasial
with tab4:
    st.subheader("Sebaran pelanggan dan tingkat keterlambatan")
    geo = order_df.dropna(subset=["lat", "lng"])
    if geo.empty:
        st.info("Tidak ada koordinat pada filter yang dipilih.")
    else:
        state_geo = (geo.groupby("customer_state")
                     .agg(lat=("lat", "mean"), lng=("lng", "mean"), orders=("order_id", "nunique"),
                          late_rate=("is_late", "mean")))
        state_geo["late_rate"] *= 100
        min_geo = st.slider("Minimal jumlah pesanan per state (peta kanan)", 0, 1000, 300, step=50, key="geo_min")
        state_geo = state_geo[state_geo["orders"] >= min_geo]

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
        ax.set_title("Late rate per state (ukuran = jumlah pesanan)", loc="left")
        for ax in axes:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.caption("Sumber data: Brazilian E-Commerce Public Dataset by Olist. Analisis oleh Adil daisuke.")
