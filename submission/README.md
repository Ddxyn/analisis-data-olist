# Proyek Analisis Data: E-Commerce Public Dataset (Olist)

Proyek akhir kelas **Belajar Fundamental Analisis Data** (Dicoding). Analisis memakai data transaksi marketplace Olist (Brasil) pada periode **Januari 2017 - Agustus 2018** untuk menjawab tiga pertanyaan bisnis:

1. 10 kategori produk apa yang menghasilkan revenue terbesar, berapa persen kontribusi dan pertumbuhan masing-masing (Jan-Agu 2018 vs Jan-Agu 2017), sebagai dasar penentuan kategori prioritas untuk alokasi stok dan anggaran promosi?
2. Berapa persen pesanan yang tiba setelah tanggal estimasi, berapa selisih rata-rata review antara pesanan tepat waktu dan terlambat, dan 5 state mana (minimal 300 pesanan) dengan keterlambatan tertinggi, sebagai dasar penentuan wilayah prioritas evaluasi logistik?
3. Berapa persen pelanggan dan persen revenue dari setiap segmen RFM, sebagai dasar pemilihan segmen target dan insentif kampanye retensi?

Teknik analisis lanjutan: **RFM Analysis**, **Geospatial Analysis**, dan **Binning / Manual Grouping** (tanpa algoritma machine learning).

## Struktur Direktori

```
submission
├───dashboard
│   ├───main_data.csv        # data hasil cleaning untuk dashboard
│   └───dashboard.py         # aplikasi Streamlit
├───data                     # dataset asli (Olist)
│   ├───customers_dataset.csv
│   ├───geolocation_dataset.csv
│   ├───order_items_dataset.csv
│   ├───order_reviews_dataset.csv
│   ├───orders_dataset.csv
│   ├───product_category_name_translation.csv
│   └───products_dataset.csv
├───notebook.ipynb           # proses analisis lengkap
├───README.md
├───requirements.txt
└───url.txt                  # tautan dashboard di Streamlit Cloud
```

## Setup Environment

Disarankan memakai Python 3.10 atau lebih baru.

```
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Menjalankan Notebook

Notebook memakai path relatif, jadi jalankan dari folder `submission` (folder yang berisi `notebook.ipynb`). Notebook membutuhkan Jupyter (`pip install notebook`) atau dapat dibuka di VS Code / Google Colab (unggah juga folder `data`). Menjalankan seluruh sel akan membuat ulang `dashboard/main_data.csv`.

## Menjalankan Dashboard Secara Lokal

```
cd submission
streamlit run dashboard/dashboard.py
```

Dashboard terbuka di `http://localhost:8501`. Fitur: filter rentang tanggal dan state, empat tab (Produk & Revenue, Pengiriman & Kepuasan, Segmentasi RFM, Peta Geospasial).

## Deploy ke Streamlit Community Cloud

1. Unggah folder proyek ke repositori GitHub (`requirements.txt` berada di root repositori).
2. Masuk ke [share.streamlit.io](https://share.streamlit.io) dan klik **Create app**.
3. Pilih repositori dan branch, lalu isi **Main file path** dengan `dashboard/dashboard.py`.
4. Klik **Deploy**, lalu salin tautan aplikasi ke `url.txt`.

## Sumber Data

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
