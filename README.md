# Sistem Peringatan Dini Hujan untuk 1 Jam ke Depan di Kota Pekanbaru

Prototipe sistem pendukung keputusan berbasis pembelajaran mesin untuk memprakirakan peluang kejadian hujan satu jam ke depan di wilayah Kota Pekanbaru menggunakan data meteorologi per jam tahun 2026.

## Ringkasan Metodologi dan Hasil

Pemodelan membandingkan enam konfigurasi klasifikasi berbasis Support Vector Machine dan Artificial Neural Network dengan tujuh fitur rekayasa dinamika atmosfer. Model terbaik diperoleh melalui arsitektur Artificial Neural Network yang dikombinasikan dengan penyeimbangan data latih SMOTE Tomek dengan capaian PR AUC sebesar 0,8596, F1 Score sebesar 0,7897, dan Balanced Accuracy sebesar 82,69 persen pada data uji.

## Struktur Berkas Repositori

* `app.py` sebagai kode utama antarmuka aplikasi web interaktif berbasis Streamlit
* `artefak_hujan_pekanbaru.joblib` sebagai berkas model terlatih dan parameter penskalaan fitur
* `requirements.txt` sebagai daftar pustaka Python untuk menjalankan aplikasi
* File notebook Google Colab (`.ipynb`) sebagai dokumentasi prapemrosesan data dan pelatihan model
* File presentasi proyek kelompok

## Oleh:
* Hanna Toula Sinaga

Program Studi Statistika
