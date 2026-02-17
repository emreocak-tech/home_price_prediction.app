🏠 Ev Fiyat Tahmin Sistemi (PyTorch + Streamlit)
Bu proje, ev fiyatlarını tahmin etmek için derin öğrenme modeli kullanan ve Streamlit ile kullanıcı arayüzü sunan bir uygulamadır.

🚀 Özellikler
🤖 Derin Öğrenme Modeli (PyTorch)
4 Giriş Özelliği: Toplam alan, oda sayısı, toplam kat, yaşam alanı

Sinir Ağı Mimarisi: Linear(4,50) → ReLU → Linear(50,1)

Optimizasyon: Adam optimizer, MSELoss

Normalizasyon: Min-max normalizasyonu

Eğitim/Validasyon/Test: %60/%20/%20 oranında ayrıştırma

🌐 Kullanıcı Arayüzü (Streamlit)
Slider Kontrolleri: Toplam alan, oda sayısı, kat sayısı, yaşam alanı

Eğitim Uyarısı: Sitenin eğitim amaçlı olduğuna dair onay kutusu

Tahmin Butonu: Tek tıkla fiyat tahmini

Anlık Çıktı: Tahmin edilen fiyat değeri

📁 Gereksinimler
bash
pip install torch pandas numpy python-dotenv streamlit matplotlib
🔧 Kurulum
Projeyi klonlayın

.env dosyası oluşturun:

env
DATA_PATH=veri_setinizin_yolu.csv
CSV dosyanızı hazırlayın (aşağıdaki sütunları içermeli):

total_area

rooms

floors_total

living_area

last_price

Modeli eğitin:

bash
python model.py
Streamlit uygulamasını başlatın:

bash
streamlit run home_price_prediction_for_ui.py
📂 Dosya Yapısı
model.py - PyTorch model eğitim dosyası

home_price_prediction_for_ui.py - Streamlit arayüz dosyası

ev_fiyat_tahmin.pth - Eğitilmiş model checkpoint dosyası

.env - Veri yolu konfigürasyonu

🎯 Kullanım
Model Eğitimi
CSV dosyanızı hazırlayın (23.698 satır veri)

model.py çalıştırın:

Veri normalizasyonu

Train/validation/test ayrımı

1000 epoch eğitim

Model kaydetme

Tahmin Arayüzü
Onay kutusunu işaretleyin (eğitim amaçlı site olduğunu kabul edin)

Slider'lardan ev özelliklerini seçin:

Toplam alan (30-160 m²)

Oda sayısı (1-5)

Toplam kat (1-15)

Yaşam alanı (20-70 m²)

"Make a Prediction" butonuna tıklayın

Tahmin edilen fiyatı görüntüleyin

🧠 Model Mimarisi
text
Input (4) → Linear(4,50) → ReLU → Linear(50,1) → Output (1)
Katman Detayları:

Giriş katmanı: 4 nöron (total_area, rooms, floors_total, living_area)

Gizli katman: 50 nöron + ReLU aktivasyonu

Çıkış katmanı: 1 nöron (fiyat tahmini)

📊 Veri Seti
Özellikler:

total_area: Toplam alan (m²)

rooms: Oda sayısı

floors_total: Toplam kat sayısı

living_area: Yaşam alanı (m²)

last_price: Son fiyat (hedef değişken)

Veri Boyutu: 23.698 satır

🔄 Veri Ön İşleme
Eksik Veri Temizleme: fillna(0)

Train/Validation/Test Split: %60 (14.219) / %20 / %20

Min-Max Normalizasyonu: (x - min) / (max - min)

Tensor Dönüşümü: PyTorch tensor formatı

DataLoader: Batch size=10.000, shuffle=True

💾 Model Checkpoint
ev_fiyat_tahmin.pth dosyası içerir:

model_state: Model ağırlıkları

norm_maks: Normalizasyon maksimum değerleri

norm_min: Normalizasyon minimum değerleri

price_maks: Fiyat maksimum değeri

price_min: Fiyat minimum değeri

📈 Eğitim Detayları
Loss Fonksiyonu: MSELoss (Ortalama Kare Hata)

Optimizer: Adam (lr=0.1)

Epoch: 1000

Batch Size: 10000

Loss Takibi: Her 100 epoch'ta bir loss değeri yazdırma

🖥️ Streamlit Arayüzü
Bileşenler:

st.warning(): Uyarı mesajı

st.checkbox(): Onay kutusu

st.slider(): Sayısal giriş

st.select_slider(): Oda sayısı seçimi

st.button(): Tahmin butonu

st.info(): Bilgi mesajı

st.success(): Başarı mesajı

⚙️ Teknik Özellikler
Framework: PyTorch (Derin Öğrenme)

UI: Streamlit (Web Arayüzü)

Veri İşleme: Pandas, NumPy

Normalizasyon: Min-Max Scaling

Model Serileştirme: PyTorch checkpoint

🎓 Eğitim Notu
Bu proje eğitim amaçlıdır. Gerçek alım-satım kararları için profesyonel danışmanlık alınmalıdır.

🔍 Örnek Kullanım
text
Toplam Alan: 85 m²
Oda Sayısı: 4
Toplam Kat: 7
Yaşam Alanı: 35 m²
Tahmin: 2.450.000 TL
