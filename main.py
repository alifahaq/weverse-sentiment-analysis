import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from google_play_scraper import Sort, reviews
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# TAHAP 1: DATA ACQUISITION (CRAWLING)
# ==========================================
print("Memulai pengambilan data Weverse...")
result, _ = reviews(
    'co.benx.weverse',
    lang='id', 
    country='id', 
    sort=Sort.NEWEST, 
    count=1000 
)
df = pd.DataFrame(result)
df.to_csv('weverse_reviews_raw.csv', index=False)
print(f"Berhasil mengambil {len(df)} data.")

# TAHAP 2: DATA PREPROCESSING
# ==========================================
print("Memulai pembersihan data (Preprocessing)...")
stop_factory = StopWordRemoverFactory()
stopword_remover = stop_factory.create_stop_word_remover()
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def cleaning_proses(text):
    text = str(text).lower() 
    text = re.sub(r'[^a-z\s]', '', text) 
    text = stopword_remover.remove(text) 
    text = stemmer.stem(text) 
    return text

df['content_cleaned'] = df['content'].apply(cleaning_proses)
df.to_csv('weverse_reviews_cleaned.csv', index=False)
print("Pembersihan selesai!")

# TAHAP 3: EDA (VISUALISASI)
# ==========================================
# Distribusi Rating
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(x='score', data=df, palette='viridis')
plt.title('Distribusi Rating Weverse')
plt.savefig('distribusi_rating.png') 
plt.show()

# WordCloud
all_text = ' '.join(df['content_cleaned'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Kata Kunci Ulasan Weverse')
plt.savefig('wordcloud_weverse.png')
plt.show()

# TAHAP 4: MODELING & EVALUASI
# ==========================================
df_model = df[df['score'] != 3].copy()
df_model['label'] = df_model['score'].apply(lambda x: 1 if x > 3 else 0)

X_train, X_test, y_train, y_test = train_test_split(
    df_model['content_cleaned'], 
    df_model['label'], 
    test_size=0.2, 
    random_state=42
)

tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print(f"\nAkurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()