import pandas as pd
import numpy as np
import plotly.express as px
import nltk
import gensim.models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from tqdm import tqdm
from umap import UMAP
from mittens import GloVe


pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


#Chapter 5&6
df = pd.read_csv('/home/azmi/Documents/Folder_Root/Semester_3/Any_files/Pemrosesan_Teks/Any/chapter_5-6/gt_wiki_TP.csv')
df = df.dropna().reset_index().drop(columns=['index'])
df_pre = df['Item_Name'].str.lower().str.replace(r'https?://\S+|www\.\S+|\S+@\S+', '', regex=True).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
df_pre.info()

# Mengubah teks menjadi matriks TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_pre)
df_FE = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df_FE)


nltk.download('punkt')
nltk.download('punkt_tab')
sentences = [word_tokenize(Item_Name.lower()) for Item_Name in tqdm(df_pre)]
print(sentences)


# WORD2VEC MODEL
## Train Model
# model = gensim.models.Word2Vec(sentences, min_count=1, vector_size=100, window=5, sg=0)

## 2. Save Model
# model.save('sentimen_Item.w2v')

## 3. Load Model
# model = gensim.models.Word2Vec.load('sentimen_Item.w2v')

## 4. Model Information
# w2v = model.wv
# mof = w2v.index_to_key[:10]
# print(mof)
# # 5. Vector Size
# wfec = w2v.vectors.shape, w2v.vector_size
# print(wfec)
# # 6. Similarity
# wsim = w2v.similar_by_key('blocks')
# print(wsim)
# # 5. FE result
# def get_document_vector(tokens, model):
#     valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
#     # if not valid_vectors:
#     #   return np.zeros(model.vector_size)
#     return np.mean(valid_vectors, axis=0)
# X_features = np.array([get_document_vector(doc, model) for doc in sentences])
# df_Features = pd.DataFrame(X_features)
# print(df_Features)
# # 6. Visualization
# X = UMAP().fit_transform(w2v.vectors)

# df2 = pd.DataFrame(X, columns=['umap1', 'umap2'])
# df2['word'] = w2v.index_to_key

# fig = px.scatter(df2, x='umap1', y='umap2', text='word')
# fig.update_traces(textposition='top center')
# fig.update_layout(height = 800,
#                   title_text = 'Word2Vec GT Item Visualization')
# fig.show()


# # FASTEXT MODEL
# model_fasttext = gensim.models.FastText(sentences, min_count=1,
#                                         vector_size=100, window=5,
#                                         min_n=3, max_n=6,
#                                         sg=1, epochs=10)
# # 2. save model FastText
# model_fasttext.save('sentimen_Item.fasttext')
# # 3. load model FastText
# model_fasttext = gensim.models.FastText.load('sentimen_Item.fasttext')
# # 4. model information
# fastext = model_fasttext.wv
# mot = fastext.index_to_key[:10]
# print(mot)
# # 5. vector size FasText
# fvec = fastext.vectors.shape
# print(fvec)
# # 6. Similarity FastText
# fsim = fastext.similar_by_key('blocks')
# print(fsim)
# # 7. FE result FastText
# def get_document_vector(tokens, model):
#     valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
#     # if not valid_vectors:
#     #   return np.zeros(model.vector_size)
#     return np.mean(valid_vectors, axis=0)
# X_features = np.array([get_document_vector(doc, model_fasttext) for doc in sentences])
# df_ft = pd.DataFrame(X_features)
# print(df_ft)
# # 8. Visualization FastText
# X = UMAP().fit_transform(fastext.vectors)

# df2 = pd.DataFrame(X, columns=['umap1', 'umap2'])
# df2['word'] = fastext.index_to_key

# fig = px.scatter(df2, x='umap1', y='umap2', text='word')
# fig.update_traces(textposition='top center')
# fig.update_layout(height = 800,
#                   title_text = 'fastext GT Item Visualization')
# fig.show()


# GLOVE MODEL DENGAN MITTENS
# Buat vokabular dan co-occurrence matrix
vocab = list(set([word for doc in sentences for word in doc]))
word2idx = {word: idx for idx, word in enumerate(vocab)}
cooccur_matrix = np.zeros((len(vocab), len(vocab)))
window_size = 5
for doc in sentences:
    for i, word in enumerate(doc):
        for j in range(max(0, i - window_size), min(len(doc), i + window_size + 1)):
            if i != j:
                cooccur_matrix[word2idx[word]][word2idx[doc[j]]] += 1

# trainning glove model
model_glove = GloVe(n=100, max_iter=100)
word_vectors = model_glove.fit(cooccur_matrix)
glove_dict = {word: word_vectors[idx] for word, idx in word2idx.items()} #Saving vektor per-kata

print("Vocab (10):", vocab[:10])
print("Vector Size:", word_vectors.shape)
 
# Feature Extraction (Document Vector)
def get_document_vector_glove(tokens):
    valid_vectors = [glove_dict[word] for word in tokens if word in glove_dict]
    if not valid_vectors:
        return np.zeros(100)
    return np.mean(valid_vectors, axis=0)
X_features_glove = np.array([get_document_vector_glove(doc) for doc in sentences])
df_glove = pd.DataFrame(X_features_glove)
print(df_glove)

# Visualisasi
X = UMAP().fit_transform(word_vectors)
df2 = pd.DataFrame(X, columns=['umap1', 'umap2'])
df2['word'] = vocab
fig = px.scatter(df2, x='umap1', y='umap2', text='word')
fig.update_traces(textposition='top center')
fig.update_layout(
    height=800,
    title_text='GloVe GT Item Visualization'
)
fig.show()