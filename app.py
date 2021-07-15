import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from kmodes.kmodes import KModes

column = ['NO','KABUPATEN_KOTA','TAHAPAN','JABATAN_PELAPOR','JABATAN_TERLAPOR','UMUR_PELAPOR','UMUR_TERLAPOR','JENIS_KELAMIN_TERLAPOR','HASIL_KAJIAN','JENIS_PELANGGARAN']


header = dataset = features = training = st.beta_container()

def makeCol(code,column):
    res = {}
    for i,x in enumerate(column.str.upper().unique()):
        res[x] =code+str(i+1)
    return column.str.upper().map(res)  

@st.cache
def getdata(csv,separator=';') -> pd.DataFrame:
    return pd.read_csv(csv,sep=separator)

with header:
    st.title('Welcome !!')
    st.text("""
    odio euismod lacinia at quis risus sed vulputate odio ut enim 
    blandit volutpat maecenas volutpat blandit aliquam etiam erat velit
    """)
    
with dataset:
    st.header('Datasets')
    df = getdata('DATA_TESIS.csv',separator=';')
    df = df[column].copy()
    shape = df.shape
    st.text('Data Awal '+str(shape))
    st.write(df.sample(10));

    
    X = pd.read_csv('data_ready.csv',sep=';')
    shape = X.shape
    st.text('Data Setelah Proses Pengkodean '+str(shape))
    st.write(X.sample(10));
    
    
with features:
    st.header('Pemilihan nilai K Menggunakan Elbow Method')
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    
    K = range(1, 10)
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)
        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_
     
     
    st.text('Metode Distortion')
    fig = plt.figure(figsize=(4,2))
    plt.plot(K,distortions,'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel("Distortion")
    st.write(fig)


    st.text('Metode Inertia')
    fig2 = plt.figure(figsize=(4,2))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    st.write(fig2)
    
    cost = []
    K = range(1,9)
 
    for k in K:
        kmode = KModes(n_clusters=k, init = "Cao", n_init = 1)
        kmode.fit_predict(X)
        cost.append(kmode.cost_)
        
    st.text('The Elbow Method using K-modes Cost')
    fig2a = plt.figure(figsize=(4,2))
    plt.plot(K, cost, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Cost')
    st.write(fig2a)
    

with training:
    st.header('K-Means Modelling')
    k_value  = st.slider('Nilai K (3-8)', min_value=3, max_value=10, step=1, value=4)
    

    model = KMeans(n_clusters=k_value) # isnisialisasi Kmeans dgn  nilai K yg dipilih
    label = model.fit_predict(X) #proses Clustering
    pca = PCA(2) #mengubah menajdi 2 kolom
    dfnp = pca.fit_transform(X) #Transform data
    center = pca.fit_transform(model.cluster_centers_)
    
    #dibuat menjadi dataFrame
    df['x1'] = dfnp[:,0]
    df['y1'] = dfnp[:,1]
    df['label'] = label
    
    
    fig3= plt.figure()
    sns.scatterplot(x='x1', y='y1',hue='label',data=df,alpha=1, s=40, palette='deep')
    plt.scatter(x=center[:, 0], y=center[:, 1], s=100, c='black', ec='red',label='centroid')
    plt.legend(bbox_to_anchor = (1.05,1))
    st.write(fig3)
    
    fig4= plt.figure()
    sns.countplot(x ='label', data=df)
    st.write(fig4)

    
    st.header('K-Modes Modelling')
    
    k_value  = st.slider('Nilai K (3-8)', min_value=3, max_value=8, step=1, value=5)
    model = KModes(n_clusters=k_value, init = "Cao", n_init = 1, verbose=1)# isnisialisasi Kmodes dgn  nilai K yg dipilih
    label = model.fit_predict(X) #proses Clustering
    
    
    pca = PCA(2) #mengubah menajdi 2 kolom
    dfnp = pca.fit_transform(X) #Transform data
    center = pca.fit_transform(model.cluster_centroids_)
    
    #dibuat menjadi dataFrame
    df['x1'] = dfnp[:,0]
    df['y1'] = dfnp[:,1]
    df['label'] = label
    
    
    fig3a= plt.figure()
    sns.scatterplot(x='x1', y='y1',hue='label',data=df,alpha=1, s=40, palette='deep')
    plt.scatter(x=center[:, 0], y=center[:, 1], s=100, c='black', ec='red',label='centroid')
    plt.legend(bbox_to_anchor = (1.05,1))
    st.write(fig3a)
    
    fig4a= plt.figure()
    sns.countplot(x ='label', data=df)
    st.write(fig4a)
    
