import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from kmodes.kmodes import KModes

column = ['','KABUPATEN_KOTA','TAHAPAN','JABATAN_PELAPOR','JABATAN_TERLAPOR','JENIS_KELAMIN_TERLAPOR','HASIL_KAJIAN','JENIS_PELANGGARAN']
    
    
def eda():
    st.header('Datasets')
    df = pd.read_csv('DATA_TESIS.csv',sep=';')
    
    st.subheader('Data Awal '+str(df.shape))
    st.write(df.sample(10));
    
    if st.checkbox("Summary"):
        st.write(df[['UMUR_PELAPOR','UMUR_TERLAPOR']].describe())
    
    if st.checkbox("Show Columns Histogram"):
        selected_columns = st.selectbox("Select Column",column)
        if selected_columns != '':
            fig4= plt.figure()
            sns.countplot(x = selected_columns, data=df)
            plt.xticks(rotation=45,ha='right')
            st.write(fig4)

def kmeans():
    st.header('K-Means')     
    df1 = pd.read_csv('data_ready.csv',sep=',')
   
     
    st.subheader('Pemilihan nilai K Menggunakan Elbow Method')
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    
    K = range(1, 11)
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(df1)
        kmeanModel.fit(df1)
        distortions.append(sum(np.min(cdist(df1, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / df1.shape[0])
        inertias.append(kmeanModel.inertia_)
        mapping1[k] = sum(np.min(cdist(df1, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / df1.shape[0]
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

    st.header('K-Means Modelling')
    k_value  = st.slider('Nilai K (3-10)', min_value=3, max_value=10, step=1, value=4)
    

    model = KMeans(n_clusters=k_value) # isnisialisasi Kmeans dgn  nilai K yg dipilih
    label = model.fit_predict(df1) #proses Clustering
    pca = PCA(2) #mengubah menajdi 2 kolom
    dfnp = pca.fit_transform(df1) #Transform data
    center = pca.fit_transform(model.cluster_centers_)
    
    #dibuat menjadi dataFrame
    df1['x1'] = dfnp[:,0]
    df1['y1'] = dfnp[:,1]
    df1['label'] = label
    
    
    fig3= plt.figure()
    sns.scatterplot(x='x1', y='y1',hue='label',data=df1,alpha=1, s=40, palette='deep')
    plt.scatter(x=center[:, 0], y=center[:, 1], s=100, c='black', ec='red',label='centroid')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    st.write(fig3)
    
    fig4= plt.figure()
    sns.countplot(x ='label', data=df1)
    st.write(fig4)
     
     
     
     
def kmodes():
    st.header('K-Modes')
    df = X = pd.read_csv('data_ready.csv',sep=',')
    
    cost = []
    K = range(1,9)
 
    for k in K:
        kmode = KModes(n_clusters=k, init = "Cao", n_init = 1)
        kmode.fit_predict(df)
        cost.append(kmode.cost_)
        
    st.text('The Elbow Method using K-modes Cost')
    fig2a = plt.figure(figsize=(4,2))
    plt.plot(K, cost, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Cost')
    st.write(fig2a)

    st.header('K-Modes Modelling')
    
    k_value  = st.slider('Nilai K (3-10)', min_value=3, max_value=10, step=1, value=4)
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
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    st.write(fig3a)
    
    fig4a= plt.figure()
    sns.countplot(x ='label', data=df)
    st.write(fig4a)
    
    
    
def main():
    """ Streamlit Pelanggaran Pilkada Jabar """

    activities = ['EDA','K-Means','K-Modes']	
    choice = st.sidebar.selectbox("Select Activities",activities)
    
    if choice == 'EDA':
        eda()
    elif choice == 'K-Means':
        kmeans()
    elif choice == 'K-Modes':
        kmodes()
        

if __name__ == '__main__':
	main()