import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#import pickle
import base64

import io
 


column = ['','KABUPATEN_KOTA','TAHAPAN','JABATAN_PELAPOR','JABATAN_TERLAPOR','JENIS_KELAMIN_TERLAPOR','HASIL_KAJIAN','JENIS_PELANGGARAN']

def proses_data(DATAKU):
    TESISKU = DATAKU.drop(['NO','PROVINSI','DAERAH_PEMILIHAN','LAPORAN_TEMUAN','NOMOR', 'TANGGAL_TEMUAN_LAPORAN', 'NAMA_PELAPOR', 'UMUR_PELAPOR', 'NAMA_TERLAPOR','SAKSI'], axis = 1)
    KABUPATEN_KOTA   = pd.get_dummies(TESISKU.KABUPATEN_KOTA , drop_first=True)
    TAHAPAN  = pd.get_dummies(TESISKU.TAHAPAN, drop_first=True)
    JABATAN_PELAPOR = pd.get_dummies(TESISKU.JABATAN_PELAPOR, drop_first=True)
    JABATAN_TERLAPOR = pd.get_dummies(TESISKU.JABATAN_TERLAPOR , drop_first=True)
    JENIS_KELAMIN_TERLAPOR = pd.get_dummies(TESISKU.JENIS_KELAMIN_TERLAPOR, drop_first=True)
    HASIL_KAJIAN = pd.get_dummies(TESISKU. HASIL_KAJIAN, drop_first=True)  
    JENIS_PELANGGARAN = pd.get_dummies(TESISKU. JENIS_PELANGGARAN, drop_first=True)
    TESISKU2 = pd.concat([TESISKU, KABUPATEN_KOTA, TAHAPAN, JABATAN_PELAPOR, JABATAN_TERLAPOR, JENIS_KELAMIN_TERLAPOR, HASIL_KAJIAN, JENIS_PELANGGARAN], axis = 1)

    TESISKU3 = TESISKU2.merge(KABUPATEN_KOTA, left_index=True, right_index=True)
    TESISKU3 = TESISKU2.merge(TAHAPAN, left_index=True, right_index=True)
    TESISKU3 = TESISKU2.merge(JABATAN_PELAPOR, left_index=True, right_index=True)
    TESISKU3 = TESISKU2.merge(JABATAN_TERLAPOR, left_index=True, right_index=True)
    TESISKU3 = TESISKU2.merge(JENIS_KELAMIN_TERLAPOR, left_index=True, right_index=True)
    TESISKU3 = TESISKU2.merge(HASIL_KAJIAN, left_index=True, right_index=True)
    TESISKU3 = TESISKU2.merge(JENIS_PELANGGARAN, left_index=True, right_index=True)

    TESISKU4 = TESISKU3.drop(['KABUPATEN_KOTA','TAHAPAN','JABATAN_PELAPOR','HASIL_KAJIAN','JENIS_KELAMIN_TERLAPOR'], axis = 1)
    TESISKU4 = TESISKU4.drop(['JABATAN_TERLAPOR','UMUR_TERLAPOR','JENIS_PELANGGARAN'],axis=1)

    return TESISKU4
#end of proses_data

def get_table_download_link(df):
    towrite = io.BytesIO()
    pd.ExcelWriter(towrite, encoding='utf-8', index=False, header=True,engine='xlsxwriter')
    towrite.seek(0)  # reset pointer
    #csv = df.to_csv(index=False,sep=';')
    b64 = base64.b64encode(towrite.read()).decode()
    new_filename = "datahasil.xlsx"
    href= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{new_filename}">Download excel file</a>'

    #href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download file hasil clustering</a>'
    return href
#end of proses_data

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
    df_master = pd.read_csv('DATA_TESIS.csv',sep=';')  
    df1 = proses_data(df_master)
    pca = PCA(2) #mengubah menajdi 2 kolom
    df1 = pca.fit_transform(df1) #Transform data
   
     
    st.subheader('Pemilihan nilai K Menggunakan Elbow Method')
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    
    K = range(2, 11)
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
    k_value  = st.slider('Nilai K (3-10)', min_value=3, max_value=10, step=1, value=3)
    

    model = KMeans(n_clusters=k_value) # isnisialisasi Kmeans dgn  nilai K yg dipilih
    model.fit_predict(df1) #proses Clustering
    #pickle.dump(model, open('model_save', 'wb'))
    
    label = model.fit_predict(df1) #proses Clustering
    
    
    
    center = model.cluster_centers_
    
    #dibuat menjadi dataFrame
    df_master['x1'] = df1[:,0]
    df_master['y1'] = df1[:,1]
    df_master['cluster'] = label
    
    
    fig3= plt.figure()
    sns.scatterplot(x='x1', y='y1',hue='cluster',data=df_master,alpha=1, s=40, palette='deep')
    plt.scatter(x=center[:, 0], y=center[:, 1], s=100, c='black', ec='red',label='centroid')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    st.write(fig3)
    
    fig4= plt.figure()
    sns.countplot(x ='cluster', data=df_master)
    st.write(fig4)
    
    cluster = df_master['cluster'].unique()
    cluster.sort()

    
    choice = st.selectbox("Pilih Kluster",cluster)
    res = df_master.loc[df_master['cluster'] == choice]
    st.subheader('Cluster '+str(choice)+': '+str(res.shape[0])+' data')
    st.write(res)
     
     
     

    
    
def apps():
    k_value = int(st.text_input('Nilai K:',value=3))
    data = st.file_uploader("Upload a Dataset", type=["csv"])
    if data is not None:
        
        
        df = pd.read_csv(data,sep=';')
        st.dataframe(df)
        df1 = proses_data(df)
        #model = pickle.load(open('model_save.pkl', 'rb'))
        #label = model.predict(df1)
        
        pca = PCA(2) #mengubah menajdi 2 kolom
        df1 = pca.fit_transform(df1) #Transform data
        
        model = KMeans(n_clusters=k_value,random_state=101)
        model.fit(df1)
        label = model.predict(df1)
        center = model.cluster_centers_
        
        #dibuat menjadi dataFrame
        df['x1'] = df1[:,0]
        df['y1'] = df1[:,1]
        df['cluster'] = label
        st.write('Proses Dimulai...')
        for index, row in df.iterrows():
            st.write(str(row['NO'])+'... cluster: ',str(row['cluster']))
        st.write('Proses Selesai')
        st.dataframe(df)
        
        fig3= plt.figure()
        sns.scatterplot(x='x1', y='y1',hue='cluster',data=df,alpha=1, s=40, palette='deep')
        plt.scatter(x=center[:, 0], y=center[:, 1], s=100, c='black', ec='red',label='centroid')
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        st.write(fig3)

        fig4= plt.figure()
        sns.countplot(x ='cluster', data=df)
        st.write(fig4)
        
        
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
        
        
    
#end of apps
    
def main():
    """ Streamlit Pelanggaran Pilkada Jabar """

    activities = ['EDA','K-Means','Aplikasi Perhitungan']
    choice = st.sidebar.selectbox("Select Activities",activities)
    
    if choice == 'EDA':
        eda()
    elif choice == 'K-Means':
        kmeans()
    elif choice == 'Aplikasi Perhitungan':
        apps()
        

if __name__ == '__main__':
    main()
