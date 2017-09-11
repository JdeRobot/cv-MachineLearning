/*
*
* Copyright 2014-2016 Ignacio San Roman Lana
*
* This file is part of OpenCV_ML_Tool
*
* OpenCV_ML_Tool is free software: you can redistribute it and/or
* modify it under the terms of the GNU General Public License as
* published by the Free Software Foundation, either version 3 of the
* License, or (at your option) any later version.
*
* OpenCV_ML_Tool is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with OpenCV_ML_Tool. If not, see http://www.gnu.org/licenses/.
*
* For those usages not covered by this license please contact with
* isanromanlana@gmail.com
*/

#include "dimensionalidad.h"

bool ordenar (float A,float B) {
    return A>B;
}

MLT::Dimensionalidad::Dimensionalidad(string Nombre){
    nombre=Nombre;
}

int MLT::Dimensionalidad::Reducir(vector<Mat> Imagenes, vector<Mat> &Reducidas, vector<float> Labels, Reducciones reduccion, Generacion::Info_Datos &info, bool save){
    int e=0;
    Mat lexic_data,output;
    Auxiliares aux;
    e=aux.Image2Lexic(Imagenes,lexic_data);
    if(Imagenes.size()!=Labels.size()){
        cout<<"ERROR en Reducir: El numero de datos y de etiquetas no coincide"<<endl;
        return 1;
    }
    if(Imagenes.empty()){
        cout<<"ERROR en Reducir: img esta vacia"<<endl;
        return 1;
    }
    if(Labels.size()==0){
        cout<<"ERROR en Reducir: Etiquetas esta vacia"<<endl;
        return 1;
    }
    for(uint i=0; i<Labels.size(); i++){
        if(Labels[i]==0){
            cout<<"ERROR en Reducir: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    if(reduccion.si_lda){
        Dimensionalidad dim(nombre);
        e=dim.LDA_matriz(lexic_data,Labels,reduccion.tam_reduc,reduccion.LDA,save);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en LDA_matriz"<<endl;
            return 1;
        }
        Mat Proyectada;
        e=dim.Proyeccion(lexic_data,Proyectada,LDA_DIM,reduccion.LDA);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en Proyeccion"<<endl;
            return 1;
        }
        Proyectada.copyTo(output);
    }
    else if(reduccion.si_pca){
        Dimensionalidad dim(nombre);
        e=dim.PCA_matriz(lexic_data,reduccion.tam_reduc,reduccion.PCA,save);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en PCA_matriz"<<endl;
            return 1;
        }
        Mat Proyectada;
        e=dim.Proyeccion(lexic_data,Proyectada,PCA_DIM,reduccion.PCA);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en Proyeccion"<<endl;
            return 1;
        }
        Proyectada.copyTo(output);
    }
    else if(reduccion.si_dist){
        Dimensionalidad dim(nombre);
        e=dim.MaxDist_matriz(lexic_data,Labels,reduccion.tam_reduc,reduccion.DS,save);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en MaxDist_matriz"<<endl;
            return 1;
        }
        Mat Proyectada;
        e=dim.Proyeccion(lexic_data,Proyectada,MAXDIST_DIM,reduccion.DS);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en Proyeccion"<<endl;
            return 1;
        }
        Proyectada.copyTo(output);
    }
    else if(reduccion.si_d_prime){
        Dimensionalidad dim(nombre);
        e=dim.D_Prime_matriz(lexic_data,Labels,reduccion.tam_reduc,reduccion.D_PRIME,save);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en D_PRIME_matriz"<<endl;
            return 1;
        }
        Mat Proyectada;
        e=dim.Proyeccion(lexic_data,Proyectada,D_PRIME_DIM,reduccion.D_PRIME);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en Proyeccion"<<endl;
            return 1;
        }
        Proyectada.copyTo(output);
    }
    for(int i=0; i<output.rows; i++){
        Mat fila;
        output.row(i).copyTo(fila);
        Reducidas.push_back(fila);
    }
    info.DS=reduccion.DS;
    info.D_PRIME=reduccion.D_PRIME;
    info.LDA=reduccion.LDA;
    info.PCA=reduccion.PCA;
    info.si_dist=reduccion.si_dist;
    info.si_d_prime=reduccion.si_d_prime;
    info.si_lda=reduccion.si_lda;
    info.si_pca=reduccion.si_pca;
    info.Tam_Y=1;
    info.Tam_X=Reducidas[0].cols;
    if(save){
        Generacion gen;
        gen.Guardar_Datos(nombre,Reducidas,Labels,info);
    }
    return 0;
}

int MLT::Dimensionalidad::LDA_matriz(Mat img, std::vector<float> Etiquetas, int tam_final, Mat &lda, bool guardar){
    if((uint)img.rows!=Etiquetas.size()){
        cout<<"ERROR en LDA_matriz: El numero de datos y de etiquetas no coincide"<<endl;
        return 1;
    }
    if(img.empty()){
        cout<<"ERROR en LDA_matriz: img esta vacia"<<endl;
        return 1;
    }
    if(Etiquetas.size()==0){
        cout<<"ERROR en LDA_matriz: Etiquetas esta vacia"<<endl;
        return 1;
    }
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0){
            cout<<"ERROR en LDA_matriz: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    Mat data;
    img.convertTo(data, CV_32FC1);
    bool negativa;
    Auxiliares ax;
    int num_etiq=ax.numero_etiquetas(Etiquetas,negativa);
    std::vector<int> num2label;
    for(int i=0; i<num_etiq; i++)
        num2label.push_back(num_etiq);
    std::vector<int> labels;
    for(uint i=0; i<Etiquetas.size(); i++){
        if(negativa && Etiquetas[i]<0)
            labels.push_back(Etiquetas[i]+1);
        else if(negativa)
            labels.push_back(Etiquetas[i]);
        else if(!negativa)
            labels.push_back(Etiquetas[i]-1);
    }
    int N = data.rows;
    int D = data.cols;
    if(D<tam_final){
        cout<<"ERROR en LDA: tam_final es mayor que el tamaño de los datos"<<endl;
        return 1;
    }
    int C = (int)num2label.size();
    if(C == 1) {
        cout<<"ERROR en LDA: Solo hay un tipo de etiqueta"<<endl;
        return 1;
    }
    if (labels.size() != static_cast<size_t>(N)) {
        cout<<"ERROR en LDA: Hay menos etiquetas que datos"<<endl;
        return 1;
    }
    if (N < D) {
#ifdef WARNINGS
        std::cout << "WARNING: El número de datos es menor que su tamaño"<< std::endl;
#endif
    }
    Mat meanTotal = Mat::zeros(1, D, data.type());
    std::vector<Mat> meanClass(C);
    std::vector<int> numClass(C);
    for (int i = 0; i < C; i++) {
        numClass[i] = 0;
        meanClass[i] = Mat::zeros(1, D, data.type()); //! Dx1 image vector
    }
    for (int i = 0; i < N; i++) {
        Mat instance = data.row(i);
        int classIdx = labels[i];
        meanTotal=meanTotal+instance;
        meanClass[classIdx]=meanClass[classIdx]+instance;
        numClass[classIdx]++;
    }
    meanTotal.convertTo(meanTotal, meanTotal.type(), 1.0 / static_cast<double> (N));
    for (int i = 0; i < C; i++) {
        meanClass[i].convertTo(meanClass[i], meanClass[i].type(), 1.0 / static_cast<double> (numClass[i]));
    }
    for (int i = 0; i < N; i++) {
        int classIdx = labels[i];
        Mat instance = data.row(i);
        instance=instance-meanClass[classIdx];
    }
    Mat Sw = Mat::zeros(D, D, data.type());
    mulTransposed(data, Sw, true);
    Mat Sb = Mat::zeros(D, D, data.type());
    for (int i = 0; i < C; i++) {
        Mat tmp=meanClass[i]-meanTotal;
        tmp=tmp.t()*tmp;
        Sb=Sb+tmp;
    }
    Mat M=Sw.inv()*Sb;
    Mat eigenvalues,eigenvectors;
    cv::eigen(M,eigenvalues,eigenvectors);
    cv::Mat aux;
    eigenvectors.colRange(0,tam_final).copyTo(aux);
    cv::transpose(aux,lda);
    if(guardar){
        DIR    *dir_p = opendir ("../Data/Configuracion");
        if(dir_p == NULL) {
            string command = "mkdir ../Data/Configuracion";
            int er=system(command.c_str());
            if(er==1){
                cout<<"ERROR en Read_Data: Error al crear carpeta"<<endl;
                return 1;
            }
        }
        string g="../Data/Configuracion/"+nombre+"_LDA.xml";
        cv::FileStorage archivo_w(g,CV_STORAGE_WRITE);
        archivo_w<<"mean"<<meanTotal.t();
        archivo_w<<"LDA"<<lda;
        archivo_w.release();
    }
    return 0;
}


int MLT::Dimensionalidad::PCA_matriz(Mat img,int tam_final, Mat &pca, bool guardar){
    if(img.empty()){
        cout<<"ERROR en PCA_matriz: img esta vacia"<<endl;
        return 1;
    }
    Mat data;
    Mat img_t;
    cv::transpose(img,img_t);
    img_t.convertTo(data, CV_32FC1);

    int m = data.rows;
    int n = data.cols;
    if(m<tam_final){
        cout<<"ERROR en PCA: tam_final es mayor que el tamaño de los datos"<<endl;
        return 1;
    }
    Mat mean=Mat::zeros(m,1,CV_32FC1);
    for(int i=0; i<n; i++){
        mean.col(0)=mean.col(0)+data.col(i);
    }
    mean=mean/n;
    for(int i=0; i<n; i++)
        data.col(i)=data.col(i)-mean.col(0);
    Mat Covariance=(1/(double)n)*data*data.t();
    Mat eigenvalues,eigenvectors;
    cv::eigen(Covariance,eigenvalues,eigenvectors);
    Mat aux=Mat::zeros(m,tam_final,CV_32FC1);
	for(int i=0; i<tam_final; i++){
//		eigenvectors.col(tam_final-1-i).copyTo(aux.col(i));
        eigenvectors.col(i).copyTo(aux.col(i));
	}
    cv::transpose(aux,pca);
    if(guardar){
        DIR    *dir_p = opendir ("../Data/Configuracion");
        if(dir_p == NULL) {
            string command = "mkdir ../Data/Configuracion";
            int er=system(command.c_str());
            if(er==1){
                cout<<"ERROR en Read_Data: Error al crear carpeta"<<endl;
                return 1;
            }
        }
        string g="../Data/Configuracion/"+nombre+"_PCA.xml";
        cv::FileStorage archivo_w(g,CV_STORAGE_WRITE);
        archivo_w<<"mean"<<mean;
        archivo_w<<"PCA"<<pca;
        archivo_w.release();
    }
    return 0;
}

int MLT::Dimensionalidad::MaxDist_matriz(Mat img, std::vector<float> Etiquetas, int tam_final, Mat &mat_reduc, bool guardar){
    if(img.empty()){
        cout<<"ERROR en MaxDist_matriz: img esta vacia"<<endl;
        return 1;
    }
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0){
            cout<<"ERROR en MaxDist_matriz: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    Mat data;
    img.convertTo(data, CV_32FC1);
    if(data.cols<tam_final){
        cout<<"ERROR en MaxDist_matriz: tam_final es mayor que el tamaño de los datos"<<endl;
        return 1;
    }
    Mat Media=Mat::zeros(1,data.cols,CV_32FC1);
    for(int i=0; i<data.rows; i++){
        Media.row(0)=Media.row(0)+data.row(i);
    }
    Media=Media/data.rows;
    vector<Mat> Medias,Varianzas,Des_Tipics;
    Auxiliares ax;
    bool neg;
    int num_etiq=ax.numero_etiquetas(Etiquetas,neg);
    for(int i=0; i<num_etiq; i++){
        Mat zer=Mat::zeros(1,data.cols,CV_32FC1);
        Mat zer2=Mat::zeros(1,data.cols,CV_32FC1);
        Medias.push_back(zer);
        Varianzas.push_back(zer2);
    }
    vector<int> num(num_etiq);
    for(int i=0; i<num_etiq; i++)
        num[i]=0;
    for(int i=0; i<data.rows; i++){
        if(neg && Etiquetas[i]==-1){
            Medias[0]=Medias[0]+data.row(i);
            num[0]=num[0]+1;
        }
        else if(neg && Etiquetas[i]!=-1){
            Medias[Etiquetas[i]]=Medias[Etiquetas[i]]+data.row(i);
            num[Etiquetas[i]]=num[Etiquetas[i]]+1;
        }
        else if(!neg){
            Medias[Etiquetas[i]-1]=Medias[Etiquetas[i]-1]+data.row(i);
            num[Etiquetas[i]-1]=num[Etiquetas[i]-1]+1;
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        Medias[i]=Medias[i]/num[i];
    }
    for(int i=0; i<data.rows; i++){
        if(neg && Etiquetas[i]==-1){
            Mat A=data.row(i)-Medias[0];
            Varianzas[0]=Varianzas[0]+A.mul(A);
        }
        else if(neg && Etiquetas[i]!=-1){
            Mat A=data.row(i)-Medias[Etiquetas[i]];
            Varianzas[Etiquetas[i]]=Varianzas[Etiquetas[i]]+A.mul(A);
        }
        else if(!neg){
            Mat A=data.row(i)-Medias[Etiquetas[i]-1];
            Varianzas[Etiquetas[i]-1]=Varianzas[Etiquetas[i]-1]+A.mul(A);
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        Varianzas[i]=Varianzas[i]/num[i];
        Des_Tipics.push_back(Varianzas[i]);
        for(int j=0; j<Varianzas[i].cols; j++){
            Des_Tipics[i].at<float>(0,j)=sqrt(Varianzas[i].at<float>(0,j));
        }
    }
    vector<Mat> dist_etiq;
    for(uint i=0; i<Medias.size(); i++){
        Mat zer=Mat::zeros(1,Media.cols,CV_32FC1);
        dist_etiq.push_back(zer);
    }
    for(uint i=0; i<dist_etiq.size(); i++){
        Mat distancia=cv::abs(Media.row(0)-Medias[i].row(0));
        dist_etiq[i]=distancia.row(0)-Des_Tipics[i].row(0);
    }
    Mat distancias=Mat::zeros(1,Media.cols,CV_32FC1);
    for(uint i=0; i<dist_etiq.size(); i++)
        distancias.row(0)=distancias.row(0)+dist_etiq[i].row(0);
    vector<bool> Validos(Media.cols);
    for(uint i=0; i<Validos.size(); i++)
        Validos[i]=false;
    vector<float> dist_vec;
    for(int i=0; i<distancias.cols; i++)
        dist_vec.push_back(distancias.at<float>(0,i));
    for(int i=0; i<tam_final; i++){
        float maximo=-99999999;
        int pos=0;
        for(uint j=0; j<dist_vec.size(); j++){
            if(dist_vec[j]>maximo){
                maximo=dist_vec[j];
                pos=j;
            }
        }
        Validos[pos]=true;
        dist_vec[pos]=-99999999;
    }
    mat_reduc=Mat::zeros(tam_final,data.cols,CV_32FC1);
    int contador=0;
    for(uint i=0; i<Validos.size(); i++)
        if(Validos[i]){
            mat_reduc.at<float>(contador,i)=1;
            contador++;
        }
    if(contador!=tam_final){
        cout<<"ERROR en MaxDist_matriz: No se ha conseguido el tamaño final deseado"<<endl;
        return 1;
    }
    if(guardar){
        DIR    *dir_p = opendir ("../Data/Configuracion");
        if(dir_p == NULL) {
            string command = "mkdir ../Data/Configuracion";
            int er=system(command.c_str());
            if(er==1){
                cout<<"ERROR en Read_Data: Error al crear carpeta"<<endl;
                return 1;
            }
        }
        string g="../Data/Configuracion/"+nombre+"_MAXDIST.xml";
        cv::FileStorage archivo_w(g,CV_STORAGE_WRITE);
        archivo_w<<"mean"<<Media.t();
        archivo_w<<"MAXDIST"<<mat_reduc;
        archivo_w.release();
    }
    return 0;
}

int MLT::Dimensionalidad::D_Prime_matriz(Mat img, std::vector<float> Etiquetas, int tam_final, Mat &mat_reduc, bool guardar){
    if(img.empty()){
        cout<<"ERROR en D_Prime_matriz: img esta vacia"<<endl;
        return 1;
    }
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0){
            cout<<"ERROR en D_Prime_matriz: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    Mat data;
    img.convertTo(data, CV_32FC1);
    if(data.cols<tam_final){
        cout<<"ERROR en D_Prime_matriz: tam_final es mayor que el tamaño de los datos"<<endl;
        return 1;
    }
    Mat Media=Mat::zeros(1,data.cols,CV_32FC1);
    for(int i=0; i<data.rows; i++){
        Media.row(0)=Media.row(0)+data.row(i);
    }
    Media=Media/data.rows;
    vector<Mat> Medias,Varianzas;
    Auxiliares ax;
    bool neg;
    int num_etiq=ax.numero_etiquetas(Etiquetas,neg);
    for(int i=0; i<num_etiq; i++){
        Mat zer=Mat::zeros(1,data.cols,CV_32FC1);
        Mat zer2=Mat::zeros(1,data.cols,CV_32FC1);
        Medias.push_back(zer);
        Varianzas.push_back(zer2);
    }
    vector<int> num(num_etiq);
    for(int i=0; i<num_etiq; i++)
        num[i]=0;
    for(int i=0; i<data.rows; i++){
        if(neg && Etiquetas[i]==-1){
            Medias[0]=Medias[0]+data.row(i);
            num[0]=num[0]+1;
        }
        else if(neg && Etiquetas[i]!=-1){
            Medias[Etiquetas[i]]=Medias[Etiquetas[i]]+data.row(i);
            num[Etiquetas[i]]=num[Etiquetas[i]]+1;
        }
        else if(!neg){
            Medias[Etiquetas[i]-1]=Medias[Etiquetas[i]-1]+data.row(i);
            num[Etiquetas[i]-1]=num[Etiquetas[i]-1]+1;
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        Medias[i]=Medias[i]/num[i];
    }
    for(int i=0; i<data.rows; i++){
        if(neg && Etiquetas[i]==-1){
            Mat A=data.row(i)-Medias[0];
            Varianzas[0]=Varianzas[0]+A.mul(A);
        }
        else if(neg && Etiquetas[i]!=-1){
            Mat A=data.row(i)-Medias[Etiquetas[i]];
            Varianzas[Etiquetas[i]]=Varianzas[Etiquetas[i]]+A.mul(A);
        }
        else if(!neg){
            Mat A=data.row(i)-Medias[Etiquetas[i]-1];
            Varianzas[Etiquetas[i]-1]=Varianzas[Etiquetas[i]-1]+A.mul(A);
        }
    }
    Mat d_prime=Mat::zeros(1,data.cols,CV_32FC1);
    for(uint i=0; i<Medias.size(); i++){
        for(uint j=0; j<Medias.size(); j++){
            if(i!=j){
                for(int k=0; k<d_prime.cols; k++){
                    d_prime.at<float>(0,k)=d_prime.at<float>(0,k)+abs((Medias[i].at<float>(0,k)-Medias[j].at<float>(0,k))/sqrt(Varianzas[i].at<float>(0,k)+Varianzas[j].at<float>(0,k)));
                }
            }
        }
    }
    vector<bool> Validos(d_prime.cols);
    for(uint i=0; i<Validos.size(); i++)
        Validos[i]=false;
    vector<float> dist_vec;
    for(int i=0; i<d_prime.cols; i++)
        dist_vec.push_back(d_prime.at<float>(0,i));
    for(int i=0; i<tam_final; i++){
        float maximo=-99999999;
        int pos=0;
        for(uint j=0; j<dist_vec.size(); j++){
            if(dist_vec[j]>maximo){
                maximo=dist_vec[j];
                pos=j;
            }
        }
        Validos[pos]=true;
        dist_vec[pos]=-99999999;
    }
    mat_reduc=Mat::zeros(tam_final,data.cols,CV_32FC1);
    int contador=0;
    for(uint i=0; i<Validos.size(); i++)
        if(Validos[i]){
            mat_reduc.at<float>(contador,i)=1;
            contador++;
        }
    if(contador!=tam_final){
        cout<<"ERROR en D_Prime_matriz: No se ha conseguido el tamaño final deseado"<<endl;
        return 1;
    }
    if(guardar){
        DIR    *dir_p = opendir ("../Data/Configuracion");
        if(dir_p == NULL) {
            string command = "mkdir ../Data/Configuracion";
            int er=system(command.c_str());
            if(er==1){
                cout<<"ERROR en Read_Data: Error al crear carpeta"<<endl;
                return 1;
            }
        }
        string g="../Data/Configuracion/"+nombre+"_D_PRIME.xml";
        cv::FileStorage archivo_w(g,CV_STORAGE_WRITE);
        archivo_w<<"mean"<<Media.t();
        archivo_w<<"D_PRIME"<<mat_reduc;
        archivo_w.release();
    }
    return 0;
}

int MLT::Dimensionalidad::Proyeccion(Mat img, Mat &Proyectada,int tipo, Mat reduc){
    if(img.empty()){
        cout<<"ERROR en Proyeccion: img esta vacia"<<endl;
        return 1;
    }
    if(tipo==LDA_DIM){
        if(reduc.empty()){
            string g="../Data/Configuracion/"+nombre+"_LDA.xml";
            cv::FileStorage archivo_r(g,CV_STORAGE_READ);
            if(archivo_r.isOpened()){
                archivo_r["LDA"]>>reduc;
            }
            else{
                cout<<"ERROR en Proyeccion: Matriz lda vacía y no existe archivo LDA.xml"<<endl;
                return 1;
            }
            archivo_r.release();
        }
    }
    else if(tipo==PCA_DIM){
        if(reduc.empty()){
            string g="../Data/Configuracion/"+nombre+"_PCA.xml";
            cv::FileStorage archivo_r(g,CV_STORAGE_READ);
            if(archivo_r.isOpened()){
                archivo_r["PCA"]>>reduc;
            }
            else{
                cout<<"ERROR en Proyeccion: Matriz pca vacía y no existe archivo PCA.xml"<<endl;
                return 1;
            }
            archivo_r.release();
        }
    }
    else if(tipo==MAXDIST_DIM){
        if(reduc.empty()){
            string g="../Data/Configuracion/"+nombre+"_MAXDIST.xml";
            cv::FileStorage archivo_r(g,CV_STORAGE_READ);
            if(archivo_r.isOpened()){
                archivo_r["MAXDIST"]>>reduc;
            }
            else{
                cout<<"ERROR en Proyeccion: Matriz maxdist vacía y no existe archivo MAXDIST.xml"<<endl;
                return 1;
            }
            archivo_r.release();
        }
    }
    else if(tipo==D_PRIME_DIM){
        if(reduc.empty()){
            string g="../Data/Configuracion/"+nombre+"_D_PRIME.xml";
            cv::FileStorage archivo_r(g,CV_STORAGE_READ);
            if(archivo_r.isOpened()){
                archivo_r["D_PRIME"]>>reduc;
            }
            else{
                cout<<"ERROR en Proyeccion: Matriz d_prime vacía y no existe archivo D_PRIME.xml"<<endl;
                return 1;
            }
            archivo_r.release();
        }
    }
    else{
        cout<<"ERROR en Proyeccion: Tipo de reduccion invalido"<<endl;
        return 1;
    }
    if(img.cols!=reduc.cols){
        cout<<"ERROR en Proyeccion: Los datos no tienen el mismo tamaño que en la matriz de reduccion"<<endl;
        return 1;
    }
    reduc.convertTo(reduc,CV_32FC1);
    img.convertTo(img,CV_32FC1);
    Proyectada=Mat::zeros(img.rows,reduc.cols,CV_32FC1);
    cv::Mat transpuesta;
    cv::transpose(img,transpuesta);
    if(img.empty() || reduc.empty()){
        cout<<"ERROR en Proyeccion: Imagen o matriz de reduccion vacías"<<endl;
        return 1;
    }
    cv::Mat Proy=Mat::zeros(reduc.rows,transpuesta.cols,CV_32FC1);
    for(int i=0; i<transpuesta.cols; i++){
        Proy.col(i)=reduc*transpuesta.col(i);
    }
    cv::transpose(Proy,Proyectada);
    return 0;
}

int MLT::Dimensionalidad::Retro_Proyeccion(Mat img, Mat &Proyectada,int tipo){
    if(img.empty()){
        cout<<"ERROR en Retro_Proyeccion: img esta vacia"<<endl;
        return 1;
    }
    Mat mean;
    Mat reduc;
    if(tipo==LDA_DIM){
        string g="../Data/Configuracion/"+nombre+"_LDA.xml";
        cv::FileStorage archivo_r(g,CV_STORAGE_READ);
        if(archivo_r.isOpened()){
            archivo_r["mean"]>>mean;
            archivo_r["LDA"]>>reduc;
        }
        else{
            cout<<"ERROR en Retro_Proyeccion: No existe archivo LDA.xml"<<endl;
            return 1;
        }
        archivo_r.release();
    }
    else if(tipo==PCA_DIM){
        string g="../Data/Configuracion/"+nombre+"_PCA.xml";
        cv::FileStorage archivo_r(g,CV_STORAGE_READ);
        if(archivo_r.isOpened()){
            archivo_r["mean"]>>mean;
            archivo_r["PCA"]>>reduc;
        }
        else{
            cout<<"ERROR en Retro_Proyeccion:No existe archivo PCA.xml"<<endl;
            return 1;
        }
        archivo_r.release();
    }
    else if(tipo==MAXDIST_DIM){
        string g="../Data/Configuracion/"+nombre+"_MAXDIST.xml";
        cv::FileStorage archivo_r(g,CV_STORAGE_READ);
        if(archivo_r.isOpened()){
            archivo_r["mean"]>>mean;
            archivo_r["MAXDIST"]>>reduc;
        }
        else{
            cout<<"ERROR en Retro_Proyeccion:No existe archivo MAXDIST.xml"<<endl;
            return 1;
        }
        archivo_r.release();
    }
    else if(tipo==D_PRIME_DIM){
        string g="../Data/Configuracion/"+nombre+"_D_PRIME.xml";
        cv::FileStorage archivo_r(g,CV_STORAGE_READ);
        if(archivo_r.isOpened()){
            archivo_r["mean"]>>mean;
            archivo_r["D_PRIME"]>>reduc;
        }
        else{
            cout<<"ERROR en Retro_Proyeccion:No existe archivo D_PRIME.xml"<<endl;
            return 1;
        }
        archivo_r.release();
    }
    else{
        cout<<"ERROR en Retro_Proyeccion: Tipo de reduccion invalido"<<endl;
        return 1;
    }
    reduc.convertTo(reduc,CV_32FC1);
    img.convertTo(img,CV_32FC1);
    Proyectada=Mat::zeros(img.rows,reduc.cols,CV_32FC1);
    cv::Mat transpuesta;
    cv::transpose(img,transpuesta);
    if(img.empty() || reduc.empty()){
        cout<<"ERROR en Retro_Proyeccion: Imagen o matriz de reduccion vacías"<<endl;
        return 1;
    }
    cv::Mat Proy=Mat::zeros(reduc.cols,transpuesta.cols,CV_32FC1);
    for(int i=0; i<transpuesta.cols; i++){
        Proy.col(i)=(reduc.t()*transpuesta.col(i)+mean);
    }
    cv::transpose(Proy,Proyectada);
    return 0;
}

int MLT::Dimensionalidad::Calidad_dimensiones_distancia(vector<Mat> img, vector<float> Etiquetas, int tipo_reduccion, int dim_max, Mat &Separabilidad, Mat &Separabilidad_acumulada, int &dim_optim){
    int e=0;
    if(img.empty()){
        cout<<"ERROR en Calidad_dimensiones_distancia: img esta vacia"<<endl;
        return 1;
    }
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0){
            cout<<"ERROR en Calidad_dimensiones_distancia: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    Auxiliares ax;
    Mat Datos;
    ax.Image2Lexic(img,Datos);
    Mat data;
    Datos.convertTo(data, CV_32FC1);
    Mat reducida,mat_reduc;
    Mat Sep=Mat::zeros(dim_max,2,CV_32FC1);
    Mat Sep_acum=Mat::zeros(dim_max,2,CV_32FC1);
    if(tipo_reduccion==LDA_DIM){
        e=LDA_matriz(data,Etiquetas,dim_max,mat_reduc,false);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_distancia: Error en LDA_matriz"<<endl;
            return 1;
        }
        e=Proyeccion(data,reducida,LDA_DIM,mat_reduc);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_distancia: Error en Proyeccion"<<endl;
            return 1;
        }
    }
    else if(tipo_reduccion==PCA_DIM){
        e=PCA_matriz(data,dim_max,mat_reduc,false);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_distancia: Error en PCA_matriz"<<endl;
            return 1;
        }
        e=Proyeccion(data,reducida,PCA_DIM,mat_reduc);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_distancia: Error en Proyeccion"<<endl;
            return 1;
        }
    }
    else if(tipo_reduccion==MAXDIST_DIM){
        e=MaxDist_matriz(data,Etiquetas,dim_max,mat_reduc,false);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_distancia: Error en MaxDist_matriz"<<endl;
            return 1;
        }
        e=Proyeccion(data,reducida,MAXDIST_DIM,mat_reduc);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_distancia: Error en Proyeccion"<<endl;
            return 1;
        }
    }
    else if(tipo_reduccion==D_PRIME_DIM){
        e=D_Prime_matriz(data,Etiquetas,dim_max,mat_reduc,false);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_distancia: Error en D_Prime_matriz"<<endl;
            return 1;
        }
        e=Proyeccion(data,reducida,D_PRIME_DIM,mat_reduc);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_distancia: Error en Proyeccion"<<endl;
            return 1;
        }
    }
    else{
        cout<<"ERROR en Calidad_dimensiones_distancia: Tipo de reduccion de dimensionalidad no valido"<<endl;
        return 1;
    }
    Mat Media=Mat::zeros(1,reducida.cols,CV_32FC1);
    for(int i=0; i<reducida.rows; i++){
        Media.row(0)=Media.row(0)+reducida.row(i);
    }
    Media=Media/reducida.rows;
    vector<Mat> Medias,Varianzas,Des_Tipics;
    bool neg;
    int num_etiq=ax.numero_etiquetas(Etiquetas,neg);
    for(int i=0; i<num_etiq; i++){
        Mat zer=Mat::zeros(1,reducida.cols,CV_32FC1);
        Mat zer2=Mat::zeros(1,reducida.cols,CV_32FC1);
        Medias.push_back(zer);
        Varianzas.push_back(zer2);
    }
    vector<int> num(num_etiq);
    for(int i=0; i<num_etiq; i++)
        num[i]=0;
    for(int i=0; i<reducida.rows; i++){
        if(neg && Etiquetas[i]==-1){
            Medias[0]=Medias[0]+reducida.row(i);
            num[0]=num[0]+1;
        }
        else if(neg && Etiquetas[i]!=-1){
            Medias[Etiquetas[i]]=Medias[Etiquetas[i]]+reducida.row(i);
            num[Etiquetas[i]]=num[Etiquetas[i]]+1;
        }
        else if(!neg){
            Medias[Etiquetas[i]-1]=Medias[Etiquetas[i]-1]+reducida.row(i);
            num[Etiquetas[i]-1]=num[Etiquetas[i]-1]+1;
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        Medias[i]=Medias[i]/num[i];
    }

    for(int i=0; i<reducida.rows; i++){
        if(neg && Etiquetas[i]==-1){
            Mat A=reducida.row(i)-Medias[0];
            Varianzas[0]=Varianzas[0]+A.mul(A);
        }
        else if(neg && Etiquetas[i]!=-1){
            Mat A=reducida.row(i)-Medias[Etiquetas[i]];
            Varianzas[Etiquetas[i]]=Varianzas[Etiquetas[i]]+A.mul(A);
        }
        else if(!neg){
            Mat A=reducida.row(i)-Medias[Etiquetas[i]-1];
            Varianzas[Etiquetas[i]-1]=Varianzas[Etiquetas[i]-1]+A.mul(A);
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        Varianzas[i]=Varianzas[i]/num[i];
        Des_Tipics.push_back(Varianzas[i]);
        for(int j=0; j<Varianzas[i].cols; j++){
            Des_Tipics[i].at<float>(0,j)=sqrt(Varianzas[i].at<float>(0,j));
        }
    }
    vector<Mat> dist_etiq;
    for(uint i=0; i<Medias.size(); i++){
        Mat zer=Mat::zeros(1,Media.cols,CV_32FC1);
        dist_etiq.push_back(zer);
    }
    for(uint i=0; i<dist_etiq.size(); i++){
        Mat distancia=cv::abs(Media.row(0)-Medias[i].row(0));
        dist_etiq[i]=distancia.row(0)-Des_Tipics[i].row(0);
    }
    Mat distancias=Mat::zeros(1,Media.cols,CV_32FC1);
    for(uint i=0; i<dist_etiq.size(); i++)
        distancias.row(0)=distancias.row(0)+dist_etiq[i].row(0);
    vector<float> ordenados;
    for(int i=0; i<dim_max; i++){
        ordenados.push_back(distancias.at<float>(0,i));
    }
    sort(ordenados.begin(), ordenados.end(), ordenar);
    for(int i=0; i<dim_max; i++){
        float dist_total=0;
        for(int j=0; j<i+1; j++){
            dist_total=dist_total+ordenados[j];
        }
        Sep.at<float>(i,0)=i+1;
        Sep.at<float>(i,1)=ordenados[i];
        Sep_acum.at<float>(i,0)=i+1;
        Sep_acum.at<float>(i,1)=dist_total;
    }
    int cont=0;
    while(Sep_acum.at<float>(cont+1,1)>Sep_acum.at<float>(cont,1) && (cont-1)<Sep_acum.rows)
        cont++;
    dim_optim=cont+1;
    Sep_acum.copyTo(Separabilidad_acumulada);
    Sep.copyTo(Separabilidad);
    return 0;
}

int MLT::Dimensionalidad::Calidad_dimensiones_d_prime(vector<Mat> img, vector<float> Etiquetas, int tipo_reduccion, int dim_max, Mat &Separabilidad, Mat &Separabilidad_acumulada, int &dim_optim){
    int e=0;
    if(img.empty()){
        cout<<"ERROR en Calidad_dimensiones_d_prime: img esta vacia"<<endl;
        return 1;
    }
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0){
            cout<<"ERROR en Calidad_dimensiones_d_prime: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    Auxiliares ax;
    Mat Datos;
    ax.Image2Lexic(img,Datos);
    Mat data;
    Datos.convertTo(data, CV_32FC1);
    Mat reducida,mat_reduc;
    Mat Sep=Mat::zeros(dim_max,2,CV_32FC1);
    Mat Sep_acum=Mat::zeros(dim_max,2,CV_32FC1);
    if(tipo_reduccion==LDA_DIM){
        e=LDA_matriz(data,Etiquetas,dim_max,mat_reduc,false);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_d_prime: Error en LDA_matriz"<<endl;
            return 1;
        }
        e=Proyeccion(data,reducida,LDA_DIM,mat_reduc);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_d_prime: Error en Proyeccion"<<endl;
            return 1;
        }
    }
    else if(tipo_reduccion==PCA_DIM){
        e=PCA_matriz(data,dim_max,mat_reduc,false);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_d_prime: Error en PCA_matriz"<<endl;
            return 1;
        }
        e=Proyeccion(data,reducida,PCA_DIM,mat_reduc);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_d_prime: Error en Proyeccion"<<endl;
            return 1;
        }
    }
    else if(tipo_reduccion==MAXDIST_DIM){
        e=MaxDist_matriz(data,Etiquetas,dim_max,mat_reduc,false);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_d_prime: Error en MaxDist_matriz"<<endl;
            return 1;
        }
        e=Proyeccion(data,reducida,MAXDIST_DIM,mat_reduc);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_d_prime: Error en Proyeccion"<<endl;
            return 1;
        }
    }
    else if(tipo_reduccion==D_PRIME_DIM){
        e=D_Prime_matriz(data,Etiquetas,dim_max,mat_reduc,false);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_d_prime: Error en D_Prime_matriz"<<endl;
            return 1;
        }
        e=Proyeccion(data,reducida,D_PRIME_DIM,mat_reduc);
        if(e==1){
            cout<<"ERROR en Calidad_dimensiones_d_prime: Error en Proyeccion"<<endl;
            return 1;
        }
    }
    else{
        cout<<"ERROR en Calidad_dimensiones_d_prime: Tipo de reduccion de dimensionalidad no valido"<<endl;
        return 1;
    }
    vector<Mat> Medias,Varianzas;
    bool neg;
    int num_etiq=ax.numero_etiquetas(Etiquetas,neg);
    for(int i=0; i<num_etiq; i++){
        Mat zer=Mat::zeros(1,reducida.cols,CV_32FC1);
        Mat zer2=Mat::zeros(1,reducida.cols,CV_32FC1);
        Medias.push_back(zer);
        Varianzas.push_back(zer2);
    }
    vector<int> num(num_etiq);
    for(int i=0; i<num_etiq; i++)
        num[i]=0;
    for(int i=0; i<reducida.rows; i++){
        if(neg && Etiquetas[i]==-1){
            Medias[0]=Medias[0]+reducida.row(i);
            num[0]=num[0]+1;
        }
        else if(neg && Etiquetas[i]!=-1){
            Medias[Etiquetas[i]]=Medias[Etiquetas[i]]+reducida.row(i);
            num[Etiquetas[i]]=num[Etiquetas[i]]+1;
        }
        else if(!neg){
            Medias[Etiquetas[i]-1]=Medias[Etiquetas[i]-1]+reducida.row(i);
            num[Etiquetas[i]-1]=num[Etiquetas[i]-1]+1;
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        Medias[i]=Medias[i]/num[i];
    }

    for(int i=0; i<reducida.rows; i++){
        if(neg && Etiquetas[i]==-1){
            Mat A=reducida.row(i)-Medias[0];
            Varianzas[0]=Varianzas[0]+A.mul(A);
        }
        else if(neg && Etiquetas[i]!=-1){
            Mat A=reducida.row(i)-Medias[Etiquetas[i]];
            Varianzas[Etiquetas[i]]=Varianzas[Etiquetas[i]]+A.mul(A);
        }
        else if(!neg){
            Mat A=reducida.row(i)-Medias[Etiquetas[i]-1];
            Varianzas[Etiquetas[i]-1]=Varianzas[Etiquetas[i]-1]+A.mul(A);
        }
    }

    Mat d_prime=Mat::zeros(1,data.cols,CV_32FC1);
    for(uint i=0; i<Medias.size(); i++){
        for(uint j=0; j<Medias.size(); j++){
            if(i!=j){
                for(int k=0; k<d_prime.cols; k++){
                    d_prime.at<float>(0,k)=d_prime.at<float>(0,k)+abs((Medias[i].at<float>(0,k)-Medias[j].at<float>(0,k))/sqrt(Varianzas[i].at<float>(0,k)+Varianzas[j].at<float>(0,k)));
                }
            }
        }
    }
    vector<float> ordenados;
    for(int i=0; i<dim_max; i++){
        ordenados.push_back(d_prime.at<float>(0,i));
    }
    sort(ordenados.begin(), ordenados.end(), ordenar);
    for(int i=0; i<dim_max; i++){
        float dist_total=0;
        for(int j=0; j<i+1; j++){
            dist_total=dist_total+ordenados[j];
        }
        Sep.at<float>(i,0)=i+1;
        Sep.at<float>(i,1)=ordenados[i];
        Sep_acum.at<float>(i,0)=i+1;
        Sep_acum.at<float>(i,1)=dist_total;
    }
    int cont=0;
    while(Sep_acum.at<float>(cont+1,1)>Sep_acum.at<float>(cont,1) && (cont-1)<Sep_acum.rows)
        cont++;
    dim_optim=cont+1;
    Sep_acum.copyTo(Separabilidad_acumulada);
    Sep.copyTo(Separabilidad);
    return 0;
}
