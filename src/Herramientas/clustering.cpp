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

#include "clustering.h"

MLT::Clustering::Clustering(){}
MLT::Clustering::~Clustering(){}

int MLT::Clustering::K_mean(vector<Mat> Data, int K, vector<float> &Labels, Mat &Centers, int attempts, int inicializacion){
//    Data – Data for clustering.
//    K – Number of clusters to split the set by.
//    labels – Input/output integer array that stores the cluster indices for every sample.
//    inicializacion:
//    -KMEANS_RANDOM_CENTERS Select random initial centers in each attempt.
//    -KMEANS_PP_CENTERS Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007].
//    -KMEANS_USE_INITIAL_LABELS During the first (and possibly the only) attempt, use the user-supplied labels
//    instead of computing them from the initial centers. For the second and further attempts, use the random or semi-random centers. Use one of KMEANS_*_CENTERS flag to specify the exact method.
//    centers – Output matrix of the cluster centers, one row per each cluster center.
    int e=0;
    if(Data.size()==0){
        cout<<"ERROR en K_mean: No hay datos"<<endl;
        this->error=1;
        return 1;
    }
    if(K<1){
        cout<<"ERROR en K_mean: Valor erroneo en K"<<endl;
        this->error=1;
        return 1;
    }
    if(attempts<1){
        cout<<"ERROR en K_mean: Valor erroneo en attempts"<<endl;
        this->error=1;
        return 1;
    }
    if(inicializacion != KMEANS_RANDOM_CENTERS && inicializacion!= KMEANS_PP_CENTERS && inicializacion !=KMEANS_USE_INITIAL_LABELS){
        cout<<"ERROR en K_mean: inicializacion erronea, debe estar dentre 0 y 2"<<endl;
        this->error=1;
        return 1;
    }
    Auxiliares ax;
    Mat Datos;
    e=ax.Image2Lexic(Data, Datos);
    if(e==1){
        cout<<"ERROR en K_mean: Error en Image2lexic"<<endl;
        this->error=1;
        return 1;
    }
    Mat lab;
    TermCriteria crit=TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001);
    kmeans(Datos, K, lab, crit, attempts, inicializacion, Centers);
    Labels.clear();
    for(int i=0; i<lab.rows; i++)
        Labels.push_back(lab.at<int>(i,0)+1);
    this->error=0;
    return 0;
}

int MLT::Clustering::Min_Max(vector<Mat> Data, float max_dist, vector<float> &Labels, Mat &Centers){
    int e=0;
    if(Data.size()==0){
        cout<<"ERROR en Min_Max: No hay datos"<<endl;
        this->error=1;
        return 1;
    }
    if(max_dist<0){
        cout<<"ERROR en Min_Max: Valor erroneo en max_dist"<<endl;
        this->error=1;
        return 1;
    }
    Auxiliares ax;
    Mat Datos;
    e=ax.Image2Lexic(Data, Datos);
    if(e==1){
        cout<<"ERROR en Min_Max: Error en Image2lexic"<<endl;
        this->error=1;
        return 1;
    }
    Mat dato=Mat::zeros(1,Datos.cols,CV_32FC1);
    vector<float> labels(Datos.rows);
    for(uint i=0; i<labels.size(); i++)
        labels[i]=-1;
    Datos.row(0).copyTo(dato);
    Mat mas_alejado=Mat::zeros(1,Datos.cols,CV_32FC1);
    bool cambio=true;
    int contador=0;
    while(cambio==true){
        float maxima=0;
        cambio=false;
        for(int j=0; j<Datos.rows; j++){
            if(labels[j]==-1){
                float suma=0;
                for(int k=0; k<Datos.cols; k++){
                    suma=suma+pow(Datos.at<float>(j,k)-dato.at<float>(0,k),2);
                }
                float distancia=sqrt(suma);
                if(distancia>max_dist){
                    cambio=true;
                    if(distancia>maxima){
                        maxima=distancia;
                        Datos.row(j).copyTo(mas_alejado.row(0));
                    }
                }
                else
                    labels[j]=contador;
            }
        }
        if(cambio==true){
            mas_alejado.row(0).copyTo(dato.row(0));
            contador++;
        }
    }
    Labels.clear();
    for(uint i=0; i<labels.size(); i++)
        Labels.push_back(labels[i]+1);
    bool neg;
    int num=ax.numero_etiquetas(Labels,neg);
    Centers=Mat::zeros(num,Datos.cols,CV_32FC1);
    if(neg==true){
        for(int i=0; i<num; i++){
            Mat suma=Mat::zeros(1,Datos.cols,CV_32FC1);
            for(uint j=0; j<Labels.size(); j++){
                if(i==0){
                    if(Labels[j]==-1){
                        suma=suma+Datos.row(j);
                    }
                }
                else{
                    if(Labels[j]==i){
                        suma=suma+Datos.row(j);
                    }
                }
            }
            Centers.row(i)=suma.row(0)/(float)Labels.size();
        }
    }
    else{
        for(int i=1; i<num+1; i++){
            Mat suma=Mat::zeros(1,Datos.cols,CV_32FC1);
            for(uint j=0; j<Labels.size(); j++){
                if(Labels[j]==i){
                    suma=suma+Datos.row(j);
                }
            }
            Centers.row(i-1)=suma/(float)Labels.size();
        }
    }
    this->error=0;
    return 0;
}

int MLT::Clustering::Distancias_Encadenadas(vector<Mat> Data, float max_dist, vector<float> &Labels, Mat &Centers){
    int e=0;
    if(Data.size()==0){
        cout<<"ERROR en Distancias_Encadenadas: No hay datos"<<endl;
        this->error=1;
        return 1;
    }
    if(max_dist<0){
        cout<<"ERROR en Distancias_Encadenadas: Valor erroneo en max_dist"<<endl;
        this->error=1;
        return 1;
    }
    Auxiliares ax;
    Mat Datos;
    e=ax.Image2Lexic(Data, Datos);
    if(e==1){
        cout<<"ERROR en Distancias_Encadenadas: Error en Image2lexic"<<endl;
        this->error=1;
        return 1;
    }
    vector<float> distancias(Datos.rows);
    for(uint i=0; i<distancias.size(); i++)
        distancias[i]=0;
    vector<int> posicion_inicial(Datos.rows);
    for(uint i=0; i<posicion_inicial.size(); i++)
        posicion_inicial[i]=i;
    for(int i=1; i<Datos.rows; i++){
        float suma=0;
        for(int k=0; k<Datos.cols; k++){
            suma=suma+pow(Datos.at<float>(i,k)-Datos.at<float>(0,k),2);
        }
        distancias[i]=sqrt(suma);
    }
    bool cambio=true;
    while(cambio){
        cambio=false;
        for(uint i=0; i<distancias.size()-1; i++){
            if(distancias[i+1]<distancias[i]){
                cambio=true;
                float aux_dist;
                aux_dist=distancias[i];
                distancias[i]=distancias[i+1];
                distancias[i+1]=aux_dist;
                int aux_pos;
                aux_pos=posicion_inicial[i];
                posicion_inicial[i]=posicion_inicial[i+1];
                posicion_inicial[i+1]=aux_pos;
            }
        }
    }
    Labels.clear();
    for(uint i=0; i<distancias.size(); i++)
        Labels.push_back(0);
    int contador=1;
    Labels[posicion_inicial[0]]=1;
    for(uint i=1; i<distancias.size(); i++){
        if(abs(distancias[i]-distancias[i-1])>max_dist){
            contador++;
        }
        Labels[posicion_inicial[i]]=contador;
    }
    bool neg;
    int num=ax.numero_etiquetas(Labels,neg);
    Centers=Mat::zeros(num,Datos.cols,CV_32FC1);
    if(neg==true){
        for(int i=0; i<num; i++){
            Mat suma=Mat::zeros(1,Datos.cols,CV_32FC1);
            for(uint j=0; j<Labels.size(); j++){
                if(i==0){
                    if(Labels[j]==-1){
                        suma=suma+Datos.row(j);
                    }
                }
                else{
                    if(Labels[j]==i){
                        suma=suma+Datos.row(j);
                    }
                }
            }
            Centers.row(i)=suma.row(0)/(float)Labels.size();
        }
    }
    else{
        for(int i=1; i<num+1; i++){
            Mat suma=Mat::zeros(1,Datos.cols,CV_32FC1);
            for(uint j=0; j<Labels.size(); j++){
                if(Labels[j]==i){
                    suma=suma+Datos.row(j);
                }
            }
            Centers.row(i-1)=suma/(float)Labels.size();
        }
    }
    this->error=0;
    return 0;
}

int MLT::Clustering::Histograma(vector<Mat> Data, float tam_celda, vector<float> &Labels, Mat &Centers){
    int e=0;
    if(Data.size()==0){
        cout<<"ERROR en Histograma: No hay datos"<<endl;
        this->error=1;
        return 1;
    }
    if(tam_celda<0){
        cout<<"ERROR en Histograma: Valor erroneo en tam_celda"<<endl;
        this->error=1;
        return 1;
    }
    Auxiliares ax;
    Mat Datos;
    e=ax.Image2Lexic(Data, Datos);
    if(e==1){
        cout<<"ERROR en Histograma: Error en Image2lexic"<<endl;
        this->error=1;
        return 1;
    }
    Mat celdas=Mat::zeros(Datos.rows,Datos.cols,CV_32FC1);
    Mat min_celda,max_celda;
    min_celda=Mat::zeros(1,Datos.cols,CV_32FC1)+999999999;
    max_celda=Mat::zeros(1,Datos.cols,CV_32FC1)-999999999;
    for(int i=0; i<Datos.rows; i++){
        for(int j=0; j<Datos.cols; j++){
            float pos=floor(Datos.at<float>(i,j)/tam_celda);
            celdas.at<float>(i,j)=pos;
            if(pos<min_celda.at<float>(0,j))
                min_celda.at<float>(0,j)=pos;
            if(pos>max_celda.at<float>(0,j))
                max_celda.at<float>(0,j)=pos;
        }
    }
    max_celda.at<float>(0,0)=max_celda.at<float>(0,0)+1;
    Mat celda_actual;
    min_celda.copyTo(celda_actual);
    vector<Mat> celdas_usadas;
    vector<int> contadores;
    while(cv::countNonZero(max_celda!=celda_actual)!=0){
        int contador=0;
        for(int i=0; i<celdas.rows; i++){
            if(cv::sum(cv::abs(celdas.row(i)-celda_actual))[0]==0){
                contador++;
            }
        }
        if(contador>0){
            Mat celda_a_vector;
            celda_actual.copyTo(celda_a_vector);
            celdas_usadas.push_back(celda_a_vector);
            contadores.push_back(contador);
        }
        celda_actual.at<float>(0,0)=celda_actual.at<float>(0,0)+1;
        for(int i=0; i<(celda_actual.cols-1); i++){
            if(celda_actual.at<float>(0,i)>max_celda.at<float>(0,i)){
                celda_actual.at<float>(0,i)=min_celda.at<float>(0,i);
                celda_actual.at<float>(0,i+1)=celda_actual.at<float>(0,i+1)+1;
            }
        }
    }
    vector<bool> etiquetados;
    vector<float> etiquetas;
    for(uint i=0; i<contadores.size(); i++){
        etiquetados.push_back(false);
        etiquetas.push_back(0);
    }
    int cuenta_etiqueta=1;
    bool cambio=true;
    while(cambio){
        int maximo=0;
        int pos=0;
        for(uint i=0; i<contadores.size(); i++){
            if(contadores[i]>maximo && etiquetados[i]==false){
                maximo=contadores[i];
                pos=i;
            }
        }
        vector<int> por_comprobar;
        por_comprobar.push_back(pos);
        while(por_comprobar.size()!=0){
            pos=por_comprobar[0];
            por_comprobar.erase(por_comprobar.begin());
            if(etiquetados[pos]==false){
                celdas_usadas[pos].copyTo(min_celda);
                celdas_usadas[pos].copyTo(max_celda);
                min_celda=min_celda-1;
                max_celda=max_celda+1;
                max_celda.at<float>(0,0)=max_celda.at<float>(0,0)+1;
                min_celda.copyTo(celda_actual);
                etiquetas[pos]=cuenta_etiqueta;
                etiquetados[pos]=true;
                while(cv::countNonZero(max_celda!=celda_actual)!=0){
                    for(uint i=0; i<celdas_usadas.size(); i++){
                        if(etiquetados[i]==false){
                            if(cv::sum(abs(celdas_usadas[i]-celda_actual))[0]==0){
                                por_comprobar.push_back(i);
                            }
                        }
                    }
                    celda_actual.at<float>(0,0)=celda_actual.at<float>(0,0)+1;
                    for(int i=0; i<(celda_actual.cols-1); i++){
                        if(celda_actual.at<float>(0,i)>max_celda.at<float>(0,i)){
                            celda_actual.at<float>(0,i)=min_celda.at<float>(0,i);
                            celda_actual.at<float>(0,i+1)=celda_actual.at<float>(0,i+1)+1;
                        }
                    }
                }
            }
        }
        cuenta_etiqueta++;
        cambio=false;
        for(uint i=0; i<etiquetados.size(); i++){
            if(etiquetados[i]==false)
                cambio=true;
        }
    }
    Labels.clear();
    for(int i=0; i<Datos.rows; i++)
        Labels.push_back(0);
    for(int i=0; i<celdas.rows; i++){
        for(uint j=0; j<celdas_usadas.size(); j++){
            if(cv::countNonZero(celdas.row(i)!=celdas_usadas[j])==0)
                Labels[i]=etiquetas[j];
        }
    }
    bool neg;
    int num=ax.numero_etiquetas(Labels,neg);
    Centers=Mat::zeros(num,Datos.cols,CV_32FC1);
    if(neg==true){
        for(int i=0; i<num; i++){
            Mat suma=Mat::zeros(1,Datos.cols,CV_32FC1);
            for(uint j=0; j<Labels.size(); j++){
                if(i==0){
                    if(Labels[j]==-1){
                        suma=suma+Datos.row(j);
                    }
                }
                else{
                    if(Labels[j]==i){
                        suma=suma+Datos.row(j);
                    }
                }
            }
            Centers.row(i)=suma.row(0)/(float)Labels.size();
        }
    }
    else{
        for(int i=1; i<num+1; i++){
            Mat suma=Mat::zeros(1,Datos.cols,CV_32FC1);
            for(uint j=0; j<Labels.size(); j++){
                if(Labels[j]==i){
                    suma=suma+Datos.row(j);
                }
            }
            Centers.row(i-1)=suma/(float)Labels.size();
        }
    }
    this->error=0;
    return 0;
}

int MLT::Clustering::EXP_MAX(vector<Mat> Data, vector<float> &Labels, Mat &Centers, int nclusters, int covMatType){
//Parameters:
//nclusters – The number of mixture components in the Gaussian mixture model. Default value of the parameter is EM::DEFAULT_NCLUSTERS=5. Some of EM implementation could determine the optimal number of mixtures within a specified value range, but that is not the case in ML yet.
//covMatType –
//Constraint on covariance matrices which defines type of matrices. Possible values are:

//EM::COV_MAT_SPHERICAL A scaled identity matrix \mu_k * I. There is the only parameter \mu_k to be estimated for each matrix. The option may be used in special cases, when the constraint is relevant, or as a first step in the optimization (for example in case when the data is preprocessed with PCA). The results of such preliminary estimation may be passed again to the optimization procedure, this time with covMatType=EM::COV_MAT_DIAGONAL.
//EM::COV_MAT_DIAGONAL A diagonal matrix with positive diagonal elements. The number of free parameters is d for each matrix. This is most commonly used option yielding good estimation results.
//EM::COV_MAT_GENERIC A symmetric positively defined matrix. The number of free parameters in each matrix is about d^2/2. It is not recommended to use this option, unless there is pretty accurate initial estimation of the parameters and/or a huge number of training samples.
//termCrit – The termination criteria of the EM algorithm. The EM algorithm can be terminated by the number of iterations termCrit.maxCount (number of M-steps) or when relative change of likelihood logarithm is less than termCrit.epsilon. Default maximum number of iterations is EM::DEFAULT_MAX_ITERS=100.
    int e=0;
    if(Data.size()==0){
        cout<<"ERROR en EXP_MAX: No hay datos"<<endl;
        this->error=1;
        return 1;
    }
    if(nclusters<1){
        cout<<"ERROR en EXP_MAX: Valor erroneo en nclusters"<<endl;
        this->error=1;
        return 1;
    }
    Auxiliares ax;
    Mat Datos;
    e=ax.Image2Lexic(Data, Datos);
    if(e==1){
        cout<<"ERROR en EXP_MAX: Error en Image2lexic"<<endl;
        this->error=1;
        return 1;
    }
    cv::Ptr<cv::ml::EM> em=cv::ml::EM::create();
    em->setClustersNumber(nclusters);
    em->setCovarianceMatrixType(covMatType);
    em->trainEM(Datos);
    for(int i=0; i<Datos.rows; i++){
        int resultado = cvRound(em->predict2( Datos.row(i), noArray() )[1]);
        Labels.push_back((float)resultado+1);
    }
    bool neg;
    int num=ax.numero_etiquetas(Labels,neg);
    Centers=Mat::zeros(num,Datos.cols,CV_32FC1);
    if(neg==true){
        for(int i=0; i<num; i++){
            Mat suma=Mat::zeros(1,Datos.cols,CV_32FC1);
            for(uint j=0; j<Labels.size(); j++){
                if(i==0){
                    if(Labels[j]==-1){
                        suma=suma+Datos.row(j);
                    }
                }
                else{
                    if(Labels[j]==i){
                        suma=suma+Datos.row(j);
                    }
                }
            }
            Centers.row(i)=suma.row(0)/(float)Labels.size();
        }
    }
    else{
        for(int i=1; i<num+1; i++){
            Mat suma=Mat::zeros(1,Datos.cols,CV_32FC1);
            for(uint j=0; j<Labels.size(); j++){
                if(Labels[j]==i){
                    suma=suma+Datos.row(j);
                }
            }
            Centers.row(i-1)=suma/(float)Labels.size();
        }
    }
    this->error=0;
    return 0;
}
