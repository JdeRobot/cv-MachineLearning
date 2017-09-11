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

#include "clasificador_svm.h"


MLT::Clasificador_SVM::Clasificador_SVM(string Nombre,int train, int Type, int kernel_type, Mat class_weights, double degree, double gamma, double coef0, double C, double nu, double p){

//Parameters:
//svm_type –
//Type of a SVM formulation. Possible values are:
//C_SVC=100, NU_SVC=101, ONE_CLASS=102, EPS_SVR=103, NU_SVR=104
//-CvSVM::C_SVC C-Support Vector Classification. n-class classification (n \geq 2), allows imperfect separation of classes with penalty multiplier C for outliers.
//-CvSVM::NU_SVC \nu-Support Vector Classification. n-class classification with possible imperfect separation. Parameter \nu (in the range 0..1, the larger the value, the smoother the decision boundary) is used instead of C.
//-CvSVM::ONE_CLASS Distribution Estimation (One-class SVM). All the training data are from the same class, SVM builds a boundary that separates the class from the rest of the feature space.
//-CvSVM::EPS_SVR \epsilon-Support Vector Regression. The distance between feature vectors from the training set and the fitting hyper-plane must be less than p. For outliers the penalty multiplier C is used.
//-CvSVM::NU_SVR \nu-Support Vector Regression. \nu is used instead of p.

//kernel_type –
//Type of a SVM kernel. Possible values are:
//LINEAR=0, POLY=1, RBF=2, SIGMOID=3
//-CvSVM::LINEAR Linear kernel. No mapping is done, linear discrimination (or regression) is done in the original feature space. It is the fastest option. K(x_i, x_j) = x_i^T x_j.
//-CvSVM::POLY Polynomial kernel: K(x_i, x_j) = (\gamma x_i^T x_j + coef0)^{degree}, \gamma > 0.
//-CvSVM::RBF Radial basis function (RBF), a good choice in most cases. K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}, \gamma > 0.
//-CvSVM::SIGMOID Sigmoid kernel: K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0).

//degree – Parameter degree of a kernel function (POLY).
//gamma – Parameter \gamma of a kernel function (POLY / RBF / SIGMOID).
//coef0 – Parameter coef0 of a kernel function (POLY / SIGMOID).
//Cvalue – Parameter C of a SVM optimization problem (C_SVC / EPS_SVR / NU_SVR).
//nu – Parameter \nu of a SVM optimization problem (NU_SVC / ONE_CLASS / NU_SVR).
//p – Parameter \epsilon of a SVM optimization problem (EPS_SVR).
//class_weights – Optional weights in the C_SVC problem , assigned to particular classes. They are multiplied by C so the parameter C of class #i becomes class\_weights_i * C. Thus these weights affect the misclassification penalty for different classes. The larger weight, the larger penalty on misclassification of data from the corresponding class.
//term_crit – Termination criteria of the iterative SVM training procedure which solves a partial case of constrained quadratic optimization problem. You can specify tolerance and/or the maximum number of iterations.
//train - Tipo de entrenamiento

    Parametrizar(train,Type,kernel_type,class_weights,degree,gamma,coef0,C,nu,p);
    nombre=Nombre;
    tipo_clasificador=C_SVM;
}

MLT::Clasificador_SVM::~Clasificador_SVM(){}

int MLT::Clasificador_SVM::Parametrizar(int train, int Type, int kernel_type, Mat class_weights, double degree, double gamma, double coef0, double C, double nu, double p){
    SVM=ml::SVM::create();
    SVM->setType(Type);
    SVM->setKernel(kernel_type);
    SVM->setDegree(degree);
    SVM->setGamma(gamma);
    SVM->setCoef0(coef0);
    SVM->setC(C);
    SVM->setNu(nu);
    SVM->setP(p);
    SVM->setClassWeights(class_weights);
    SVM->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON));
    tipo_entrena=train;
    return 0;
}

int MLT::Clasificador_SVM::Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save){
    if(Data.size()==0){
        cout<<"ERROR en Autotrain: No hay datos"<<endl;
        return 1;
    }
    if(Labels.size()==0){
        cout<<"ERROR en Autotrain: No hay Etiquetas"<<endl;
        return 1;
    }
    if(Data.size()!=Labels.size()){
        cout<<"ERROR en Autotrain: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
    if((reduc.si_dist&&(reduc.si_pca||reduc.si_lda||reduc.si_d_prime))||
            (reduc.si_d_prime&&(reduc.si_pca||reduc.si_lda||reduc.si_dist))||
            (reduc.si_pca&&(reduc.si_dist||reduc.si_lda||reduc.si_d_prime))||
            (reduc.si_lda&&(reduc.si_pca||reduc.si_dist||reduc.si_d_prime))){
        cout<<"ERROR en Autotrain: Solo puede haber un metodo de reduccion de dimensionalidad activado"<<endl;
        return 1;
    }
    if((reduc.si_lda || reduc.si_pca || reduc.si_dist || reduc.si_d_prime) && reduc.tam_reduc<=0){
        cout<<"ERROR en Autotrain: si_lda=true o si_pca=true o si_dist=true o si_d_prime=true pero t_reduc es igual o menor a 0"<<endl;
        return 1;
    }
    ventana_o_x=info.Tam_Orig_X;
    ventana_o_y=info.Tam_Orig_Y;
    ventana_x=info.Tam_X;
    ventana_y=info.Tam_Y;
    tipo_dato=info.Tipo_Datos;
    if((reduc.si_dist==true || reduc.si_d_prime==true || reduc.si_lda==true || reduc.si_pca==true)&&(info.si_dist==true || info.si_d_prime==true || info.si_lda==true || info.si_pca==true)){
        cout<<"ERROR en Autotrain: Ya se le ha hecho una reduccion anteriormente a los datos"<<endl;
        return 1;
    }
    reduccion=reduc;
    Auxiliares ax;
    bool negativa;
    numero_etiquetas=ax.numero_etiquetas(Labels,negativa);
    Mat lexic_data;
    int e=ax.Image2Lexic(Data,lexic_data);
    if(e==1){
        cout<<"ERROR en Autorain: Error en Image2Lexic"<<endl;
        return 1;
    }
    Mat lexic_labels(Labels.size(), 1, CV_32FC1, Labels.data());
    lexic_data.convertTo(lexic_data,CV_32FC1);
    lexic_labels.convertTo(lexic_labels,CV_32FC1);
    Mat trainingDataMat;
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
        Proyectada.copyTo(trainingDataMat);
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
        Proyectada.copyTo(trainingDataMat);
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
        Proyectada.copyTo(trainingDataMat);
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
        Proyectada.copyTo(trainingDataMat);
    }
    else
        lexic_data.copyTo(trainingDataMat);
    Entrenamiento(trainingDataMat, lexic_labels);
    if(reduc.si_dist==false && reduc.si_d_prime==false && reduc.si_lda==false && reduc.si_pca==false){
        reduccion.si_dist=info.si_dist;
        reduccion.si_d_prime=info.si_d_prime;
        reduccion.si_lda=info.si_lda;
        reduccion.si_pca=info.si_pca;
        reduccion.DS=info.DS;
        reduccion.D_PRIME=info.D_PRIME;
        reduccion.LDA=info.LDA;
        reduccion.PCA=info.PCA;
        reduccion.tam_reduc=info.Tam_X*info.Tam_Y;
    }
    if(save){
        e=Save_Data();
        if(e==1){
            cout<<"ERROR en Autotrain: Error en Save_Data"<<endl;
            return 1;
        }
    }
    return 0;
}

int MLT::Clasificador_SVM::Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read){
    int e=0;
    if(read){
        e=Read_Data();
        if(e==1){
            cout<<"ERROR en Autoclasificacion: Error en Read_Data"<<endl;
            return 1;
        }
    }
    Auxiliares ax;
    Mat lexic_data;
    e=ax.Image2Lexic(Data,lexic_data);
    if(e==1){
        cout<<"ERROR en Autoclasificacion: Error en Image2Lexic"<<endl;
        return 1;
    }
    Mat trainingDataMat;
    if(reducir){
        if(reduccion.si_lda){
            Dimensionalidad dim(nombre);
            Mat Proyectada;
            e=dim.Proyeccion(lexic_data,Proyectada,LDA_DIM,reduccion.LDA);
            if(e==1){
                cout<<"ERROR en Autoclasificacion: Error en Proyeccion"<<endl;
                return 1;
            }
            Proyectada.copyTo(trainingDataMat);
        }
        else if(reduccion.si_pca){
            Dimensionalidad dim(nombre);
            Mat Proyectada;
            e=dim.Proyeccion(lexic_data,Proyectada,PCA_DIM,reduccion.PCA);
            if(e==1){
                cout<<"ERROR en Autoclasificacion: Error en Proyeccion"<<endl;
                return 1;
            }
            Proyectada.copyTo(trainingDataMat);
        }
        else if(reduccion.si_dist){
            Dimensionalidad dim(nombre);
            Mat Proyectada;
            e=dim.Proyeccion(lexic_data,Proyectada,MAXDIST_DIM,reduccion.DS);
            if(e==1){
                cout<<"ERROR en Autoclasificacion: Error en Proyeccion"<<endl;
                return 1;
            }
            Proyectada.copyTo(trainingDataMat);
        }
        else if(reduccion.si_d_prime){
            Dimensionalidad dim(nombre);
            Mat Proyectada;
            e=dim.Proyeccion(lexic_data,Proyectada,D_PRIME_DIM,reduccion.D_PRIME);
            if(e==1){
                cout<<"ERROR en Autoclasificacion: Error en Proyeccion"<<endl;
                return 1;
            }
            Proyectada.copyTo(trainingDataMat);
        }
        else
            lexic_data.copyTo(trainingDataMat);
    }
    else
        lexic_data.copyTo(trainingDataMat);
    for(int i=0; i<trainingDataMat.rows; i++){
        float response=Clasificacion(trainingDataMat.row(i));
        Labels.push_back(response);
#ifdef GUI
            progreso++;
            window->progress_Clasificar->setValue(base_progreso+(max_progreso*progreso/total_progreso));
#endif
    }
    return 0;
}

void MLT::Clasificador_SVM::Entrenamiento(Mat trainingDataMat, Mat labelsMat){
    trainingDataMat.convertTo(trainingDataMat,CV_32FC1);
    labelsMat.convertTo(labelsMat,CV_32S);
    if(tipo_entrena==TRAIN)
        SVM->train(trainingDataMat, ml::ROW_SAMPLE, labelsMat.t());
    if(tipo_entrena==TRAIN_AUTO){
        Ptr<ml::TrainData> data=ml::TrainData::create(trainingDataMat,ml::ROW_SAMPLE,labelsMat);
        SVM->trainAuto(data);
    }
}

float MLT::Clasificador_SVM::Clasificacion(Mat Data){
    Data.convertTo(Data,CV_32FC1);
    float response=0;
    if(Data.cols==(ventana_x*ventana_y) || Data.cols==reduccion.tam_reduc)
        response = SVM->predict(Data);
    if(response==0)
        response=-1;
    return response;
}


int MLT::Clasificador_SVM::Save_Data(){
    DIR    *dir_p = opendir ("../Data/Configuracion");
    if(dir_p == NULL) {
        string command = "mkdir ../Data/Configuracion";
        int er=system(command.c_str());
        if(er!=0){
            cout<<"ERROR en Read_Data: Error al crear carpeta"<<endl;
            return 1;
        }
    }
    String dir="../Data/Configuracion/"+nombre;
    DIR    *dir_p2 = opendir (dir.c_str());
    if(dir_p2 == NULL) {
        string command = "mkdir "+dir;
        int er=system(command.c_str());
        if(er!=0){
            cout<<"ERROR en Read_Data: Error al crear carpeta"<<endl;
            return 1;
        }
    }
    string g="../Data/Configuracion/"+nombre+"/SVM2.xml";
    cv::FileStorage archivo_w(g,CV_STORAGE_WRITE);
    if(archivo_w.isOpened()){
        archivo_w<<"ventana_x"<<ventana_x;
        archivo_w<<"ventana_y"<<ventana_y;
        archivo_w<<"ventana_o_x"<<ventana_o_x;
        archivo_w<<"ventana_o_y"<<ventana_o_y;
        archivo_w<<"numero_etiquetas"<<numero_etiquetas;
        archivo_w<<"tipo_dato"<<tipo_dato;
        archivo_w<<"tam_reduc"<<reduccion.tam_reduc;
        archivo_w<<"lda"<<reduccion.si_lda;
        archivo_w<<"LDA"<<reduccion.LDA;
        archivo_w<<"pca"<<reduccion.si_pca;
        archivo_w<<"Pca"<<reduccion.PCA;
        archivo_w<<"dist"<<reduccion.si_dist;
        archivo_w<<"DS"<<reduccion.DS;
    }
    else
        return 1;
    archivo_w.release();
    g="../Data/Configuracion/"+nombre+"/SVM.xml";
    SVM->save(g.c_str());
    g="../Data/Configuracion/"+nombre+"/Clasificador.xml";
    cv::FileStorage clas(g,CV_STORAGE_WRITE);
    if(clas.isOpened()){
        int id=C_SVM;
        clas<<"Tipo"<<id;
    }
    else
        return 1;
    clas.release();
    return 0;
}

int MLT::Clasificador_SVM::Read_Data(){
    string g="../Data/Configuracion/"+nombre+"/SVM2.xml";
    cv::FileStorage archivo_r(g,CV_STORAGE_READ);
    if(archivo_r.isOpened()){
        archivo_r["ventana_x"]>>ventana_x;
        archivo_r["ventana_y"]>>ventana_y;
        archivo_r["ventana_o_x"]>>ventana_o_x;
        archivo_r["ventana_o_y"]>>ventana_o_y;
        archivo_r["numero_etiquetas"]>>numero_etiquetas;
        archivo_r["tipo_dato"]>>tipo_dato;
        archivo_r["tam_reduc"]>>reduccion.tam_reduc;
        archivo_r["lda"]>>reduccion.si_lda;
        archivo_r["LDA"]>>reduccion.LDA;
        archivo_r["pca"]>>reduccion.si_pca;
        archivo_r["Pca"]>>reduccion.PCA;
        archivo_r["dist"]>>reduccion.si_dist;
        archivo_r["DS"]>>reduccion.DS;
    }
    else
        return 1;
    archivo_r.release();
    g="../Data/Configuracion/"+nombre+"/SVM.xml";
//    SVM->load(g.c_str());
    SVM = ml::StatModel::load<ml::SVM>(g.c_str());
    return 0;
}
