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

#include "clasificador_neuronal.h"

MLT::Clasificador_Neuronal::Clasificador_Neuronal(string Nombre, Mat layerSize, int Method, int Function, double bp_dw_scale, double bp_moment_scale, double rp_dw0, double rp_dw_max, double rp_dw_min, double rp_dw_minus, double rp_dw_plus, double fparam1, double fparam2){

//Parameters:
//train_method –

//CvANN_MLP_TrainParams::BACKPROP The back-propagation algorithm.
//CvANN_MLP_TrainParams::RPROP The RPROP algorithm.
//term_crit – Termination criteria of the training algorithm. You can specify the maximum number of iterations (max_iter) and/or how much the error could change between the iterations to make the algorithm continue (epsilon).

//The back-propagation algorithm parameters:
//double bp_dw_scale Strength of the weight gradient term. The recommended value is about 0.1.
//double bp_moment_scale Strength of the momentum term (the difference between weights on the 2 previous iterations). This parameter provides some inertia to smooth the random fluctuations of the weights. It can vary from 0 (the feature is disabled) to 1 and beyond. The value 0.1 or so is good enough

//The RPROP algorithm parameters (see [RPROP93] for details):
//double rp_dw0 Initial value \Delta_0 of update-values \Delta_{ij}.
//double rp_dw_plus Increase factor \eta^+. It must be >1.
//double rp_dw_minus Decrease factor \eta^-. It must be <1.
//double rp_dw_min Update-values lower limit \Delta_{min}. It must be positive.
//double rp_dw_max Update-values upper limit \Delta_{max}. It must be >1.

//layerSizes – Integer vector specifying the number of neurons in each layer including the input and output layers.
//activateFunc – Parameter specifying the activation function for each neuron: one of
//CvANN_MLP::IDENTITY
//CvANN_MLP::SIGMOID_SYM
//CvANN_MLP::GAUSSIAN
//fparam1 – Free parameter of the activation function, \alpha. See the formulas in the introduction section.
//fparam2 – Free parameter of the activation function, \beta. See the formulas in the introduction section.

    if(!layerSize.empty())
        Parametrizar(layerSize, Method, Function, bp_dw_scale, bp_moment_scale, rp_dw0,
                     rp_dw_max, rp_dw_min, rp_dw_minus, rp_dw_plus,
                     fparam1, fparam2);
    nombre=Nombre;
    tipo_clasificador=NEURONAL;
}

MLT::Clasificador_Neuronal::~Clasificador_Neuronal(){}

int MLT::Clasificador_Neuronal::Parametrizar(cv::Mat layerSize, int Method, int Function, double bp_dw_scale, double bp_moment_scale, double rp_dw0,
                                        double rp_dw_max, double rp_dw_min, double rp_dw_minus, double rp_dw_plus,
                                        double fparam1, double fparam2){
    MLP=ml::ANN_MLP::create();
    MLP->setTrainMethod(Method);
    MLP->setBackpropWeightScale(bp_dw_scale);
    MLP->setBackpropMomentumScale(bp_moment_scale);
    MLP->setRpropDW0(rp_dw0);
    MLP->setRpropDWMax(rp_dw_max);
    MLP->setRpropDWMin(rp_dw_min);
    MLP->setRpropDWMinus(rp_dw_minus);
    MLP->setRpropDWPlus(rp_dw_plus);
    MLP->setTermCriteria(cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.01 ));
    MLP->setLayerSizes(layerSize);
    MLP->setActivationFunction(Function,fparam1,fparam2);
    layerSize.copyTo(Layers);
    MLP->create();
    return 0;
}

int MLT::Clasificador_Neuronal::Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save){
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
    if(Layers.empty()){
        cout<<"ERROR en Autotrain: No se ha inicializado la matriz layerSize"<<endl;
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

int MLT::Clasificador_Neuronal::Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read){
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

void MLT::Clasificador_Neuronal::Entrenamiento(Mat trainingDataMat, Mat labelsMat){
    labelsMat.convertTo(labelsMat,CV_32FC1);
    Mat labels=Mat::zeros(labelsMat.rows,numero_etiquetas,CV_32FC1);
    for(int i=0; i<labelsMat.rows; i++){
        float etiqueta=labelsMat.at<float>(i,0);
        int pos=-1;
        if(negativa && etiqueta==-1)
            pos=0;
        else if(negativa && etiqueta>0)
            pos=etiqueta;
        else
            pos=etiqueta-1;
        labels.at<float>(i,pos)=1.;
    }
    trainingDataMat.convertTo(trainingDataMat,CV_32FC1);
    MLP->train(trainingDataMat, ml::ROW_SAMPLE, labels);
}

float MLT::Clasificador_Neuronal::Clasificacion(Mat Data){
    Data.convertTo(Data,CV_32FC1);
    float response=0;
    cv :: Mat resp;
    if(Data.cols==(ventana_x*ventana_y) || Data.cols==reduccion.tam_reduc){
        response=MLP->predict(Data,resp);
        if(negativa && response==0)
            response=-1;
        else if(!negativa)
            response=response+1;
    }
     return response;
}

int MLT::Clasificador_Neuronal::Save_Data(){
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
    string g="../Data/Configuracion/"+nombre+"/NEURONAL2.xml";
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
    g="../Data/Configuracion/"+nombre+"/NEURONAL.xml";
    MLP->save(g.c_str());
    g="../Data/Configuracion/"+nombre+"/Clasificador.xml";
    cv::FileStorage clas(g,CV_STORAGE_WRITE);
    if(clas.isOpened()){
        int id=NEURONAL;
        clas<<"Tipo"<<id;
    }
    else
        return 1;
    clas.release();
    return 0;
}

int MLT::Clasificador_Neuronal::Read_Data(){
    string g="../Data/Configuracion/"+nombre+"/NEURONAL2.xml";
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
    g="../Data/Configuracion/"+nombre+"/NEURONAL.xml";
//    MLP->load(g.c_str());
    MLP = ml::StatModel::load<ml::ANN_MLP>(g.c_str());
    return 0;
}
