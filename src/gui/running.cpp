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

#include "running.h"

MLT::Running::Running(){}

void MLT::Running::update_gen(){
    this->window->v_progress_datamanaging->setValue(this->base_progreso+(this->max_progreso*this->gen.progreso/this->gen.total_progreso));
    this->window->i_progress_datamanaging->setValue(this->base_progreso+(this->max_progreso*this->gen.progreso/this->gen.total_progreso));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void MLT::Running::update_analysis(){
    this->window->v_progress_Analysis->setValue(this->ana.progreso);
//    this->window->i_progress_Analysis->setValue(this->ana->progreso);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

int MLT::Running::load_dataset(QString path, string &ref, std::vector<float> &labels, std::vector<cv::Mat> &images){
    std::string dir=path.toStdString();
    int pos=0;
    for(uint i=0; i<dir.size(); i++){
        if(dir[i]=='/')
            pos=i;
    }
    for(uint i=pos+1; i<dir.size(); i++)
        ref=ref+dir[i];
    string archivo_i=dir+"/Info.xml";
    cv::FileStorage Archivo_i(archivo_i,CV_STORAGE_READ);

    if(!Archivo_i.isOpened())
        return 1;

    int num;
    Archivo_i["Num_Datos"]>>num;
    Archivo_i.release();

    string input_directory=dir+"/Recortes.txt";
    Generacion::Info_Datos info;

    this->base_progreso=1;
    this->max_progreso=100;
    std::thread thrd(&MLT::Generacion::Cargar_Fichero,&gen,input_directory,std::ref(images),std::ref(labels),std::ref(info));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(gen.running==true)
        update_gen();

    thrd.join();

    if(gen.error==1)
        return 2;

    return 0;
}

int MLT::Running::synthetic_data(QString nombre, int num_clases, int num_data_clase, int vector_size, float ancho, float separacion_clases, vector<Mat> &data, vector<float> &labels){
    Size size_img;
    size_img.width=vector_size;
    size_img.height=1;

    std::string name=nombre.toStdString();
    this->base_progreso=1;
    this->max_progreso=100;

    Generacion::Info_Datos info;

    std::thread thrd(&MLT::Generacion::Random_Synthetic_Data,&gen, name, num_clases, num_data_clase, size_img, ancho, separacion_clases, std::ref(data), std::ref(labels),std::ref(info), this->save_data);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(gen.running==true)
        update_gen();

    thrd.join();

    if(this->gen.error==1)
        return 1;

}

int MLT::Running::analyse(vector<Mat> images, vector<float> labels, QStandardItemModel *model){
    vector<Mat> means,  std, covariance;
    vector<vector<Mat> > d_prime;
    bool negative;
    vector<int> number;

    std::thread thrd(&MLT::Analisis::Estadisticos_Covarianzas,&ana, images, labels, std::ref(means),std::ref(std), std::ref(d_prime), std::ref(covariance), std::ref(negative), std::ref(number));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(this->ana.running==true)
        update_analysis();

    thrd.join();


    if(this->ana.error==1)
        return 1;

    if(images[0].cols*images[0].rows*images[0].channels()<=1024){
        this->max_progreso=number.size()*(means[0].cols+std[0].cols+d_prime.size()*d_prime[0].size()+covariance[0].rows*covariance[0].cols);
    }
    else
        this->max_progreso=number.size()*(means[0].cols+std[0].cols+d_prime.size()*d_prime[0].size());

    this->window->v_progress_Analysis->setValue(30);
    int progreso=0;
    for(int i=0; i<number.size(); i++){
        int etiqueta;
        if(negative){
            if(i==0)
                etiqueta=-1;
            else
                etiqueta=i;
        }
        else
            etiqueta=i+1;
        QStandardItem *Lab = new QStandardItem(QString("Label %1").arg(etiqueta));
        QStandardItem *Num_Datos = new QStandardItem(QString("amount of data"));
        QStandardItem *Num = new QStandardItem(QString("%1").arg(number[i]));
        Num_Datos->appendRow(Num);
        Lab->appendRow(Num_Datos);
        QStandardItem *Dimensiones = new QStandardItem(QString("Dimensions"));
        QStandardItem *Dim = new QStandardItem(QString("%1").arg(images.size()));
        Dimensiones->appendRow(Dim);
        Lab->appendRow(Dimensiones);
        QStandardItem *Medias = new QStandardItem(QString("Mean"));
        for(int j=0; j<means[i].cols; j++){
            stringstream media;
            media<<fixed<<means[i].at<float>(0,j);
            QString valor=QString::fromStdString(media.str());
            QStandardItem *Media = new QStandardItem(QString(valor));
            Medias->appendRow(Media);
            progreso++;
            this->window->v_progress_Analysis->setValue(30+(70*progreso/this->max_progreso));
        }
        Lab->appendRow(Medias);
        QStandardItem *Des = new QStandardItem(QString("Std"));
        for(int j=0; j<std[i].cols; j++){
            stringstream desviacion;
            desviacion<<fixed<<std[i].at<float>(0,j);
            QString valor=QString::fromStdString(desviacion.str());
            QStandardItem *desvi = new QStandardItem(QString(valor));
            Des->appendRow(desvi);
            progreso++;
            this->window->v_progress_Analysis->setValue(30+(70*progreso/this->max_progreso));
        }
        Lab->appendRow(Des);
        QStandardItem *DPrime = new QStandardItem(QString("D-Prime"));
        for(int j=0; j<number.size(); j++){
            int etiqueta2;
            if(negative){
                if(j==0)
                    etiqueta2=-1;
                else
                    etiqueta2=j;
            }
            else
                etiqueta2=j+1;
            if(etiqueta!=etiqueta2){
                QStandardItem *Etiqueta_Etiqueta = new QStandardItem(QString("Label %1").arg(etiqueta2));
                for(int k=0; k<d_prime[i][j].cols; k++){
                    stringstream dprime;
                    dprime<<fixed<<d_prime[i][j].at<float>(0,k);
                    QString valor=QString::fromStdString(dprime.str());
                    QStandardItem *Dprime = new QStandardItem(QString(valor));
                    Etiqueta_Etiqueta->appendRow(Dprime);
                }
                DPrime->appendRow(Etiqueta_Etiqueta);
                progreso++;
                this->window->v_progress_Analysis->setValue(30+(70*progreso/this->max_progreso));
            }
        }
        Lab->appendRow(DPrime);
        if(images[0].cols*images[0].rows*images[0].channels()<=1024){
            QStandardItem *Cov = new QStandardItem(QString("Covariance"));
            for(int j=0; j<covariance[i].rows; j++){
                stringstream linea_covarianza;
                for(int k=0; k<covariance[i].cols; k++){
                    linea_covarianza<<fixed<<covariance[i].at<float>(j,k);
                    linea_covarianza<<" ";
                    progreso++;
                    this->window->v_progress_Analysis->setValue(30+(70*progreso/this->max_progreso));
                }
                linea_covarianza<<";\n";
                QString valor=QString::fromStdString(linea_covarianza.str());
                QStandardItem *covarianza = new QStandardItem(QString(valor));
                Cov->appendRow(covarianza);
            }
            Lab->appendRow(Cov);
        }
        model->appendRow(Lab);
    }
}

int MLT::Running::clustering(vector<Mat> images, int type, int k, int repetitions, float max_dist, float cell_size, vector<float> &labels){
    Clustering clus;
    Mat centers;
    labels.clear();
    int er=0;
    std::thread thrd;
    if(type==1)
        thrd=std::thread(&MLT::Clustering::K_mean,&clus, images,k,std::ref(labels),std::ref(centers),repetitions,KMEANS_RANDOM_CENTERS);
    else if(type==2)
        thrd=std::thread (&MLT::Clustering::K_mean,&clus, images,k,std::ref(labels),std::ref(centers),repetitions,KMEANS_PP_CENTERS);
    else if(type==3)
        thrd=std::thread (&MLT::Clustering::Distancias_Encadenadas,&clus, images,max_dist,std::ref(labels),std::ref(centers));
    else if(type==4)
        thrd=std::thread (&MLT::Clustering::Min_Max,&clus, images,max_dist,std::ref(labels),std::ref(centers));
    else if(type==5)
        thrd=std::thread (&MLT::Clustering::Histograma,&clus, images,cell_size,std::ref(labels),std::ref(centers));
    else if(type==6)
        thrd=std::thread (&MLT::Clustering::EXP_MAX,&clus, images,std::ref(labels),std::ref(centers),k,ml::EM::COV_MAT_SPHERICAL);
    else if(type==7)
        thrd=std::thread (&MLT::Clustering::EXP_MAX,&clus, images,std::ref(labels),std::ref(centers),k,ml::EM::COV_MAT_DIAGONAL);
    else if(type==8)
        thrd=std::thread (&MLT::Clustering::EXP_MAX,&clus, images,std::ref(labels),std::ref(centers),k,ml::EM::COV_MAT_GENERIC);


    thrd.join();

    if(this->gen.error==1)
        return 1;
}
