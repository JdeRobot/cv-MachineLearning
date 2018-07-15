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

int MLT::Running::load_dataset(string path){
    this->org_ref="";
    int pos=0;
    for(uint i=0; i<path.size(); i++){
        if(path[i]=='/')
            pos=i;
    }
    for(uint i=pos+1; i<path.size(); i++)
        this->org_ref=this->org_ref+path[i];
    string archivo_i=path+"/Info.xml";
    cv::FileStorage Archivo_i(archivo_i,CV_STORAGE_READ);

    if(!Archivo_i.isOpened())
        return 1;

    int num;
    Archivo_i["Num_Datos"]>>num;
    Archivo_i.release();

    string input_directory=path+"/Recortes.txt";

    this->base_progreso=1;
    this->max_progreso=100;
    this->gen.progreso=0;

    std::thread thrd(&MLT::Generacion::Cargar_Fichero,&gen,input_directory,std::ref(this->org_images),std::ref(this->org_labels),std::ref(this->org_info));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(gen.running==true)
        update_gen();

    thrd.join();

    if(gen.error==1)
        return 2;

    return 0;
}

int MLT::Running::synthetic_data(string ref, int num_clases, int num_data_clase, int size_x, int size_y, float ancho, float separacion_clases){
    Size size_img;
    size_img.width=size_x;
    size_img.height=size_y;
    if(size_img.height>1)
        size_img.height=1;


    this->base_progreso=1;
    this->max_progreso=100;
    this->gen.progreso=0;

    std::thread thrd(&MLT::Generacion::Random_Synthetic_Data,&gen, ref, num_clases, num_data_clase, size_img, ancho, separacion_clases, std::ref(this->org_images), std::ref(this->org_labels),std::ref(this->org_info), this->save_data);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(gen.running==true)
        update_gen();

    thrd.join();
    this->org_ref=ref;

    if(this->gen.error==1)
        return 1;

}

int MLT::Running::save(string ref){
    this->base_progreso=1;
    this->max_progreso=100;

    std::thread thrd(&MLT::Generacion::Guardar_Datos,&gen, ref,this->result_images,this->result_labels,this->result_info);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(gen.running==true)
        update_gen();

    thrd.join();

    if(this->gen.error==1)
        return 1;
}

int MLT::Running::join_data(string ref, string path){
    this->base_progreso=1;
    this->max_progreso=100;
    this->gen.progreso=0;

    std::thread thrd(&MLT::Generacion::Juntar_Recortes,&gen, ref,path);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(gen.running==true)
        update_gen();

    thrd.join();

    if(this->gen.error==1)
        return 1;
}

int MLT::Running::plot_data(int type_plot, vector<int> dim){
    if(dim[0]>this->org_images[0].cols*this->org_images[0].rows*this->org_images[0].channels())
        return 1;
    Representacion rep;
    int e=0;
    vector<float> labels;
    vector<Scalar> colors;
    if(type_plot<4){
        for(int i=0; i<this->org_labels.size(); i++)
            labels.push_back(org_labels[i]);
        for(int i=0; i<this->colors.size(); i++)
            colors.push_back(this->colors[i]);
    }
    else{
        if(this->result_images.empty())
            return 1;
        type_plot=type_plot-4;
        bool all_successes=true;
        for(int i=0; i<this->result_labels.size(); i++){
            if(this->result_labels[i]==this->org_labels[i])
                labels.push_back(1.);
            else{
                labels.push_back(-1.);
                all_successes=false;
            }
        }
        if(all_successes)
            colors.push_back(Scalar(0,255,0));
        else{
            colors.push_back(Scalar(0,0,255));
            colors.push_back(Scalar(0,255,0));
        }
    }

    if(type_plot==0){
        e=rep.Data_represent("DATA "+this->org_ref,this->org_images,labels,dim,colors);
    }
    else if(type_plot==1){
        e=rep.Ellipse_represent("ELLIPSES "+this->org_ref,this->org_images,labels,dim,colors);
    }
    else if(type_plot==2){
        e=rep.Data_Ellipse_represent("DATA AND ELLIPSES "+this->org_ref,this->org_images,labels,dim,colors);
    }
    else if(type_plot==3){
        Analisis an;
        vector<vector<Mat> > histogram;
        vector<vector<int> > bars;
        an.Histograma(this->org_images,labels,this->num_bar,histogram,bars);
        e=rep.Histogram_represent("HISTOGRAM "+this->org_ref,histogram,colors, dim[0]);
    }
    return e;
}

int MLT::Running::analyse_data(QStandardItemModel *model){
    vector<Mat> means,  std, covariance;
    vector<vector<Mat> > d_prime;
    bool negative;
    vector<int> number;

    std::thread thrd(&MLT::Analisis::Estadisticos_Covarianzas,&ana, this->org_images, this->org_labels, std::ref(means),std::ref(std), std::ref(d_prime), std::ref(covariance), std::ref(negative), std::ref(number));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(this->ana.running==true)
        update_analysis();

    thrd.join();


    if(this->ana.error==1)
        return 1;

    if(this->org_images[0].cols*this->org_images[0].rows*this->org_images[0].channels()<=1024){
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
        QStandardItem *Dim = new QStandardItem(QString("%1").arg(this->org_images.size()));
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
        if(this->org_images[0].cols*this->org_images[0].rows*this->org_images[0].channels()<=1024){
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

int MLT::Running::analyse_result(QStandardItemModel *model){
    Mat confusion;
    float error;
    vector<Analisis::Ratios_data> rates;

    std::thread thrd(&MLT::Analisis::Confusion_Ratios,&ana, this->org_labels, this->result_labels, std::ref(confusion),std::ref(error), std::ref(rates));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(this->ana.running==true)
        update_analysis();

    thrd.join();


    if(this->ana.error==1)
        return 1;

    int progreso=0;

    Auxiliares aux;
    bool negativa;
    int num=aux.numero_etiquetas(this->result_labels,negativa);
    this->max_progreso=num;

    QStandardItem *ER= new QStandardItem(QString("Error"));
    QStandardItem *Er = new QStandardItem(QString("%1").arg(error));
    ER->appendRow(Er);
    model->appendRow(ER);
    QStandardItem *Conf = new QStandardItem(QString("Confusion"));
    for(int j=0; j<confusion.rows; j++){
        stringstream linea_conf;
        for(int k=0; k<confusion.cols; k++){
            linea_conf<<confusion.at<float>(j,k);
            linea_conf<<"   ";
        }
        linea_conf<<";\n";
        QString valor=QString::fromStdString(linea_conf.str());
        QStandardItem *conf = new QStandardItem(QString(valor));
        Conf->appendRow(conf);
    }
    model->appendRow(Conf);
    QStandardItem *Rati= new QStandardItem(QString("Rates"));
    model->appendRow(Rati);
    for(int i=0; i<num; i++){
        int etiqueta;
        if(negativa){
            if(i==0)
                etiqueta=-1;
            else
                etiqueta=i;
        }
        else
            etiqueta=i+1;
        QStandardItem *Lab = new QStandardItem(QString("Label %1").arg(etiqueta));
        QStandardItem *VP = new QStandardItem(QString("TP"));
        QStandardItem *vp = new QStandardItem(QString("%1").arg(rates[i].VP));
        VP->appendRow(vp);
        Lab->appendRow(VP);
        QStandardItem *VN = new QStandardItem(QString("TN"));
        QStandardItem *vn = new QStandardItem(QString("%1").arg(rates[i].VN));
        VN->appendRow(vn);
        Lab->appendRow(VN);
        QStandardItem *FP = new QStandardItem(QString("FP"));
        QStandardItem *fp = new QStandardItem(QString("%1").arg(rates[i].FP));
        FP->appendRow(fp);
        Lab->appendRow(FP);
        QStandardItem *FN = new QStandardItem(QString("FN"));
        QStandardItem *fn = new QStandardItem(QString("%1").arg(rates[i].FN));
        FN->appendRow(fn);
        Lab->appendRow(FN);
        QStandardItem *TAR = new QStandardItem(QString("TAR"));
        QStandardItem *tar = new QStandardItem(QString("%1").arg(rates[i].TAR));
        TAR->appendRow(tar);
        Lab->appendRow(TAR);
        QStandardItem *TRR = new QStandardItem(QString("TRR"));
        QStandardItem *trr = new QStandardItem(QString("%1").arg(rates[i].TRR));
        TRR->appendRow(trr);
        Lab->appendRow(TRR);
        QStandardItem *FAR = new QStandardItem(QString("FAR"));
        QStandardItem *far = new QStandardItem(QString("%1").arg(rates[i].FAR));
        FAR->appendRow(far);
        Lab->appendRow(FAR);
        QStandardItem *FRR = new QStandardItem(QString("FRR"));
        QStandardItem *frr = new QStandardItem(QString("%1").arg(rates[i].FRR));
        FRR->appendRow(frr);
        Lab->appendRow(FRR);
        QStandardItem *PPV = new QStandardItem(QString("PPV"));
        QStandardItem *ppv = new QStandardItem(QString("%1").arg(rates[i].PPV));
        PPV->appendRow(ppv);
        Lab->appendRow(PPV);
        QStandardItem *NPV = new QStandardItem(QString("NPV"));
        QStandardItem *npv = new QStandardItem(QString("%1").arg(rates[i].NPV));
        NPV->appendRow(npv);
        Lab->appendRow(NPV);
        QStandardItem *FDR = new QStandardItem(QString("FDR"));
        QStandardItem *fdr = new QStandardItem(QString("%1").arg(rates[i].FDR));
        FDR->appendRow(fdr);
        Lab->appendRow(FDR);
        QStandardItem *F1 = new QStandardItem(QString("F1"));
        QStandardItem *f1 = new QStandardItem(QString("%1").arg(rates[i].F1));
        F1->appendRow(f1);
        Lab->appendRow(F1);
        QStandardItem *INFORMEDNESS = new QStandardItem(QString("INFORMEDNESS"));
        QStandardItem *informedness = new QStandardItem(QString("%1").arg(rates[i].INFORMEDNESS));
        INFORMEDNESS->appendRow(informedness);
        Lab->appendRow(INFORMEDNESS);
        QStandardItem *MARKEDNESS = new QStandardItem(QString("MARKEDNESS"));
        QStandardItem *markedness = new QStandardItem(QString("%1").arg(rates[i].MARKEDNESS));
        MARKEDNESS->appendRow(markedness);
        Lab->appendRow(MARKEDNESS);
        QStandardItem *EXP_ERROR = new QStandardItem(QString("EXP_ERROR"));
        QStandardItem *exp_error = new QStandardItem(QString("%1").arg(rates[i].EXP_ERROR));
        EXP_ERROR->appendRow(exp_error);
        Lab->appendRow(EXP_ERROR);
        QStandardItem *LR_NEG = new QStandardItem(QString("LR_NEG"));
        QStandardItem *lr_neg = new QStandardItem(QString("%1").arg(rates[i].LR_NEG));
        LR_NEG->appendRow(lr_neg);
        Lab->appendRow(LR_NEG);
        QStandardItem *LR_POS = new QStandardItem(QString("LR_POS"));
        QStandardItem *lr_pos = new QStandardItem(QString("%1").arg(rates[i].LR_POS));
        LR_POS->appendRow(lr_pos);
        Lab->appendRow(LR_POS);
        QStandardItem *DOR = new QStandardItem(QString("DOR"));
        QStandardItem *dor = new QStandardItem(QString("%1").arg(rates[i].DOR));
        DOR->appendRow(dor);
        Lab->appendRow(DOR);
        QStandardItem *ACC = new QStandardItem(QString("ACC"));
        QStandardItem *acc = new QStandardItem(QString("%1").arg(rates[i].ACC));
        ACC->appendRow(acc);
        Lab->appendRow(ACC);
        QStandardItem *PREVALENCE = new QStandardItem(QString("PREVALENCE"));
        QStandardItem *prevalence = new QStandardItem(QString("%1").arg(rates[i].PREVALENCE));
        PREVALENCE->appendRow(prevalence);
        Lab->appendRow(PREVALENCE);
        Rati->appendRow(Lab);
        progreso++;
        this->window->v_progress_Analysis->setValue(30+(70*progreso/this->max_progreso));
    }
    this->window->v_progress_Analysis->setValue(100);
    this->window->v_progress_Analysis->setValue(0);
    QApplication::restoreOverrideCursor();
}

int MLT::Running::clustering(string ref,int type, int k, int repetitions, float max_dist, float cell_size){
    Clustering clus;
    Mat centers;
    this->result_labels.clear();
    int er=0;
    std::thread thrd;
    if(type==1)
        thrd=std::thread(&MLT::Clustering::K_mean,&clus, this->org_images,k,std::ref(this->result_labels),std::ref(centers),repetitions,KMEANS_RANDOM_CENTERS);
    else if(type==2)
        thrd=std::thread (&MLT::Clustering::K_mean,&clus, this->org_images,k,std::ref(this->result_labels),std::ref(centers),repetitions,KMEANS_PP_CENTERS);
    else if(type==3)
        thrd=std::thread (&MLT::Clustering::Distancias_Encadenadas,&clus, this->org_images,max_dist,std::ref(this->result_labels),std::ref(centers));
    else if(type==4)
        thrd=std::thread (&MLT::Clustering::Min_Max,&clus, this->org_images,max_dist,std::ref(this->result_labels),std::ref(centers));
    else if(type==5)
        thrd=std::thread (&MLT::Clustering::Histograma,&clus, this->org_images,cell_size,std::ref(this->result_labels),std::ref(centers));
    else if(type==6)
        thrd=std::thread (&MLT::Clustering::EXP_MAX,&clus, this->org_images,std::ref(this->result_labels),std::ref(centers),k,ml::EM::COV_MAT_SPHERICAL);
    else if(type==7)
        thrd=std::thread (&MLT::Clustering::EXP_MAX,&clus, this->org_images,std::ref(this->result_labels),std::ref(centers),k,ml::EM::COV_MAT_DIAGONAL);
    else if(type==8)
        thrd=std::thread (&MLT::Clustering::EXP_MAX,&clus, this->org_images,std::ref(this->result_labels),std::ref(centers),k,ml::EM::COV_MAT_GENERIC);


    thrd.join();

    if(clus.error==1)
        return 1;

    this->result_images.clear();
    for(int i=0; i<this->org_images.size(); i++){
        this->result_images.push_back(this->org_images[i]);
    }
    this->result_ref=ref;
    this->result_info.Tipo_Datos=this->org_info.Tipo_Datos;
    this->result_info.Num_Datos=this->org_info.Num_Datos;
    this->result_info.Tam_X=this->org_info.Tam_X;
    this->result_info.Tam_Y=this->org_info.Tam_Y;
    this->result_info.Tam_Orig_X=this->org_info.Tam_Orig_X;
    this->result_info.Tam_Orig_Y=this->org_info.Tam_Orig_Y;
    this->result_info.si_lda=this->org_info.si_lda;
    this->result_info.si_pca=this->org_info.si_pca;
    this->result_info.si_dist=this->org_info.si_dist;
    this->result_info.si_d_prime=this->org_info.si_d_prime;
    this->org_info.LDA.copyTo(this->result_info.LDA);
    this->org_info.PCA.copyTo(this->result_info.PCA);
    this->org_info.DS.copyTo(this->result_info.DS);
    this->org_info.D_PRIME.copyTo(this->result_info.D_PRIME);

    return 0;
}

int MLT::Running::dimensionality(string ref, int size_reduc, int type){
    Dimensionalidad::Reducciones reduc;
    reduc.tam_reduc=size_reduc;

    if(type==LDA_DIM)
        reduc.si_lda=true;
    else if(type==PCA_DIM)
        reduc.si_pca=true;
    else if(type==MAXDIST_DIM)
        reduc.si_dist=true;
    else if(type==D_PRIME_DIM)
        reduc.si_d_prime=true;
    Dimensionalidad dim(ref);
    std::thread  thrd=std::thread(&MLT::Dimensionalidad::Reducir,&dim, this->org_images, std::ref(this->result_images), this->org_labels, reduc, std::ref(this->result_info), this->save_other);

    thrd.join();

    if(this->gen.error==1)
        return 1;

    this->result_labels.clear();
    for(int i=0; i<this->org_labels.size(); i++){
        this->result_labels.push_back(this->org_labels[i]);
    }
    this->result_ref=ref;
    this->result_info.Tipo_Datos=this->org_info.Tipo_Datos;
    this->result_info.Num_Datos=this->org_info.Num_Datos;
    this->result_info.Tam_X=this->org_info.Tam_X;
    this->result_info.Tam_Y=this->org_info.Tam_Y;
    this->result_info.Tam_Orig_X=this->org_info.Tam_Orig_X;
    this->result_info.Tam_Orig_Y=this->org_info.Tam_Orig_Y;
    this->result_info.si_lda=this->org_info.si_lda;
    this->result_info.si_pca=this->org_info.si_pca;
    this->result_info.si_dist=this->org_info.si_dist;
    this->result_info.si_d_prime=this->org_info.si_d_prime;
    this->org_info.LDA.copyTo(this->result_info.LDA);
    this->org_info.PCA.copyTo(this->result_info.PCA);
    this->org_info.DS.copyTo(this->result_info.DS);
    this->org_info.D_PRIME.copyTo(this->result_info.D_PRIME);

    return 0;
}

int MLT::Running::dimension_cuality(string ref, int size_reduc, int type_reduc, int type_measure, string &result){
    int e=0;
    if(this->org_images.empty())
        return 1;
    if(this->org_labels.empty())
        return 1;

    Representacion rep;
    Dimensionalidad dim(ref);
    Mat separability,accumulative;
    int best_dim;
    Dimensionalidad::Reducciones reduc;
    reduc.tam_reduc=size_reduc;

    std::thread thrd;
    if(type_measure==0)
        thrd=std::thread(&MLT::Dimensionalidad::Calidad_dimensiones_distancia,&dim, this->org_images,this->org_labels,type_reduc,size_reduc,std::ref(separability),std::ref(accumulative),std::ref(best_dim));

    else if(type_measure==1)
        thrd=std::thread(&MLT::Dimensionalidad::Calidad_dimensiones_d_prime,&dim, this->org_images,this->org_labels,type_reduc,size_reduc,std::ref(separability),std::ref(accumulative),std::ref(best_dim));

    thrd.join();

    stringstream txt;
    txt<<"INFO: The optimum number of dimensions is "<<best_dim;
    result=txt.str();

    vector<float> graphic;
    for(int i=0; i<separability.rows; i++)
        graphic.push_back(1.0);
    for(int i=0; i<accumulative.rows; i++)
        graphic.push_back(2.0);
    vector<Scalar> colors;
    colors.push_back(Scalar(0,0,255));
    colors.push_back(Scalar(0,255,0));

    Mat sep=Mat::zeros(separability.rows+accumulative.rows,2,CV_32F);
    for(int i=0; i<separability.rows; i++)
        separability.row(i).copyTo(sep.row(i));
    for(int i=1; i<accumulative.rows+1; i++)
        accumulative.row(i-1).copyTo(sep.row(separability.rows-1+i));
    Mat most=Mat::zeros(150,400,CV_8UC3);
    most=most+Scalar(255,255,255);
    String texto="SEPARABILITY";
    putText(most,texto,Point(10,50),1,1.5,colors[0],2);
    texto="ACCUMULATIVE SEPARABILITY";
    putText(most,texto,Point(10,100),1,1.5,colors[1],2);
    imshow("LEGEND",most);
    rep.Continuous_data_represent("DIMENSION QUALITY "+this->org_ref, sep, graphic, colors);

    return 0;
}

int MLT::Running::generate_data(string ref, string input_directory, int type, int scale_x, int scale_y, bool square, int number){
    this->base_progreso=1;
    this->max_progreso=100;

    cv::Size2i scale=cv::Size2i(scale_x,scale_y);
    std::thread thrd;
    if(type==0){
        thrd=std::thread(&MLT::Generacion::Datos_Imagenes,&this->gen, ref, input_directory,scale,std::ref(this->org_labels),std::ref(this->org_images),std::ref(this->org_info),this->save_data);
    }
    else if(type==1){
        thrd=std::thread(&MLT::Generacion::Etiquetar,&this->gen, ref, input_directory,scale,std::ref(this->org_labels),std::ref(this->org_images),std::ref(this->org_info),this->save_data);
    }
    else if(type==2){
        thrd=std::thread(&MLT::Generacion::Recortar_Etiquetar_imagenes,&this->gen, ref, input_directory,square,scale,std::ref(this->org_labels),std::ref(this->org_images),std::ref(this->org_info),this->save_data);
    }
    else if(type==3){
        cv::VideoCapture cap(input_directory);
        thrd=std::thread(&MLT::Generacion::Recortar_Etiquetar_video,&this->gen, ref, cap,square,scale,std::ref(this->org_labels),std::ref(this->org_images),std::ref(this->org_info),this->save_data);
    }
    else if(type==4){
        thrd=std::thread(&MLT::Generacion::Autonegativos,&this->gen, ref, input_directory, scale, number,std::ref(this->org_images),std::ref(this->org_labels),std::ref(this->org_info),this->save_data);
    }
    else if(type==5){
        cv::VideoCapture cap(input_directory);
        thrd=std::thread(&MLT::Generacion::Autopositivos,&this->gen, ref, cap,square,scale,std::ref(this->org_labels),std::ref(this->org_images),std::ref(this->org_info),this->save_data);
    }
    else if(type==6){
        cv::VideoCapture cap(input_directory);
        thrd=std::thread(&MLT::Generacion::Autogeneracion,&this->gen, ref, cap, number, square, scale, std::ref(this->org_labels),std::ref(this->org_images),std::ref(this->org_info),this->save_data);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(this->gen.running==true)
        update_gen();

    thrd.join();

    this->org_ref=ref;
}

int MLT::Running::descriptors(string &ref, int descriptor, string pc_descriptor, string extractor, int size_x, int size_y, int block_x, int block_y, double sigma, double threshold, bool gamma, int n_levels){
    if(descriptor>=0 && descriptor<=9){
        Basic_Transformations basic(this->org_info.Tipo_Datos,descriptor);
        int e=basic.Extract(this->org_images,this->result_images);
        if(e==1)
            return 1;
    }
    else if(descriptor==10){
        cv::Size size=cv::Size(size_x,size_y);
        cv::Size block=cv::Size(block_x,block_y);
        if(size.height>this->org_images[0].rows || size.width>this->org_images[0].cols){
            return 1;
        }
        HOG H(size,block, sigma,threshold, gamma, n_levels);
        int e=H.Extract(this->org_images,this->result_images);
        if(this->result_images.size()!=this->org_images.size() || e==1)
            return 1;
    }
    else if(descriptor==11){
        descriptor=descriptor-9;
        float parameter;
        Puntos_Caracteristicos des(pc_descriptor,extractor,parameter);
        int e=des.Extract(this->org_images,this->result_images);
        if(this->result_images.size()!=this->org_images.size() || e==1)
            return 1;
    }
    else
        return 1;

    if(descriptor==0){
        this->result_ref=this->org_ref+"_RGB";
        this->result_info.Tipo_Datos=RGB;
    }
    else if(descriptor==1){
        this->result_ref=this->org_ref+"_GRAY";
        this->result_info.Tipo_Datos=GRAY;
    }
    else if(descriptor==2){
        this->result_ref=this->org_ref+"_HSV";
        this->result_info.Tipo_Datos=HSV;
    }
    else if(descriptor==3){
        this->result_ref=this->org_ref+"_H";
        this->result_info.Tipo_Datos=H_CHANNEL;
    }
    else if(descriptor==4){
        this->result_ref=this->org_ref+"_S";
        this->result_info.Tipo_Datos=S_CHANNEL;
    }
    else if(descriptor==5){
        this->result_ref=this->org_ref+"_V";
        this->result_info.Tipo_Datos=V_CHANNEL;
    }
    else if(descriptor==6){
        this->result_ref=this->org_ref+"_THRESHOLD";
        this->result_info.Tipo_Datos=THRESHOLD;
    }
    else if(descriptor==7){
        this->result_ref=this->org_ref+"_CANNY";
        this->result_info.Tipo_Datos=CANNY;
    }
    else if(descriptor==8){
        this->result_ref=this->org_ref+"_SOBEL";
        this->result_info.Tipo_Datos=SOBEL;
    }
    else if(descriptor==9){
        this->result_ref=this->org_ref+"_PREDOMINANT_COLOR";
        this->result_info.Tipo_Datos=COLOR_PREDOMINANTE;
    }
    else if(descriptor==10){
        this->result_ref=this->org_ref+"_HOG";
        this->result_info.Tipo_Datos=HOG_DES;
    }
    else if(descriptor==11){
        this->result_ref=this->org_ref+"_PC";
        this->result_info.Tipo_Datos=PUNTOS_CARACTERISTICOS;
    }

    this->result_info.Tam_X=this->result_images[0].cols;
    this->result_info.Tam_Y=this->result_images[0].rows;
    ref=this->result_ref;
}

int MLT::Running::expand_dataset(string ref, int nframe, float max_noise, float max_blur, float max_x, float max_y, float max_z){
    this->gen.total_progreso=this->org_images.size();
    this->gen.progreso=0;
    this->base_progreso=30;
    this->max_progreso=20;

    std::thread thrd(&MLT::Generacion::Synthethic_Data,&this->gen,this->org_ref,this->org_images,this->org_labels,std::ref(this->result_images),std::ref(this->result_labels),nframe,max_noise,max_blur,max_x,max_y,max_z,std::ref(this->result_info),this->save_other);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(this->gen.running==true)
        update_gen();

    thrd.join();

    this->result_ref=ref;

    if(this->gen.error==1)
        return 1;

}

//int MLT::Running::detect(string input, int classes, float variance, float interclass, int window_x, int wnidow_y, int descriptor_type){
//    int e=0;
//    cv::Mat salida;
//    vector<cv::RotatedRect> detections;
//    vector<float> labels_detections;

//    int current_type=-1;
//    cv::Mat image;
//    if(input=="random_image"){
//        this->gen.Random_Synthetic_Image(classes,Size(500,500),variance,interclass,imagen);
//        current_type=GRAY;
//    }
//    else{
//        image=cv::imread(input);
//        if(imagen.empty()){
//            return 1;
//        }
//        image.convertTo(image,CV_32F);
//        if(image.cols<window_x || image.rows<window_y){
//            return 1;
//        }
//        current_type=RGB;
//    }

//    Descriptor *descriptor;
//    if(descriptor_type==RGB){
//        descriptor=0;
//    }
//    else if(descriptor_type==GRAY){
//        Basic_Transformations *basic=new Basic_Transformations(current_type,GRAY);
//        descriptor=basic;
//    }
//    else if(descriptor_type==HSV){
//        Basic_Transformations *basic=new Basic_Transformations(current_type,HSV);
//        descriptor=basic;
//    }
//    else if(descriptor_type==H_CHANNEL){
//        Basic_Transformations *basic=new Basic_Transformations(current_type,H_CHANNEL);
//        descriptor=basic;
//    }
//    else if(descriptor_type==S_CHANNEL){
//        Basic_Transformations *basic=new Basic_Transformations(current_type,S_CHANNEL);
//        descriptor=basic;
//    }
//    else if(udescriptor_type==V_CHANNEL){
//        Basic_Transformations *basic=new Basic_Transformations(current_type,V_CHANNEL);
//        descriptor=basic;
//    }
//    else if(descriptor_type==THRESHOLD){
//        Basic_Transformations *basic=new Basic_Transformations(current_type,THRESHOLD);
//        descriptor=basic;
//    }
//    else if(descriptor_type==CANNY){
//        Basic_Transformations *basic=new Basic_Transformations(current_type,CANNY);
//        descriptor=basic;
//    }
//    else if(descriptor_type==SOBEL){
//        Basic_Transformations *basic=new Basic_Transformations(current_type,SOBEL);
//        descriptor=basic;
//    }
//    else if(descriptor_type==HOG_DES){
//        if(Win_Size.height>ui->Vent_Y->value()|| Win_Size.width>ui->Vent_X->value())
//            return 1;
//        HOG *Hoog=new HOG(Win_Size,Block_Stride, Win_Sigma,Threshold_L2hys, Gamma_Correction, Nlevels);
//        descriptor= Hoog;
//    }
//    else if(descriptor_type==PUNTOS_CARACTERISTICOS){
//        Puntos_Caracteristicos *des=new Puntos_Caracteristicos(Tipo_Des,Tipo_Ext,Parametro);
//        descriptor= des;
//    }
//    else if(descriptor_type==COLOR_PREDOMINANTE){
//        Basic_Transformations *basic=new Basic_Transformations(current_type,COLOR_PREDOMINANTE);
//        descriptor=basic;
//    }
//    else{
//        return 1;
//    }

//    if(ui->radioPosicion->isChecked()){
//        if(ui->Clasif_Cargado_2->isChecked()){
//            if(ID==DISTANCIAS){
//                D.progreso=0;
//                D.max_progreso=100;
//                D.base_progreso=0;
//                D.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                D.window=ui;
//                Busqueda bus(&D,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==GAUSSIANO){
//                G.progreso=0;
//                G.max_progreso=100;
//                G.base_progreso=0;
//                G.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                G.window=ui;
//                Busqueda bus(&G,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==CASCADA_CLAS){
//                HA.progreso=0;
//                HA.max_progreso=100;
//                HA.base_progreso=0;
//                HA.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                HA.window=ui;
//                Busqueda bus(&HA,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==HISTOGRAMA){
//                H.progreso=0;
//                H.max_progreso=100;
//                H.base_progreso=0;
//                H.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                H.window=ui;
//                Busqueda bus(&H,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==KNN){
//                K.progreso=0;
//                K.max_progreso=100;
//                K.base_progreso=0;
//                K.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                K.window=ui;
//                Busqueda bus(&K,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==NEURONAL){
//                N.progreso=0;
//                N.max_progreso=100;
//                N.base_progreso=0;
//                N.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                N.window=ui;
//                Busqueda bus(&N,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==C_SVM){
//                S.progreso=0;
//                S.max_progreso=100;
//                S.base_progreso=0;
//                S.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                S.window=ui;
//                Busqueda bus(&S,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==RTREES){
//                RT.progreso=0;
//                RT.max_progreso=100;
//                RT.base_progreso=0;
//                RT.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                RT.window=ui;
//                Busqueda bus(&RT,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==DTREES){
//                DT.progreso=0;
//                DT.max_progreso=100;
//                DT.base_progreso=0;
//                DT.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                DT.window=ui;if(ui->radioPosicion->isChecked()){
//                    if(ui->Clasif_Cargado_2->isChecked()){
//                        if(ID==DISTANCIAS){
//                            D.progreso=0;
//                            D.max_progreso=100;
//                            D.base_progreso=0;
//                            D.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            D.window=ui;
//                            Busqueda bus(&D,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else if(ID==GAUSSIANO){
//                            G.progreso=0;
//                            G.max_progreso=100;
//                            G.base_progreso=0;
//                            G.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            G.window=ui;
//                            Busqueda bus(&G,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else if(ID==CASCADA_CLAS){
//                            HA.progreso=0;
//                            HA.max_progreso=100;
//                            HA.base_progreso=0;
//                            HA.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            HA.window=ui;
//                            Busqueda bus(&HA,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else if(ID==HISTOGRAMA){
//                            H.progreso=0;
//                            H.max_progreso=100;
//                            H.base_progreso=0;
//                            H.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            H.window=ui;
//                            Busqueda bus(&H,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else if(ID==KNN){
//                            K.progreso=0;
//                            K.max_progreso=100;
//                            K.base_progreso=0;
//                            K.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            K.window=ui;
//                            Busqueda bus(&K,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else if(ID==NEURONAL){
//                            N.progreso=0;
//                            N.max_progreso=100;
//                            N.base_progreso=0;
//                            N.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            N.window=ui;
//                            Busqueda bus(&N,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else if(ID==C_SVM){
//                            S.progreso=0;
//                            S.max_progreso=100;
//                            S.base_progreso=0;
//                            S.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            S.window=ui;
//                            Busqueda bus(&S,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else if(ID==RTREES){
//                            RT.progreso=0;
//                            RT.max_progreso=100;
//                            RT.base_progreso=0;
//                            RT.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            RT.window=ui;
//                            Busqueda bus(&RT,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else if(ID==DTREES){
//                            DT.progreso=0;
//                            DT.max_progreso=100;
//                            DT.base_progreso=0;
//                            DT.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            DT.window=ui;
//                            Busqueda bus(&DT,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else if(ID==BOOSTING){
//                            B.progreso=0;
//                            B.max_progreso=100;
//                            B.base_progreso=0;
//                            B.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            B.window=ui;
//                            Busqueda bus(&B,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else if(ID==EXP_MAX){
//                            E.progreso=0;
//                            E.max_progreso=100;
//                            E.base_progreso=0;
//                            E.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            E.window=ui;
//                            Busqueda bus(&E,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else if(ID==MICLASIFICADOR){
//                            MC.progreso=0;
//                            MC.max_progreso=100;
//                            MC.base_progreso=0;
//                            MC.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            MC.window=ui;
//                            Busqueda bus(&MC,tipo_dato,descriptor);
//                            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                        }
//                        else{
//                            return 1;
//                        }
//                        if(e==1){
//                            return 1;
//                        }
//                    }
//                    else if(ui->Multiclasif_2->isChecked()){
//                        vector<Clasificador*> clasificadores;
//                        for(uint i=0; i<id_clasificadores.size(); i++){
//                            if(id_clasificadores[i]==DISTANCIAS){
//                                Clasificador_Distancias *clasi=new Clasificador_Distancias(nombres[i]);
//                                clasi->Read_Data();
//                                clasificadores.push_back(clasi);
//                            }
//                            else if(id_clasificadores[i]==GAUSSIANO){
//                                Clasificador_Gaussiano *clasi=new Clasificador_Gaussiano(nombres[i]);
//                                clasi->Read_Data();
//                                clasificadores.push_back(clasi);
//                            }
//                            else if(id_clasificadores[i]==CASCADA_CLAS){
//                                Clasificador_Cascada *clasi=new Clasificador_Cascada(nombres[i]);
//                                clasi->Read_Data();
//                                clasificadores.push_back(clasi);
//                            }
//                            else if(id_clasificadores[i]==HISTOGRAMA){
//                                Clasificador_Histograma *clasi=new Clasificador_Histograma(nombres[i]);
//                                clasi->Read_Data();
//                                clasificadores.push_back(clasi);
//                            }
//                            else if(id_clasificadores[i]==KNN){
//                                Clasificador_KNN *clasi=new Clasificador_KNN(nombres[i]);
//                                clasi->Read_Data();
//                                clasificadores.push_back(clasi);
//                            }
//                            else if(id_clasificadores[i]==NEURONAL){
//                                Clasificador_Neuronal *clasi=new Clasificador_Neuronal(nombres[i]);
//                                clasi->Read_Data();
//                                clasificadores.push_back(clasi);
//                            }
//                            else if(id_clasificadores[i]==C_SVM){
//                                Clasificador_SVM *clasi=new Clasificador_SVM(nombres[i]);
//                                clasi->Read_Data();
//                                clasificadores.push_back(clasi);
//                            }
//                            else if(id_clasificadores[i]==RTREES){
//                                Clasificador_RTrees *clasi=new Clasificador_RTrees(nombres[i]);
//                                clasi->Read_Data();
//                                clasificadores.push_back(clasi);
//                            }
//                            else if(id_clasificadores[i]==DTREES){
//                                Clasificador_DTrees *clasi=new Clasificador_DTrees(nombres[i]);
//                                clasi->Read_Data();
//                                clasificadores.push_back(clasi);
//                            }
//                            else if(id_clasificadores[i]==BOOSTING){
//                                Clasificador_Boosting *clasi=new Clasificador_Boosting(nombres[i]);
//                                clasi->Read_Data();
//                                clasificadores.push_back(clasi);
//                            }
//                    //        else if(id_clasificadores[i]==GBT){
//                    //            Clasificador_GBTrees *clasi=new Clasificador_GBTrees(nombres[i]);
//                //                clasi->Read_Data();
//                //                clasificadores.push_back(clasi);
//                    //        }
//                    //        else if(id_clasificadores[i]==ERTREES){
//                    //            Clasificador_ERTrees *clasi=new Clasificador_ERTrees(nombres[i]);
//                //                clasi->Read_Data();
//                //                clasificadores.push_back(clasi);
//                    //        }
//                            else if(id_clasificadores[i]==EXP_MAX){
//                                Clasificador_EM *clasi=new Clasificador_EM(nombres[i]);
//                                clasi->Read_Data();
//                                clasificadores.push_back(clasi);
//                            }
//                        }
//                        MultiClasificador multi(clasificadores);
//                        multi.progreso=0;
//                        multi.max_progreso=100;
//                        multi.base_progreso=0;
//                        multi.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                        multi.window=ui;
//                        Busqueda bus(&multi,tipo_dato,Tipo_Descriptor,&Multi_tipo);
//                        e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//                    }
//                    Representacion rep;
//                    Mat mostrar;
//                    imagen.convertTo(imagen,CV_32F);
//                    double minval,maxval;
//                    cv::minMaxLoc(imagen,&minval,&maxval);
//                    imagen=(imagen-minval)/(maxval-minval);
//                    imshow("Imagen",imagen);
//                    rep.Recuadros(imagen,recuadros,labels_recuadros,Col,mostrar,show_graphics);
//                }
//                else if(ui->radioTextura->isChecked()){
//                    if(ui->Clasif_Cargado_2->isChecked()){
//                        if(ID==DISTANCIAS){
//                            D.progreso=0;
//                            D.max_progreso=100;
//                            D.base_progreso=0;
//                            D.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            D.window=ui;
//                            Busqueda bus(&D,tipo_dato,Tipo_Descriptor);
//                            e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//                        }
//                        else if(ID==GAUSSIANO){
//                            G.progreso=0;
//                            G.max_progreso=100;
//                            G.base_progreso=0;
//                            G.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            G.window=ui;
//                            Busqueda bus(&G,tipo_dato,Tipo_Descriptor);
//                            e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//                        }
//                        else if(ID==CASCADA_CLAS){
//                            HA.progreso=0;
//                            HA.max_progreso=100;
//                            HA.base_progreso=0;
//                            HA.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            HA.window=ui;
//                            Busqueda bus(&HA,tipo_dato,Tipo_Descriptor);
//                            e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//                        }
//                        else if(ID==HISTOGRAMA){
//                            H.progreso=0;
//                            H.max_progreso=100;
//                            H.base_progreso=0;
//                            H.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            H.window=ui;
//                            Busqueda bus(&H,tipo_dato,Tipo_Descriptor);
//                            e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//                        }
//                        else if(ID==KNN){
//                            K.progreso=0;
//                            K.max_progreso=100;
//                            K.base_progreso=0;
//                            K.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            K.window=ui;
//                            Busqueda bus(&K,tipo_dato,Tipo_Descriptor);
//                            e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//                        }
//                        else if(ID==NEURONAL){
//                            N.progreso=0;
//                            N.max_progreso=100;
//                            N.base_progreso=0;
//                            N.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            N.window=ui;
//                            Busqueda bus(&N,tipo_dato,Tipo_Descriptor);
//                            e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//                        }
//                        else if(ID==C_SVM){
//                            S.progreso=0;
//                            S.max_progreso=100;
//                            S.base_progreso=0;
//                            S.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            S.window=ui;
//                            Busqueda bus(&S,tipo_dato,Tipo_Descriptor);
//                            e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//                        }
//                        else if(ID==RTREES){
//                            RT.progreso=0;
//                            RT.max_progreso=100;
//                            RT.base_progreso=0;
//                            RT.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                            RT.window=ui;
//                            Busqueda bus(&RT,tipo_dato,Tipo_Descriptor);
//                Busqueda bus(&DT,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==BOOSTING){
//                B.progreso=0;
//                B.max_progreso=100;
//                B.base_progreso=0;
//                B.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                B.window=ui;
//                Busqueda bus(&B,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==EXP_MAX){
//                E.progreso=0;
//                E.max_progreso=100;
//                E.base_progreso=0;
//                E.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                E.window=ui;
//                Busqueda bus(&E,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==MICLASIFICADOR){
//                MC.progreso=0;
//                MC.max_progreso=100;
//                MC.base_progreso=0;
//                MC.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                MC.window=ui;
//                Busqueda bus(&MC,tipo_dato,descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else{
//                return 1;
//            }
//            if(e==1){
//                return 1;
//            }
//        }
//        else if(ui->Multiclasif_2->isChecked()){
//            vector<Clasificador*> clasificadores;
//            for(uint i=0; i<id_clasificadores.size(); i++){
//                if(id_clasificadores[i]==DISTANCIAS){
//                    Clasificador_Distancias *clasi=new Clasificador_Distancias(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==GAUSSIANO){
//                    Clasificador_Gaussiano *clasi=new Clasificador_Gaussiano(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==CASCADA_CLAS){
//                    Clasificador_Cascada *clasi=new Clasificador_Cascada(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==HISTOGRAMA){
//                    Clasificador_Histograma *clasi=new Clasificador_Histograma(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==KNN){
//                    Clasificador_KNN *clasi=new Clasificador_KNN(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==NEURONAL){
//                    Clasificador_Neuronal *clasi=new Clasificador_Neuronal(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==C_SVM){
//                    Clasificador_SVM *clasi=new Clasificador_SVM(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==RTREES){
//                    Clasificador_RTrees *clasi=new Clasificador_RTrees(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==DTREES){
//                    Clasificador_DTrees *clasi=new Clasificador_DTrees(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==BOOSTING){
//                    Clasificador_Boosting *clasi=new Clasificador_Boosting(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//        //        else if(id_clasificadores[i]==GBT){
//        //            Clasificador_GBTrees *clasi=new Clasificador_GBTrees(nombres[i]);
//    //                clasi->Read_Data();
//    //                clasificadores.push_back(clasi);
//        //        }
//        //        else if(id_clasificadores[i]==ERTREES){
//        //            Clasificador_ERTrees *clasi=new Clasificador_ERTrees(nombres[i]);
//    //                clasi->Read_Data();
//    //                clasificadores.push_back(clasi);
//        //        }
//                else if(id_clasificadores[i]==EXP_MAX){
//                    Clasificador_EM *clasi=new Clasificador_EM(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//            }
//            MultiClasificador multi(clasificadores);
//            multi.progreso=0;
//            multi.max_progreso=100;
//            multi.base_progreso=0;
//            multi.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//            multi.window=ui;
//            Busqueda bus(&multi,tipo_dato,Tipo_Descriptor,&Multi_tipo);
//            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//        }
//        Representacion rep;
//        Mat mostrar;
//        imagen.convertTo(imagen,CV_32F);
//        double minval,maxval;
//        cv::minMaxLoc(imagen,&minval,&maxval);
//        imagen=(imagen-minval)/(maxval-minval);
//        imshow("Imagen",imagen);
//        rep.Recuadros(imagen,recuadros,labels_recuadros,Col,mostrar,show_graphics);
//    }
//    else if(ui->radioTextura->isChecked()){
//        if(ui->Clasif_Cargado_2->isChecked()){
//            if(ID==DISTANCIAS){
//                D.progreso=0;
//                D.max_progreso=100;
//                D.base_progreso=0;
//                D.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                D.window=ui;
//                Busqueda bus(&D,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==GAUSSIANO){
//                G.progreso=0;
//                G.max_progreso=100;
//                G.base_progreso=0;
//                G.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                G.window=ui;
//                Busqueda bus(&G,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==CASCADA_CLAS){
//                HA.progreso=0;
//                HA.max_progreso=100;
//                HA.base_progreso=0;
//                HA.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                HA.window=ui;
//                Busqueda bus(&HA,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==HISTOGRAMA){
//                H.progreso=0;
//                H.max_progreso=100;
//                H.base_progreso=0;
//                H.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                H.window=ui;
//                Busqueda bus(&H,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==KNN){
//                K.progreso=0;
//                K.max_progreso=100;
//                K.base_progreso=0;
//                K.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                K.window=ui;
//                Busqueda bus(&K,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==NEURONAL){
//                N.progreso=0;
//                N.max_progreso=100;
//                N.base_progreso=0;
//                N.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                N.window=ui;
//                Busqueda bus(&N,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==C_SVM){
//                S.progreso=0;
//                S.max_progreso=100;
//                S.base_progreso=0;
//                S.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                S.window=ui;
//                Busqueda bus(&S,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==RTREES){
//                RT.progreso=0;
//                RT.max_progreso=100;
//                RT.base_progreso=0;
//                RT.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                RT.window=ui;
//                Busqueda bus(&RT,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==DTREES){
//                DT.progreso=0;
//                DT.max_progreso=100;
//                DT.base_progreso=0;
//                DT.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                DT.window=ui;
//                Busqueda bus(&DT,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==BOOSTING){
//                B.progreso=0;
//                B.max_progreso=100;
//                B.base_progreso=0;
//                B.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                B.window=ui;
//                Busqueda bus(&B,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==EXP_MAX){
//                E.progreso=0;
//                E.max_progreso=100;
//                E.base_progreso=0;
//                E.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                E.window=ui;
//                Busqueda bus(&E,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==MICLASIFICADOR){
//                MC.progreso=0;
//                MC.max_progreso=100;
//                MC.base_progreso=0;
//                MC.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                MC.window=ui;
//                Busqueda bus(&MC,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else{
//                QMessageBox msgBox;
//                msgBox.setText("ERROR: No se ha cargado ningun clasificador");
//                msgBox.exec();
//                QApplication::restoreOverrideCursor();
//                return;
//            }
//            if(e==1){
//                QMessageBox msgBox;
//                msgBox.setText("ERROR: No se ha podido clasificar la imagen");
//                msgBox.exec();
//                QApplication::restoreOverrideCursor();
//                return;
//            }
//        }
//        else if(ui->Multiclasif_2->isChecked()){
//            vector<Clasificador*> clasificadores;
//            for(uint i=0; i<id_clasificadores.size(); i++){
//                if(id_clasificadores[i]==DISTANCIAS){
//                    Clasificador_Distancias *clasi=new Clasificador_Distancias(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==GAUSSIANO){
//                    Clasificador_Gaussiano *clasi=new Clasificador_Gaussiano(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==CASCADA_CLAS){
//                    Clasificador_Cascada *clasi=new Clasificador_Cascada(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==HISTOGRAMA){
//                    Clasificador_Histograma *clasi=new Clasificador_Histograma(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==KNN){
//                    Clasificador_KNN *clasi=new Clasificador_KNN(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==NEURONAL){
//                    Clasificador_Neuronal *clasi=new Clasificador_Neuronal(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==C_SVM){
//                    Clasificador_SVM *clasi=new Clasificador_SVM(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==RTREES){
//                    Clasificador_RTrees *clasi=new Clasificador_RTrees(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==DTREES){
//                    Clasificador_DTrees *clasi=new Clasificador_DTrees(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==BOOSTING){
//                    Clasificador_Boosting *clasi=new Clasificador_Boosting(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//                else if(id_clasificadores[i]==EXP_MAX){
//                    Clasificador_EM *clasi=new Clasificador_EM(nombres[i]);
//                    clasi->Read_Data();
//                    clasificadores.push_back(clasi);
//                }
//        //        else if(id_clasificadores[i]==GBT){
//        //            Clasificador_GBTrees *clasi=new Clasificador_GBTrees(nombres[i]);
//    //                clasi->Read_Data();
//    //                clasificadores.push_back(clasi);
//        //        }
//        //        else if(id_clasificadores[i]==ERTREES){
//        //            Clasificador_ERTrees *clasi=new Clasificador_ERTrees(nombres[i]);
//    //                clasi->Read_Data();
//    //                clasificadores.push_back(clasi);
//        //        }
//            }
//            MultiClasificador multi(clasificadores);
//            multi.progreso=0;
//            multi.max_progreso=100;
//            multi.base_progreso=0;
//            multi.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//            multi.window=ui;
//            Busqueda bus(&multi,tipo_dato,Tipo_Descriptor,&Multi_tipo);
//            e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//        }
//        if(!salida.empty()){
//            Mat mostrar;
//            Representacion rep;
//            imagen.convertTo(imagen,CV_32F);
//            double minval,maxval;
//            cv::minMaxLoc(imagen,&minval,&maxval);
//            imagen=(imagen-minval)/(maxval-minval);
//            imshow("Imagen",imagen);
//            e=rep.Color(salida,Col,mostrar,show_graphics);
//            if(e==1){
//                QMessageBox msgBox;
//                msgBox.setText("ERROR: No se ha podido representar la imagen clasificada");
//                msgBox.exec();
//                QApplication::restoreOverrideCursor();
//                return;
//            }
//            QApplication::restoreOverrideCursor();
//        }
//        else{
//            QMessageBox msgBox;
//            msgBox.setText("ERROR: No se ha podido clasificar la imagen");
//            msgBox.exec();
//            QApplication::restoreOverrideCursor();
//            return;
//        }
//    }
//    ui->progress_Clasificar->setValue(100);
//    ui->progress_Clasificar->setValue(0);
//    QApplication::restoreOverrideCursor();
//}
