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
    this->window->g_progress_datamanaging->setValue(this->base_progreso+(this->max_progreso*this->gen.progreso/this->gen.total_progreso));
    this->window->v_progress_datamanaging->setValue(this->base_progreso+(this->max_progreso*this->gen.progreso/this->gen.total_progreso));
    this->window->i_progress_datamanaging->setValue(this->base_progreso+(this->max_progreso*this->gen.progreso/this->gen.total_progreso));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void MLT::Running::update_analysis(){
    this->window->v_progress_Analysis->setValue(this->ana.progreso);
    this->window->g_progress_Analysis->setValue(this->ana.progreso);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void MLT::Running::update_classifier(int progress,int total_progress){
    this->window->m_progress_classifiers->setValue(this->base_progreso+(this->max_progreso*progress/total_progress));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}



int MLT::Running::load_dataset(string path){
    int pos=0;
    for(uint i=0; i<path.size(); i++){
        if(path[i]=='/')
            pos=i;
    }
    this->org_ref="";
    for(uint i=pos+1; i<path.size(); i++)
        this->org_ref=this->org_ref+path[i];

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

int MLT::Running::synthetic_data(string ref, int classes, int number, int size_x, int size_y, float variance, float interclass){
    Size size_img;
    size_img.width=size_x;
    size_img.height=size_y;

    this->base_progreso=1;
    this->max_progreso=100;
    this->gen.progreso=0;

    std::thread thrd(&MLT::Generacion::Random_Synthetic_Data,&gen, ref, classes, number, size_img, variance, interclass, std::ref(this->org_images), std::ref(this->org_labels),std::ref(this->org_info), this->save_data);
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
    return 0;
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

    if(dim.error==1)
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
    String text="SEPARABILITY";
    putText(most,text,Point(10,50),1,1.5,colors[0],2);
    text="ACCUMULATIVE SEPARABILITY";
    putText(most,text,Point(10,100),1,1.5,colors[1],2);
    imshow("LEGEND",most);
    int e=rep.Continuous_data_represent("DIMENSION QUALITY "+this->org_ref, sep, graphic, colors);
    if(e==1)
        return 1;

    return 0;
}

int MLT::Running::generate_data(string ref, string input_directory, int type, int scale_x, int scale_y, bool square, int number){
    this->base_progreso=1;
    this->max_progreso=100;

    cv::Size2i scale=cv::Size2i(scale_x,scale_y);
    std::thread thrd;
    if(type==0){
        thrd=std::thread(&MLT::Generacion::Datos_Imagenes,&this->gen, ref, input_directory,scale,std::ref(this->org_labels),std::ref(this->org_images),std::ref(this->org_info),this->save_data);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        while(this->gen.running==true)
            update_gen();

        thrd.join();
    }
    else if(type==1){
        this->gen.Etiquetar(ref, input_directory,scale,this->org_labels,this->org_images,this->org_info,this->save_data);
    }
    else if(type==2){
        this->gen.Recortar_Etiquetar_imagenes(ref, input_directory,square,scale,this->org_labels,this->org_images,this->org_info,this->save_data);
    }
    else if(type==3){
        cv::VideoCapture cap(input_directory);
        this->gen.Recortar_Etiquetar_video(ref, cap,square,scale,this->org_labels,this->org_images,this->org_info,this->save_data);
    }
    else if(type==4){
        this->gen.Autonegativos(ref, input_directory, scale, number,this->org_images,this->org_labels,this->org_info,this->save_data);
    }
    else if(type==5){
        cv::VideoCapture cap(input_directory);
        this->gen.Autopositivos(ref, cap,square,scale,this->org_labels,this->org_images,this->org_info,this->save_data);
    }
    else if(type==6){
        cv::VideoCapture cap(input_directory);
        this->gen.Autogeneracion(ref, cap, number, square, scale, this->org_labels,this->org_images,this->org_info,this->save_data);
    }

    this->org_ref=ref;
    return gen.error;
}


int MLT::Running::descriptors(string &ref, int descriptor, string pc_descriptor, string extractor, int size_x, int size_y, int block_x, int block_y, double sigma, double threshold, bool gamma, int n_levels, bool descriptor_parameter){
    this->result_images.clear();
    if(descriptor>=0 && descriptor<=9){
        Basic_Transformations basic(this->org_info.Tipo_Datos,descriptor);
        std::thread thrd(&MLT::Basic_Transformations::Extract,basic,this->org_images,std::ref(this->result_images));
        thrd.join();

        if(basic.error==1)
            return 1;
    }
    else if(descriptor==10){
        cv::Size size=cv::Size(size_x,size_y);
        cv::Size block=cv::Size(block_x,block_y);
        if(size.height>this->org_images[0].rows || size.width>this->org_images[0].cols){
            return 2;
        }
        HOG hog(size,block, sigma,threshold, gamma, n_levels);
        std::thread thrd(&MLT::HOG::Extract,hog,this->org_images,std::ref(this->result_images));
        thrd.join();

        if(this->result_images.size()!=this->org_images.size() || hog.error==1)
            return 1;
    }
    else if(descriptor==11){
        Puntos_Caracteristicos des(pc_descriptor,extractor,descriptor_parameter);
        std::thread thrd(&MLT::Puntos_Caracteristicos::Extract,des,this->org_images,std::ref(this->result_images));
        thrd.join();

        if(this->result_images.size()!=this->org_images.size() || des.error==1)
            return 1;
    }
    else
        return 3;

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

    this->result_labels.clear();
    for(int i=0; i<this->org_labels.size(); i++){
        this->result_labels.push_back(this->org_labels[i]);
    }
    this->result_ref=ref;
    this->result_info.Tipo_Datos=this->org_info.Tipo_Datos;
    this->result_info.Num_Datos=this->org_info.Num_Datos;
    this->result_info.Tam_X=this->result_images[0].cols;
    this->result_info.Tam_Y=this->result_images[0].rows;
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

    ref=this->result_ref;
}

int MLT::Running::expand_dataset(string ref, int nframe, float max_noise, float max_blur, float max_x, float max_y, float max_z){
    this->gen.total_progreso=this->org_images.size();
    this->gen.progreso=0;
    this->base_progreso=30;
    this->max_progreso=100;

    std::thread thrd(&MLT::Generacion::Synthethic_Data,&this->gen,ref,this->org_images,this->org_labels,std::ref(this->result_images),std::ref(this->result_labels),nframe,max_noise,max_blur,max_x,max_y,max_z,std::ref(this->result_info),this->save_other);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(this->gen.running==true)
        update_gen();

    thrd.join();


    if(this->gen.error==1)
        return 1;

    this->result_ref=ref;

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
}

int MLT::Running::represent_images(int type, int label){
    int e=0;
    Representacion rep;
    if(type==0){
        if(this->org_images.empty() || this->org_labels.empty())
            return 2;
        e=rep.Imagen(this->org_images,this->org_labels,label);
    }
    else if(type==1){
        if(this->result_images.empty() || this->result_labels.empty())
            return 3;
        e=rep.Imagen(this->result_images,this->result_labels,label);
    }
    return e;
}

int MLT::Running::detect_image(int type_running, int input_type, string input_path, int descriptor_type, MultiClasificador::Multi_type multi_params,
                               int n_classes, float variance, float interclass,
                               int window_x, int window_y, int jump, int pyramid, int rotation,
                               bool postprocess, bool overlap, bool isolated, float dist_boxes, int dist_rotation,
                               string pc_descriptor, string extractor, int size_x, int size_y, int block_x, int block_y,
                               double sigma, double threhold_l2hys, bool gamma, int n_levels, bool descriptor_parameter,
                               cv::Mat &image, cv::Mat &output, vector<cv::RotatedRect> &detections, vector<float> &labels_detections){

    int e=0;
    int current_type=-1;
    if(input_type==1){
        this->gen.Random_Synthetic_Image(n_classes,Size(500,500),variance,interclass,image);
        current_type=GRAY;
    }
    else if(input_type==2){
        image=cv::imread(input_path);
        if(image.empty()){
            return 1;
        }
        image.convertTo(image,CV_32F);
        if(image.cols<window_x || image.rows<window_y){
            return 2;
        }
        current_type=RGB;
    }


    Descriptor *descriptor;
    if(descriptor_type==RGB){
        descriptor=0;
    }
    else if(descriptor_type==GRAY){
        Basic_Transformations *basic=new Basic_Transformations(current_type,GRAY);
        descriptor=basic;
    }
    else if(descriptor_type==HSV){
        Basic_Transformations *basic=new Basic_Transformations(current_type,HSV);
        descriptor=basic;
    }
    else if(descriptor_type==H_CHANNEL){
        Basic_Transformations *basic=new Basic_Transformations(current_type,H_CHANNEL);
        descriptor=basic;
    }
    else if(descriptor_type==S_CHANNEL){
        Basic_Transformations *basic=new Basic_Transformations(current_type,S_CHANNEL);
        descriptor=basic;
    }
    else if(descriptor_type==V_CHANNEL){
        Basic_Transformations *basic=new Basic_Transformations(current_type,V_CHANNEL);
        descriptor=basic;
    }
    else if(descriptor_type==THRESHOLD){
        Basic_Transformations *basic=new Basic_Transformations(current_type,THRESHOLD);
        descriptor=basic;
    }
    else if(descriptor_type==CANNY){
        Basic_Transformations *basic=new Basic_Transformations(current_type,CANNY);
        descriptor=basic;
    }
    else if(descriptor_type==SOBEL){
        Basic_Transformations *basic=new Basic_Transformations(current_type,SOBEL);
        descriptor=basic;
    }
    else if(descriptor_type==COLOR_PREDOMINANTE){
        Basic_Transformations *basic=new Basic_Transformations(current_type,COLOR_PREDOMINANTE);
        descriptor=basic;
    }
    else if(descriptor_type==HOG_DES){
        cv::Size size=cv::Size(size_x,size_y);
        cv::Size block=cv::Size(block_x,block_y);
        if(size.height>window_y || size.width>window_x){
            return 3;
        }
        HOG *hog=new HOG(size,block, sigma, threhold_l2hys, gamma, n_levels);
        descriptor= hog;
    }
    else if(descriptor_type==PUNTOS_CARACTERISTICOS){
        Puntos_Caracteristicos *des= new Puntos_Caracteristicos(pc_descriptor,extractor,descriptor_parameter);
        descriptor= des;
    }
    else{
        return 4;
    }

    if(type_running==1){
        if(this->classifier->nombre!=""){
            Busqueda bus(this->classifier,current_type,descriptor);
            std::thread thrd(&MLT::Busqueda::Posicion,&bus,image,cv::Size(window_x,window_y),pyramid,jump,rotation,postprocess,overlap,isolated,dist_boxes,dist_rotation,std::ref(detections),std::ref(labels_detections));
            thrd.join();

            if(bus.error==1){
                return 4;
            }
        }
    }
    else if(type_running==2){
        Busqueda bus(this->classifier,current_type,descriptor);
        std::thread thrd(&MLT::Busqueda::Textura,&bus,image,cv::Size(window_x,window_y),pyramid,jump,rotation,postprocess,std::ref(output));
        thrd.join();

        if(bus.error==1){
            return 4;
        }
    }
    else if(type_running==3 || type_running==4){
        if(multi_params.identificadores.empty())
            return 4;
        vector<Clasificador*> classifiers;
        for(uint i=0; i<multi_params.identificadores.size(); i++){
            if(multi_params.identificadores[i]==DISTANCIAS){
                Clasificador_Distancias *classifier=new Clasificador_Distancias(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==GAUSSIANO){
                Clasificador_Gaussiano *classifier=new Clasificador_Gaussiano(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==CASCADA_CLAS){
                Clasificador_Cascada *classifier=new Clasificador_Cascada(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==HISTOGRAMA){
                Clasificador_Histograma *classifier=new Clasificador_Histograma(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==KNN){
                Clasificador_KNN *classifier=new Clasificador_KNN(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==NEURONAL){
                Clasificador_Neuronal *classifier=new Clasificador_Neuronal(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==C_SVM){
                Clasificador_SVM *classifier=new Clasificador_SVM(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==RTREES){
                Clasificador_RTrees *classifier=new Clasificador_RTrees(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==DTREES){
                Clasificador_DTrees *classifier=new Clasificador_DTrees(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==BOOSTING){
                Clasificador_Boosting *classifier=new Clasificador_Boosting(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==EXP_MAX){
                Clasificador_EM *classifier=new Clasificador_EM(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
    //        else if(multi_params.identificadores[i]==GBT){
    //            Clasificador_GBTrees *classifier=new Clasificador_GBTrees(multi_params.nombres[i]);
    //                classifier->Read_Data();
    //                classifiers.push_back(classifier);
    //        }
    //        else if(multi_params.identificadores[i]==ERTREES){
    //            Clasificador_ERTrees *classifier=new Clasificador_ERTrees(multi_params.nombres[i]);
    //                classifier->Read_Data();
    //                classifiers.push_back(classifier);
    //        }
            MultiClasificador multi(classifiers);
            multi.progreso=0;
            this->max_progreso=100;
            this->base_progreso=1;
            multi.total_progreso=this->org_images.size();

            if(type_running==3){
                Busqueda bus(&multi,current_type,descriptor,&multi_params);
                std::thread thrd(&MLT::Busqueda::Posicion,&bus,image,cv::Size(window_x,window_y),pyramid,jump,rotation,postprocess,overlap,isolated,dist_boxes,dist_rotation,std::ref(detections),std::ref(labels_detections));
                thrd.join();

                if(bus.error==1){
                    return 4;
                }
            }
            else if(type_running==4){
                Busqueda bus(&multi,current_type,descriptor,&multi_params);
                std::thread thrd(&MLT::Busqueda::Textura,&bus,image,cv::Size(window_x,window_y),pyramid,jump,rotation,postprocess,std::ref(output));
                thrd.join();

                if(bus.error==1){
                    return 4;
                }
            }
        }
    }

    image.convertTo(image,CV_32F);
    double minval,maxval;
    cv::minMaxLoc(image,&minval,&maxval);
    image=(image-minval)/(maxval-minval);

    return 0;
}

int MLT::Running::train(string ref, int classifier_type, Clasificadores::Parametros params){
    if(this->org_images.empty()){
        return 1;
    }
    for(uint i=0; i<ref.size(); i++){
        if(ref[i]==' '){
            return 1;
        }
    }
//    delete this->classifier;
    this->classifier=new Clasificador;
    int e=0;
    if(classifier_type==1){
        Clasificador_Distancias *classifier= new MLT::Clasificador_Distancias;
        classifier->nombre=ref;
        e=classifier->Parametrizar();
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_Distancias::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else if(classifier_type==2){
        Clasificador_Gaussiano *classifier= new MLT::Clasificador_Gaussiano;
        classifier->nombre=ref;
        e=classifier->Parametrizar();
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_Gaussiano::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else if(classifier_type==3){
        Clasificador_Histograma *classifier= new MLT::Clasificador_Histograma;
        classifier->nombre=ref;
        e=classifier->Parametrizar(params.Hist_tam_celda);
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_Histograma::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else if(classifier_type==4){
        Clasificador_KNN *classifier= new MLT::Clasificador_KNN;
        classifier->nombre=ref;
        e=classifier->Parametrizar(params.KNN_k,params.KNN_regression);
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_KNN::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else if(classifier_type==5){
        if(params.Neuronal_layerSize.rows<3){
            return 1;
        }
        params.Neuronal_layerSize.row(0)=this->org_images[0].cols*this->org_images[0].rows*this->org_images[0].channels();
        Auxiliares aux;
        bool negativa;
        int numero=aux.numero_etiquetas(this->org_labels,negativa);
        params.Neuronal_layerSize.row(params.Neuronal_layerSize.rows-1)=numero;

        Clasificador_Neuronal *classifier= new MLT::Clasificador_Neuronal;
        classifier->nombre=ref;
        e=classifier->Parametrizar(params.Neuronal_layerSize,params.Neuronal_Method,params.Neuronal_Function,params.Neuronal_bp_dw_scale,params.Neuronal_bp_moment_scale,
                                  params.Neuronal_rp_dw0,params.Neuronal_rp_dw_max,params.Neuronal_rp_dw_min,params.Neuronal_rp_dw_minus,params.Neuronal_rp_dw_plus,
                                  params.Neuronal_fparam1,params.Neuronal_fparam2);
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_Neuronal::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else if(classifier_type==6){
        Clasificador_SVM *classifier= new MLT::Clasificador_SVM;
        classifier->nombre=ref;
        e=classifier->Parametrizar(params.SVM_train,params.SVM_Type,params.SVM_kernel_type,Mat(),params.SVM_degree,params.SVM_gamma,params.SVM_coef0,params.SVM_C,params.SVM_nu,params.SVM_p);
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_SVM::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else if(classifier_type==7){
        Clasificador_DTrees *classifier= new MLT::Clasificador_DTrees;
        classifier->nombre=ref;
        e=classifier->Parametrizar(params.DTrees_max_depth,params.DTrees_min_sample_count,params.DTrees_regression_accuracy,params.DTrees_use_surrogates,params.DTrees_max_categories,
                                  params.DTrees_cv_folds,params.DTrees_use_1se_rule,params.DTrees_truncate_pruned_tree,params.DTrees_priors);
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_DTrees::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else if(classifier_type==8){
        Clasificador_RTrees *classifier= new MLT::Clasificador_RTrees;
        classifier->nombre=ref;
        e=classifier->Parametrizar(params.RTrees_max_depth,params.RTrees_min_sample_count,params.RTrees_regression_accuracy,params.RTrees_use_surrogates,params.RTrees_max_categories,
                                  params.RTrees_cv_folds,params.RTrees_use_1se_rule,params.RTrees_truncate_pruned_tree,params.RTrees_priors,params.RTrees_calc_var_importance,
                                  params.RTrees_native_vars);
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_RTrees::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else if(classifier_type==9){
        Auxiliares aux;
        bool negative;
        if(aux.numero_etiquetas(this->org_labels,negative)!=2){
            return 1;
        }
        Clasificador_Boosting *classifier= new MLT::Clasificador_Boosting;
        classifier->nombre=ref;
        e=classifier->Parametrizar(params.Boosting_boost_type,params.Boosting_weak_count,params.Boosting_weight_trim_rate,params.Boosting_max_depth,params.Boosting_use_surrogates,
                                  params.Boosting_priors);
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_Boosting::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else if(classifier_type==10){
        Clasificador_Cascada *classifier= new MLT::Clasificador_Cascada;
        classifier->nombre=ref;
        e=classifier->Parametrizar("HAAR",true,params.Cascada_NumPos,params.Cascada_NumNeg,params.Cascada_Mode,params.Cascada_NumStage,params.Cascada_MinHitRate,
                                  params.Cascada_MaxFalseAlarmRate,params.Cascada_WeightTrimRate,params.Cascada_MaxWeakCount,params.Cascada_MaxDepth,params.Cascada_Bt,
                                  params.Cascada_PrecalcValBufSize,params.Cascada_PrecalcidxBufSize);
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_Cascada::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else if(classifier_type==11){
        Clasificador_Cascada *classifier= new MLT::Clasificador_Cascada;
        classifier->nombre=ref;
        e=classifier->Parametrizar("LBP",true,params.Cascada_NumPos,params.Cascada_NumNeg,params.Cascada_Mode,params.Cascada_NumStage,params.Cascada_MinHitRate,
                                  params.Cascada_MaxFalseAlarmRate,params.Cascada_WeightTrimRate,params.Cascada_MaxWeakCount,params.Cascada_MaxDepth,params.Cascada_Bt,
                                  params.Cascada_PrecalcValBufSize,params.Cascada_PrecalcidxBufSize);
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_Cascada::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else if(classifier_type==12){
        Clasificador_EM *classifier= new MLT::Clasificador_EM;
        classifier->nombre=ref;
        e=classifier->Parametrizar(params.EM_nclusters,params.EM_covMatType);
        Dimensionalidad::Reducciones reduc;
        std::thread thrd(&MLT::Clasificador_EM::Autotrain,classifier,this->org_images,this->org_labels,reduc,this->org_info,this->save_clasif);
        thrd.join();
        this->classifier=classifier;
    }
    else
        return 1;
    if(e==1)
        return 1;
    return 0;
}

int MLT::Running::load_model(string path, string &name){
//    delete this->classifier;
    this->classifier=new Clasificador;
    int pos=0;
    for(uint i=0; i<path.size(); i++){
        if(path[i]=='/')
            pos=i;
    }
    name.clear();
    for(uint i=pos+1; i<path.size(); i++)
        name=name+path[i];
    string archivo=path+"/Clasificador.xml";
    cv::FileStorage archivo_r(archivo,FileStorage::READ);
    int id;
    if(archivo_r.isOpened()){
        archivo_r["Tipo"]>>id;
    }
    else
        return 1;

    int e=0;
    if(id==DISTANCIAS){
        MLT::Clasificador_Distancias *classifier= new MLT::Clasificador_Distancias;
        classifier->nombre=name;
        e=classifier->Read_Data();
        this->classifier=classifier;
    }
    else if(id==GAUSSIANO){
        MLT::Clasificador_Gaussiano *classifier= new MLT::Clasificador_Gaussiano;
        classifier->nombre=name;
        e=classifier->Read_Data();
        this->classifier=classifier;
    }
    else if(id==CASCADA_CLAS){
        MLT::Clasificador_Cascada *classifier= new MLT::Clasificador_Cascada;
        classifier->nombre=name;
        classifier->Read_Data();
        this->classifier=classifier;
    }
    else if(id==HISTOGRAMA){
        MLT::Clasificador_Histograma *classifier= new MLT::Clasificador_Histograma;
        classifier->nombre=name;
        e=classifier->Read_Data();
        this->classifier=classifier;
    }
    else if(id==KNN){
        MLT::Clasificador_KNN *classifier= new MLT::Clasificador_KNN;
        classifier->nombre=name;
        e=classifier->Read_Data();
        this->classifier=classifier;
    }
    else if(id==NEURONAL){
        MLT::Clasificador_Neuronal *classifier= new MLT::Clasificador_Neuronal;
        classifier->nombre=name;
        e=classifier->Read_Data();
        this->classifier=classifier;
    }
    else if(id==C_SVM){
        MLT::Clasificador_SVM *classifier= new MLT::Clasificador_SVM;
        classifier->nombre=name;
        e=classifier->Read_Data();
        this->classifier=classifier;
    }
    else if(id==RTREES){
        MLT::Clasificador_RTrees *classifier= new MLT::Clasificador_RTrees;
        classifier->nombre=name;
        e=classifier->Read_Data();
        this->classifier=classifier;
    }
    else if(id==DTREES){
        MLT::Clasificador_DTrees *classifier= new MLT::Clasificador_DTrees;
        classifier->nombre=name;
        e=classifier->Read_Data();
        this->classifier=classifier;
    }
    else if(id==BOOSTING){
        MLT::Clasificador_Boosting *classifier= new MLT::Clasificador_Boosting;
        classifier->nombre=name;
        e=classifier->Read_Data();
        this->classifier=classifier;
    }
    else if(id==EXP_MAX){
        MLT::Clasificador_EM *classifier= new MLT::Clasificador_EM;
        classifier->nombre=name;
        e=classifier->Read_Data();
        this->classifier=classifier;
    }
    else
        return 1;
    if(e==1)
        return 1;
    return 0;
}

int MLT::Running::classify(string ref, int type_classification, stringstream &txt, MultiClasificador::Multi_type multi_params){
//    if(this->org_images.empty()){
//        return 1;
//    }
//    for(uint i=0; i<ref.size(); i++){
//        if(ref[i]==' '){
//            return 2;
//        }
//    }
    int e=0;

    this->result_labels.clear();
    if(type_classification==1){
        if(this->classifier->nombre=="")
            return 1;
        this->classifier->progreso=0;
        this->max_progreso=100;
        this->base_progreso=0;
        this->classifier->total_progreso=this->org_images.size();
        std::thread thrd(&MLT::Clasificador::Autoclasificacion,this->classifier,this->org_images,std::ref(this->result_labels),this->ifreduc,this->read);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        while(this->classifier->running==true)
            update_classifier(this->classifier->progreso,this->classifier->total_progreso);
        thrd.join();
    }
    else if(type_classification==2){
        if(multi_params.identificadores.empty())
            return 1;
        vector<Clasificador*> classifiers;
        for(uint i=0; i<multi_params.identificadores.size(); i++){
            if(multi_params.identificadores[i]==DISTANCIAS){
                Clasificador_Distancias *classifier=new Clasificador_Distancias(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==GAUSSIANO){
                Clasificador_Gaussiano *classifier=new Clasificador_Gaussiano(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==CASCADA_CLAS){
                Clasificador_Cascada *classifier=new Clasificador_Cascada(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==HISTOGRAMA){
                Clasificador_Histograma *classifier=new Clasificador_Histograma(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==KNN){
                Clasificador_KNN *classifier=new Clasificador_KNN(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==NEURONAL){
                Clasificador_Neuronal *classifier=new Clasificador_Neuronal(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==C_SVM){
                Clasificador_SVM *classifier=new Clasificador_SVM(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==RTREES){
                Clasificador_RTrees *classifier=new Clasificador_RTrees(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==DTREES){
                Clasificador_DTrees *classifier=new Clasificador_DTrees(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==BOOSTING){
                Clasificador_Boosting *classifier=new Clasificador_Boosting(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
            else if(multi_params.identificadores[i]==EXP_MAX){
                Clasificador_EM *classifier=new Clasificador_EM(multi_params.nombres[i]);
                classifier->Read_Data();
                classifiers.push_back(classifier);
            }
    //        else if(multi_params.identificadores[i]==GBT){
    //            Clasificador_GBTrees *classifier=new Clasificador_GBTrees(multi_params.nombres[i]);
    //                classifier->Read_Data();
    //                classifiers.push_back(classifier);
    //        }
    //        else if(multi_params.identificadores[i]==ERTREES){
    //            Clasificador_ERTrees *classifier=new Clasificador_ERTrees(multi_params.nombres[i]);
    //                classifier->Read_Data();
    //                classifiers.push_back(classifier);
    //        }
        }
        MultiClasificador multi(classifiers);
        multi.progreso=0;
        this->max_progreso=100;
        this->base_progreso=1;
        multi.total_progreso=this->org_images.size();
        if(multi_params.tipo==CASCADA){
            std::thread thrd(&MLT::MultiClasificador::Cascada,multi,this->org_images,multi_params.tipo_regla,multi_params.label_ref,std::ref(this->result_labels));
            bool running=true;
            while(running==true){
                int progress=0;
                running=false;
                for(int j=0; j<classifiers.size(); j++){
                    progress=progress+classifiers[j]->progreso;
                    if(classifiers[j]->running==true)
                        running=true;
                }
                progress=progress/classifiers.size();
                update_classifier(progress,multi.total_progreso);
            }
            thrd.join();
        }
        else if(multi_params.tipo==VOTACION){
            std::thread thrd(&MLT::MultiClasificador::Votacion,multi,this->org_images,multi_params.w_clasif,std::ref(this->result_labels));
            bool running=true;
            while(running==true){
                int progress=0;
                running=false;
                for(int j=0; j<classifiers.size(); j++){
                    progress=progress+classifiers[j]->progreso;
                    if(classifiers[j]->running==true)
                        running=true;
                }
                progress=progress/classifiers.size();
                update_classifier(progress,multi.total_progreso);
            }
            thrd.join();
        }
    }

    if(this->org_labels.size()==this->result_labels.size()){
        txt<<"N Data      Original Label      Result"<<endl;
        for(uint i=0; i<this->result_labels.size(); i++){
            txt<<i+1<<"                               "<<this->org_labels[i]<<"                       "<<this->result_labels[i]<<endl;
            if(this->result_labels[i]==0)
                e=2;
        }
    }
    else{
        txt<<"N Data      Result"<<endl;
        for(uint i=0; i<this->result_labels.size(); i++){
            txt<<i+1<<"                       "<<this->result_labels[i]<<endl;
            if(this->result_labels[i]==0)
                e=2;
        }
    }

    this->result_images.clear();
    for(int i=0; i<this->org_images.size(); i++){
        cv::Mat image;
        this->org_images[i].copyTo(image);
        this->result_images.push_back(image);
    }
    return 0;
}

int MLT::Running::optimize(int type, int id_classifier, MultiClasificador::Multi_type multi_type, stringstream &text,
                           Clasificadores::Parametros start, Clasificadores::Parametros leap, Clasificadores::Parametros stop,
                           int percentage, int num_folds, int size_fold){
    if(this->org_images.empty())
        return 1;
    if(this->org_labels.empty())
        return 2;
    Optimizacion op;
    op.total_progreso=this->org_images.size();
    op.progreso=0;
    this->base_progreso=0;
    this->max_progreso=100;

    Auxiliares aux;
    bool negative;
    int num=aux.numero_etiquetas(this->org_labels,negative);

    float error;
    Mat confusion;
    if(type==1 || type==2){
        vector<Analisis::Ratios_data> rates;
        if(type==1){

            if(id_classifier==NEURONAL){
                if(start.Neuronal_layerSize.rows<3){
                    return 3;
                }
                start.Neuronal_layerSize.row(0)=this->org_images[0].cols*this->org_images[0].rows*this->org_images[0].channels();
                start.Neuronal_layerSize.row(start.Neuronal_layerSize.rows-1)=num;
            }
            std::thread thrd(&MLT::Optimizacion::Validation,&op,this->org_images,this->org_labels,percentage,id_classifier,start,std::ref(error),std::ref(confusion),std::ref(rates));
            thrd.join();

            if(op.error==1)
                return 4;
        }
        else if(type==2){
            std::thread thrd(&MLT::Optimizacion::Validation_multi,&op, this->org_images,this->org_labels,percentage,multi_type.identificadores,start,multi_type,std::ref(error),std::ref(confusion),std::ref(rates));
            thrd.join();
            if(op.error==1)
                return 4;
        }
        text<<"Error=";
        text<<error;
        text<<endl;
        text<<"Confusion="<<endl;
        for(int i=0; i<confusion.cols; i++){
            for(int j=0; j<confusion.rows; j++){
                text<<confusion.at<float>(j,i);
                text<<"    ";
            }
            text<<endl;
        }

        for(int i=0; i<num; i++){
            int etiqueta;
            if(negative){
                if(i==0)
                    etiqueta=-1;
                else
                    etiqueta=i;
            }
            else
                etiqueta=i+1;
            text<<"Etiqueta "<<etiqueta<<":"<<endl;
            text<<"VP="<<rates[i].VP;
            text<<"    VN="<<rates[i].VN;
            text<<"    FP="<<rates[i].FP;
            text<<"    FN="<<rates[i].FN;
            text<<"    TAR="<<rates[i].TAR;
            text<<"    TRR="<<rates[i].TRR;
            text<<"    FAR="<<rates[i].FAR;
            text<<"    FRR="<<rates[i].FRR;
            text<<"    PPV="<<rates[i].PPV;
            text<<"    NPV="<<rates[i].NPV;
            text<<"    FDR="<<rates[i].FDR;
            text<<"    F1="<<rates[i].F1;
            text<<"    INFORMEDNESS="<<rates[i].INFORMEDNESS;
            text<<"    MARKEDNESS="<<rates[i].MARKEDNESS;
            text<<"    EXP_ERROR="<<rates[i].EXP_ERROR;
            text<<"    LR_POS="<<rates[i].LR_POS;
            text<<"    LR_NEG="<<rates[i].LR_NEG;
            text<<"    DOR="<<rates[i].DOR;
            text<<"    ACC="<<rates[i].ACC;
            text<<"    PREVALENCE="<<rates[i].PREVALENCE<<endl;
        }
    }

    else if(type==3){


        if(num_folds*size_fold>this->org_images.size())
            return 1;
        Clasificadores::Parametros parameters;
        std::thread thrd(&MLT::Optimizacion::Cross_Validation,&op, this->org_images,this->org_labels,num_folds,size_fold,id_classifier,start,stop,leap,std::ref(parameters),std::ref(error),std::ref(confusion));
        thrd.join();
        if(op.error==1){
            return 4;
        }
        if(id_classifier==DISTANCIAS){
            return 5;
        }
        else if(id_classifier==GAUSSIANO){
            return 5;
        }
        else if(id_classifier==CASCADA_CLAS){
//            QMessageBox msgBox;
//            msgBox.setText("ERROR: No implementado");
//            msgBox.exec();
//            QApplication::restoreOverrideCursor();
//            ui->progress_Clasificar->setValue(0);
            return 6;
        }
        else if(id_classifier==HISTOGRAMA){
            text<<"Best Parameters"<<endl;
            text<<"Hist_tam_celdea= "<<parameters.Hist_tam_celda<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(id_classifier==KNN){
            text<<"Best Parameters"<<endl;
            text<<"KNN_k= "<<parameters.KNN_k<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(id_classifier==NEURONAL){
            text<<"Best Parameters"<<endl;
            text<<"Neuronal_bp_dw_scale= "<<parameters.Neuronal_bp_dw_scale<<endl;
            text<<"Neuronal_bp_moment_scale= "<<parameters.Neuronal_bp_moment_scale<<endl;
            text<<"Neuronal_rp_dw0= "<<parameters.Neuronal_rp_dw0<<endl;
            text<<"Neuronal_rp_dw_max= "<<parameters.Neuronal_rp_dw_max<<endl;
            text<<"Neuronal_rp_dw_min= "<<parameters.Neuronal_rp_dw_min<<endl;
            text<<"Neuronal_rp_dw_minus= "<<parameters.Neuronal_rp_dw_minus<<endl;
            text<<"Neuronal_rp_dw_plus= "<<parameters.Neuronal_rp_dw_plus<<endl;
            text<<"Neuronal_fparam1= "<<parameters.Neuronal_fparam1<<endl;
            text<<"Neuronal_fparam2= "<<parameters.Neuronal_fparam2<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(id_classifier==C_SVM){
            text<<"Best Parameters"<<endl;
            text<<"SVM_C= "<<parameters.SVM_C<<endl;
            text<<"SVM_gamma= "<<parameters.SVM_gamma<<endl;
            text<<"SVM_p= "<<parameters.SVM_p<<endl;
            text<<"SVM_nu= "<<parameters.SVM_nu<<endl;
            text<<"SVM_coef0= "<<parameters.SVM_coef0<<endl;
            text<<"SVM_degree= "<<parameters.SVM_degree<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(id_classifier==RTREES){
            text<<"Best Parameters"<<endl;
            text<<"RTrees_max_depth= "<<parameters.RTrees_max_depth<<endl;
            text<<"RTrees_min_sample_count= "<<parameters.RTrees_min_sample_count<<endl;
            text<<"RTrees_regression_accuracy= "<<parameters.RTrees_regression_accuracy<<endl;
            text<<"RTrees_max_categories= "<<parameters.RTrees_max_categories<<endl;
            text<<"RTrees_cv_folds= "<<parameters.RTrees_cv_folds<<endl;
            text<<"RTrees_native_vars= "<<parameters.RTrees_native_vars<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(id_classifier==DTREES){
            text<<"Best Parameters"<<endl;
            text<<"DTrees_max_depth= "<<parameters.DTrees_max_depth<<endl;
            text<<"DTrees_min_sample_count= "<<parameters.DTrees_min_sample_count<<endl;
            text<<"DTrees_regression_accuracy= "<<parameters.DTrees_regression_accuracy<<endl;
            text<<"DTrees_max_categories= "<<parameters.DTrees_max_categories<<endl;
            text<<"DTrees_cv_folds= "<<parameters.DTrees_cv_folds<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(id_classifier==BOOSTING){
            text<<"Best Parameters"<<endl;
            text<<"Boosting_max_depth= "<<parameters.Boosting_max_depth<<endl;
            text<<"Boosting_weak_count= "<<parameters.Boosting_weak_count<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(id_classifier==EXP_MAX){
            text<<"Best Parameters"<<endl;
            text<<"EM_nclusters= "<<parameters.EM_nclusters<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
//        else if(id_classifier==GBT){
//            text<<"Best Parameters"<<endl;
//            text<<"GBT_weak_count= "<<parameters.GBT_weak_count<<endl;
//            text<<"GBT_shrinkage= "<<parameters.GBT_shrinkage<<endl;
//            text<<"GBT_max_depth= "<<parameters.GBT_max_depth<<endl;
//            text<<"Error= "<<error<<endl;
//            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
//        }
//        else if(id_classifier==ERTREES){
//            text<<"Best Parameters"<<endl;
//            text<<"ERTrees_max_depth= "<<parameters.ERTrees_max_depth<<endl;
//            text<<"ERTrees_min_sample_count= "<<parameters.ERTrees_min_sample_count<<endl;
//            text<<"ERTrees_regression_accuracy= "<<parameters.ERTrees_regression_accuracy<<endl;
//            text<<"ERTrees_max_categories= "<<parameters.ERTrees_max_categories<<endl;
//            text<<"ERTrees_cv_folds= "<<parameters.ERTrees_cv_folds<<endl;
//            text<<"ERTrees_native_vars= "<<parameters.ERTrees_native_vars<<endl;
//            text<<"Error= "<<error<<endl;
//            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
//        }
    }
    else if(type==4){
        if(num_folds*size_fold>this->org_images.size()){
            return 7;
        }
        Clasificadores::Parametros parameters;
        std::thread thrd(&MLT::Optimizacion::Super_Cross_Validation,&op, this->org_images,this->org_labels,num_folds,size_fold,std::ref(multi_type.identificadores),start,stop,leap,std::ref(parameters),std::ref(error),std::ref(confusion));
        thrd.join();
        if(op.error==1){
            return 4;
        }

        if(multi_type.identificadores[0]==DISTANCIAS){
            text<<"El mejor clasificador es Clasificador_Distancias"<<endl;
        }
        else if(multi_type.identificadores[0]==GAUSSIANO){
            text<<"El mejor clasificador es Clasificador_Gaussiano"<<endl;
        }
        else if(multi_type.identificadores[0]==CASCADA_CLAS)
            return 6;
        else if(multi_type.identificadores[0]==HISTOGRAMA){
            text<<"El mejor clasificador es Clasificador_Histograma"<<endl;
            text<<"Best Parameters"<<endl;
            text<<"Hist_tam_celdea= "<<parameters.Hist_tam_celda<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(multi_type.identificadores[0]==KNN){
            text<<"El mejor clasificador es Clasificador_KNN"<<endl;
            text<<"Best Parameters"<<endl;
            text<<"KNN_k= "<<parameters.KNN_k<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(multi_type.identificadores[0]==NEURONAL){
            text<<"El mejor clasificador es Clasificador_Neuronal"<<endl;
            text<<"Best Parameters"<<endl;
            text<<"Neuronal_bp_dw_scale= "<<parameters.Neuronal_bp_dw_scale<<endl;
            text<<"Neuronal_bp_moment_scale= "<<parameters.Neuronal_bp_moment_scale<<endl;
            text<<"Neuronal_rp_dw0= "<<parameters.Neuronal_rp_dw0<<endl;
            text<<"Neuronal_rp_dw_max= "<<parameters.Neuronal_rp_dw_max<<endl;
            text<<"Neuronal_rp_dw_min= "<<parameters.Neuronal_rp_dw_min<<endl;
            text<<"Neuronal_rp_dw_minus= "<<parameters.Neuronal_rp_dw_minus<<endl;
            text<<"Neuronal_rp_dw_plus= "<<parameters.Neuronal_rp_dw_plus<<endl;
            text<<"Neuronal_fparam1= "<<parameters.Neuronal_fparam1<<endl;
            text<<"Neuronal_fparam2= "<<parameters.Neuronal_fparam2<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(multi_type.identificadores[0]==C_SVM){
            text<<"El mejor clasificador es Clasificador_SVM"<<endl;
            text<<"Best Parameters"<<endl;
            text<<"SVM_C= "<<parameters.SVM_C<<endl;
            text<<"SVM_gamma= "<<parameters.SVM_gamma<<endl;
            text<<"SVM_p= "<<parameters.SVM_p<<endl;
            text<<"SVM_nu= "<<parameters.SVM_nu<<endl;
            text<<"SVM_coef0= "<<parameters.SVM_coef0<<endl;
            text<<"SVM_degree= "<<parameters.SVM_degree<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(multi_type.identificadores[0]==RTREES){
            text<<"El mejor clasificador es Clasificador_RTrees"<<endl;
            text<<"Best Parameters"<<endl;
            text<<"RTrees_max_depth= "<<parameters.RTrees_max_depth<<endl;
            text<<"RTrees_min_sample_count= "<<parameters.RTrees_min_sample_count<<endl;
            text<<"RTrees_regression_accuracy= "<<parameters.RTrees_regression_accuracy<<endl;
            text<<"RTrees_max_categories= "<<parameters.RTrees_max_categories<<endl;
            text<<"RTrees_cv_folds= "<<parameters.RTrees_cv_folds<<endl;
            text<<"RTrees_native_vars= "<<parameters.RTrees_native_vars<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(multi_type.identificadores[0]==DTREES){
            text<<"El mejor clasificador es Clasificador_DTrees"<<endl;
            text<<"Best Parameters"<<endl;
            text<<"DTrees_max_depth= "<<parameters.DTrees_max_depth<<endl;
            text<<"DTrees_min_sample_count= "<<parameters.DTrees_min_sample_count<<endl;
            text<<"DTrees_regression_accuracy= "<<parameters.DTrees_regression_accuracy<<endl;
            text<<"DTrees_max_categories= "<<parameters.DTrees_max_categories<<endl;
            text<<"DTrees_cv_folds= "<<parameters.DTrees_cv_folds<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(multi_type.identificadores[0]==BOOSTING){
            text<<"El mejor clasificador es Clasificador_Boosting"<<endl;
            text<<"Best Parameters"<<endl;
            text<<"Boosting_max_depth= "<<parameters.Boosting_max_depth<<endl;
            text<<"Boosting_weak_count= "<<parameters.Boosting_weak_count<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
        else if(multi_type.identificadores[0]==EXP_MAX){
            text<<"El mejor clasificador es Clasificador_EM"<<endl;
            text<<"Best Parameters"<<endl;
            text<<"EM_nclusters= "<<parameters.EM_nclusters<<endl;
            text<<"Error= "<<error<<endl;
            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
        }
//        else if(multi_type.identificadores[0]==GBT){
//            cout<<"El mejor clasificador es Clasificador_GBT"<<endl;
//            text<<"Best Parameters"<<endl;
//            text<<"GBT_weak_count= "<<parameters.GBT_weak_count<<endl;
//            text<<"GBT_shrinkage= "<<parameters.GBT_shrinkage<<endl;
//            text<<"GBT_max_depth= "<<parameters.GBT_max_depth<<endl;
//            text<<"Error= "<<error<<endl;
//            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
//        }
//        else if(multi_type.identificadores[0]==ERTREES){
//            text<<"El mejor clasificador es Clasificador_ERTrees"<<endl;
//            text<<"Best Parameters"<<endl;
//            text<<"ERTrees_max_depth= "<<parameters.ERTrees_max_depth<<endl;
//            text<<"ERTrees_min_sample_count= "<<parameters.ERTrees_min_sample_count<<endl;
//            text<<"ERTrees_regression_accuracy= "<<parameters.ERTrees_regression_accuracy<<endl;
//            text<<"ERTrees_max_categories= "<<parameters.ERTrees_max_categories<<endl;
//            text<<"ERTrees_cv_folds= "<<parameters.ERTrees_cv_folds<<endl;
//            text<<"ERTrees_native_vars= "<<parameters.ERTrees_native_vars<<endl;
//            text<<"Error= "<<error<<endl;
//            text<<"Matriz Confusion= "<<endl<<confusion<<endl;
//        }
    }
    else
        return 4;
    text<<endl;
    return 0;
}
