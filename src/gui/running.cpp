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
    std::thread thrd(&MLT::Generacion::Cargar_Fichero,&gen,input_directory,std::ref(this->org_images),std::ref(this->org_labels),std::ref(this->org_info));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    while(gen.running==true)
        update_gen();

    thrd.join();

    if(gen.error==1)
        return 2;

    return 0;
}

int MLT::Running::synthetic_data(string ref, int num_clases, int num_data_clase, int vector_size, float ancho, float separacion_clases){
    Size size_img;
    size_img.width=vector_size;
    size_img.height=1;


    this->base_progreso=1;
    this->max_progreso=100;

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
    QStandardItem *Rati= new QStandardItem(QString("Ratios"));
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
        QStandardItem *Lab = new QStandardItem(QString("Etiqueta %1").arg(etiqueta));
        QStandardItem *VP = new QStandardItem(QString("VP"));
        QStandardItem *vp = new QStandardItem(QString("%1").arg(rates[i].VP));
        VP->appendRow(vp);
        Lab->appendRow(VP);
        QStandardItem *VN = new QStandardItem(QString("VN"));
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
