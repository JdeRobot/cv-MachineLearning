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

#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    string g="../Data/Config_Data/Initial.xml";
    cv::FileStorage fs(g,CV_STORAGE_READ);
    fs["Colors"]>>Col;

    fs["Tipo_Des"]>>Tipo_Des;
    fs["Tipo_Ext"]>>Tipo_Ext;
    fs["Parametro"]>>Parametro;

//    fs["Win_Size"]>>Win_Size;
//    fs["Block_Stride"]>>Block_Stride;
    Win_Size=Size(64, 128);
    Block_Stride=Size(8, 8);
    fs["Win_Sigma"]>>Win_Sigma;
    fs["Threshold_L2hys"]>>Threshold_L2hys;
    fs["Gamma_Correction"]>>Gamma_Correction;
    fs["Nlevels"]>>Nlevels;

    fs["ID"]>>ID;

    fs["Hist_tam_celda"]>>Hist_tam_celda;

    fs["KNN_k"]>>KNN_k;
    fs["KNN_regression"]>>KNN_regression;

//    fs["Cascada_Tam"]>>Cascada_Tam;
    Cascada_Tam=Size(24,24);
    fs["Cascada_NumPos"]>>Cascada_NumPos;
    fs["Cascada_NumNeg"]>>Cascada_NumNeg;
    fs["Cascada_Mode"]>>Cascada_Mode;
    fs["Cascada_NumStage"]>>Cascada_NumStage;
    fs["Cascada_MinHitRate"]>>Cascada_MinHitRate;
    fs["Cascada_MaxFalseAlarmRate"]>>Cascada_MaxFalseAlarmRate;
    fs["Cascada_WeightTrimRate"]>>Cascada_WeightTrimRate;
    fs["Cascada_MaxWeakCount"]>>Cascada_MaxWeakCount;
    fs["Cascada_MaxDepth"]>>Cascada_MaxDepth;
    fs["Cascada_Bt"]>>Cascada_Bt;
    fs["Cascada_PrecalcValBufSize"]>>Cascada_PrecalcValBufSize;
    fs["Cascada_PrecalcidxBufSize"]>>Cascada_PrecalcidxBufSize;
    fs["si_entrenar"]>>si_entrenar;

    fs["Neuronal_Method"]>>Neuronal_Method;
    fs["Neuronal_Function"]>>Neuronal_Function;
    fs["Neuronal_bp_dw_scale"]>>Neuronal_bp_dw_scale;
    fs["Neuronal_bp_moment_scale"]>>Neuronal_bp_moment_scale;
    fs["Neuronal_rp_dw0"]>>Neuronal_rp_dw0;
    fs["Neuronal_rp_dw_max"]>>Neuronal_rp_dw_max;
    fs["Neuronal_rp_dw_min"]>>Neuronal_rp_dw_min;
    fs["Neuronal_rp_dw_minus"]>>Neuronal_rp_dw_minus;
    fs["Neuronal_rp_dw_plus"]>>Neuronal_rp_dw_plus;
    fs["Neuronal_fparam1"]>>Neuronal_fparam1;
    fs["Neuronal_fparam2"]>>Neuronal_fparam2;

    fs["SVM_train"]>>SVM_train;
    fs["SVM_Type"]>>SVM_Type;
    fs["SVM_kernel_type"]>>SVM_kernel_type;
    SVM_class_weights=0;
//    fs["SVM_class_weights"]>>SVM_class_weights;
    fs["SVM_degree"]>>SVM_degree;
    fs["SVM_coef0"]>>SVM_coef0;
    fs["SVM_nu"]>>SVM_nu;

    RTrees_priors=Mat();
    fs["RTrees_max_depth"]>>RTrees_max_depth;
    fs["RTrees_min_sample_count"]>>RTrees_min_sample_count;
    fs["RTrees_regression_accuracy"]>>RTrees_regression_accuracy;
    fs["RTrees_use_surrogates"]>>RTrees_use_surrogates;
    fs["RTrees_max_categories"]>>RTrees_max_categories;
    fs["RTrees_cv_folds"]>>RTrees_cv_folds;
    fs["RTrees_use_1se_rule"]>>RTrees_use_1se_rule;
    fs["RTrees_truncate_pruned_tree"]>>RTrees_truncate_pruned_tree;
    fs["RTrees_calc_var_importance"]>>RTrees_calc_var_importance;
    fs["RTrees_native_vars"]>>RTrees_native_vars;

    DTrees_priors=Mat();
    fs["DTrees_max_depth"]>>DTrees_max_depth;
    fs["DTrees_min_sample_count"]>>DTrees_min_sample_count;
    fs["DTrees_regression_accuracy"]>>DTrees_regression_accuracy;
    fs["DTrees_use_surrogates"]>>DTrees_use_surrogates;
    fs["DTrees_max_categories"]>>DTrees_max_categories;
    fs["DTrees_cv_folds"]>>DTrees_cv_folds;
    fs["DTrees_use_1se_rule"]>>DTrees_use_1se_rule;
    fs["DTrees_truncate_pruned_tree"]>>DTrees_truncate_pruned_tree;

    Boosting_priors=Mat();
    fs["Boosting_boost_type"]>>Boosting_boost_type;
    fs["Boosting_weak_count"]>>Boosting_weak_count;
    fs["Boosting_weight_trim_rate"]>>Boosting_weight_trim_rate;
    fs["Boosting_max_depth"]>>Boosting_max_depth;
    fs["Boosting_use_surrogates"]>>Boosting_use_surrogates;

    fs["EM_nclusters"]>>EM_nclusters;
    fs["EM_covMatType"]>>EM_covMatType;

//    fs["GBT_loss_function_type"]>>GBT_loss_function_type;
//    fs["GBT_weak_count"]>>GBT_weak_count;
//    fs["GBT_shrinkage"]>>GBT_shrinkage;
//    fs["GBT_subsample_portion"]>>GBT_subsample_portion;
//    fs["GBT_max_depth"]>>GBT_max_depth;
//    fs["GBT_use_surrogates"]>>GBT_use_surrogates;

//    ERTrees_priors=Mat();
//    fs["ERTrees_max_depth"]>>ERTrees_max_depth;
//    fs["ERTrees_min_sample_count"]>>ERTrees_min_sample_count;
//    fs["ERTrees_regression_accuracy"]>>ERTrees_regression_accuracy;
//    fs["ERTrees_use_surrogates"]>>ERTrees_use_surrogates;
//    fs["ERTrees_max_categories"]>>ERTrees_max_categories;
//    fs["ERTrees_cv_folds"]>>ERTrees_cv_folds;
//    fs["ERTrees_use_1se_rule"]>>ERTrees_use_1se_rule;
//    fs["ERTrees_truncate_pruned_tree"]>>ERTrees_truncate_pruned_tree;
//    fs["ERTrees_calc_var_importance"]>>ERTrees_calc_var_importance;
//    fs["ERTrees_native_vars"]>>ERTrees_native_vars;

    id_clasificadores.clear();
    nombres.clear();

    X=0;
    Y=0;

    fs["num_bar"]>>num_bar;
    fs["show_graphics"]>>show_graphics;
    fs["save_clasif"]>>save_clasif;
    fs["save_data"]>>save_data;
    fs["save_other"]>>save_other;
    fs["read"]>>read;
    fs["ifreduc"]>>ifreduc;

    fs.release();

    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_Tipo_Herr_activated(int index)
{
    ui->Direccion_Datos->setText("Direccion Archivo, Video o Carpeta");
    if(index==0){
        ui->Config_Herr->setEnabled(false);
        ui->Tipo_Datos->setEnabled(false);
        ui->Bot_Img->setChecked(true);
        ui->Bot_Img->setEnabled(true);
        ui->Bot_Descriptor->setChecked(false);
        ui->Bot_Descriptor->setEnabled(false);
        ui->Config_Des->setEnabled(false);
        ui->Tipo_Descrip->setEnabled(false);
        ui->Rees_X->setEnabled(false);
        ui->Rees_Y->setEnabled(false);
        ui->checkBox_Cuadrado->setEnabled(false);
        ui->Num_Clas->setEnabled(false);
        ui->Num_Dat->setEnabled(false);
        ui->Tam_X->setEnabled(false);
        ui->Tam_Y->setEnabled(false);
        ui->Varianza->setEnabled(false);
        ui->Separacion->setEnabled(false);
        ui->Num_frame->setEnabled(false);
        ui->Max_Noise->setEnabled(false);
        ui->Max_Blur->setEnabled(false);
        ui->Max_X->setEnabled(false);
        ui->Max_Y->setEnabled(false);
        ui->Max_Z->setEnabled(false);
        ui->Direccion_Datos->setEnabled(false);
        ui->toolButton->setEnabled(false);
    }
    else if(index==1){
        ui->Config_Herr->setEnabled(true);
        ui->Tipo_Datos->setEnabled(true);
        ui->Bot_Img->setChecked(true);
        ui->Bot_Img->setEnabled(true);
        ui->Bot_Descriptor->setChecked(false);
        ui->Bot_Descriptor->setEnabled(false);
        ui->Config_Des->setEnabled(false);
        ui->Tipo_Descrip->setEnabled(false);
        ui->Rees_X->setEnabled(true);
        ui->Rees_Y->setEnabled(true);
        ui->checkBox_Cuadrado->setEnabled(false);
        ui->Num_Clas->setEnabled(false);
        ui->Num_Dat->setEnabled(false);
        ui->Tam_X->setEnabled(false);
        ui->Tam_Y->setEnabled(false);
        ui->Varianza->setEnabled(false);
        ui->Separacion->setEnabled(false);
        ui->Num_frame->setEnabled(false);
        ui->Max_Noise->setEnabled(false);
        ui->Max_Blur->setEnabled(false);
        ui->Max_X->setEnabled(false);
        ui->Max_Y->setEnabled(false);
        ui->Max_Z->setEnabled(false);
        ui->Direccion_Datos->setEnabled(true);
        ui->toolButton->setEnabled(true);
    }
    else if(index==2){
        ui->Config_Herr->setEnabled(true);
        ui->Tipo_Datos->setEnabled(true);
        ui->Bot_Img->setChecked(true);
        ui->Bot_Img->setEnabled(true);
        ui->Bot_Descriptor->setChecked(false);
        ui->Bot_Descriptor->setEnabled(false);
        ui->Config_Des->setEnabled(false);
        ui->Tipo_Descrip->setEnabled(false);
        ui->Rees_X->setEnabled(true);
        ui->Rees_Y->setEnabled(true);
        ui->checkBox_Cuadrado->setEnabled(false);
        ui->Num_Clas->setEnabled(false);
        ui->Num_Dat->setEnabled(false);
        ui->Tam_X->setEnabled(false);
        ui->Tam_Y->setEnabled(false);
        ui->Varianza->setEnabled(false);
        ui->Separacion->setEnabled(false);
        ui->Num_frame->setEnabled(false);
        ui->Max_Noise->setEnabled(false);
        ui->Max_Blur->setEnabled(false);
        ui->Max_X->setEnabled(false);
        ui->Max_Y->setEnabled(false);
        ui->Max_Z->setEnabled(false);
        ui->Direccion_Datos->setEnabled(true);
        ui->toolButton->setEnabled(true);
    }
    else if(index==3){
        ui->Config_Herr->setEnabled(true);
        ui->Tipo_Datos->setEnabled(true);
        ui->Bot_Img->setChecked(true);
        ui->Bot_Img->setEnabled(true);
        ui->Bot_Descriptor->setChecked(false);
        ui->Bot_Descriptor->setEnabled(false);
        ui->Config_Des->setEnabled(false);
        ui->Tipo_Descrip->setEnabled(false);
        ui->Rees_X->setEnabled(true);
        ui->Rees_Y->setEnabled(true);
        ui->checkBox_Cuadrado->setEnabled(true);
        ui->Num_Clas->setEnabled(false);
        ui->Num_Dat->setEnabled(false);
        ui->Tam_X->setEnabled(false);
        ui->Tam_Y->setEnabled(false);
        ui->Varianza->setEnabled(false);
        ui->Separacion->setEnabled(false);
        ui->Num_frame->setEnabled(false);
        ui->Max_Noise->setEnabled(false);
        ui->Max_Blur->setEnabled(false);
        ui->Max_X->setEnabled(false);
        ui->Max_Y->setEnabled(false);
        ui->Max_Z->setEnabled(false);
        ui->Direccion_Datos->setEnabled(true);
        ui->toolButton->setEnabled(true);
    }
    else if(index==4){
        ui->Config_Herr->setEnabled(true);
        ui->Tipo_Datos->setEnabled(true);
        ui->Bot_Img->setChecked(true);
        ui->Bot_Img->setEnabled(true);
        ui->Bot_Descriptor->setChecked(false);
        ui->Bot_Descriptor->setEnabled(false);
        ui->Config_Des->setEnabled(false);
        ui->Tipo_Descrip->setEnabled(false);
        ui->Rees_X->setEnabled(true);
        ui->Rees_Y->setEnabled(true);
        ui->checkBox_Cuadrado->setEnabled(true);
        ui->Num_Clas->setEnabled(false);
        ui->Num_Dat->setEnabled(false);
        ui->Tam_X->setEnabled(false);
        ui->Tam_Y->setEnabled(false);
        ui->Varianza->setEnabled(false);
        ui->Separacion->setEnabled(false);
        ui->Num_frame->setEnabled(false);
        ui->Max_Noise->setEnabled(false);
        ui->Max_Blur->setEnabled(false);
        ui->Max_X->setEnabled(false);
        ui->Max_Y->setEnabled(false);
        ui->Max_Z->setEnabled(false);
        ui->Direccion_Datos->setEnabled(true);
        ui->toolButton->setEnabled(true);
    }
    else if(index==5){
        ui->Config_Herr->setEnabled(true);
        ui->Tipo_Datos->setEnabled(true);
        ui->Bot_Img->setChecked(true);
        ui->Bot_Img->setEnabled(true);
        ui->Bot_Descriptor->setChecked(false);
        ui->Bot_Descriptor->setEnabled(false);
        ui->Config_Des->setEnabled(false);
        ui->Tipo_Descrip->setEnabled(false);
        ui->Rees_X->setEnabled(true);
        ui->Rees_Y->setEnabled(true);
        ui->checkBox_Cuadrado->setEnabled(true);
        ui->Num_Clas->setEnabled(false);
        ui->Num_Dat->setEnabled(false);
        ui->Tam_X->setEnabled(false);
        ui->Tam_Y->setEnabled(false);
        ui->Varianza->setEnabled(false);
        ui->Separacion->setEnabled(false);
        ui->Num_frame->setEnabled(false);
        ui->Max_Noise->setEnabled(false);
        ui->Max_Blur->setEnabled(false);
        ui->Max_X->setEnabled(false);
        ui->Max_Y->setEnabled(false);
        ui->Max_Z->setEnabled(false);
        ui->Direccion_Datos->setEnabled(true);
        ui->toolButton->setEnabled(true);
    }
    else if(index==6){
        ui->Config_Herr->setEnabled(true);
        ui->Tipo_Datos->setEnabled(true);
        ui->Bot_Img->setChecked(true);
        ui->Bot_Img->setEnabled(true);
        ui->Bot_Descriptor->setChecked(false);
        ui->Bot_Descriptor->setEnabled(false);
        ui->Config_Des->setEnabled(false);
        ui->Tipo_Descrip->setEnabled(false);
        ui->Rees_X->setEnabled(true);
        ui->Rees_Y->setEnabled(true);
        ui->checkBox_Cuadrado->setEnabled(false);
        ui->Num_Clas->setEnabled(false);
        ui->Num_Dat->setEnabled(false);
        ui->Tam_X->setEnabled(false);
        ui->Tam_Y->setEnabled(false);
        ui->Varianza->setEnabled(false);
        ui->Separacion->setEnabled(false);
        ui->Num_frame->setEnabled(true);
        ui->Max_Noise->setEnabled(false);
        ui->Max_Blur->setEnabled(false);
        ui->Max_X->setEnabled(false);
        ui->Max_Y->setEnabled(false);
        ui->Max_Z->setEnabled(false);
        ui->Direccion_Datos->setEnabled(true);
        ui->toolButton->setEnabled(true);
    }
    else if(index==7){
        ui->Config_Herr->setEnabled(true);
        ui->Tipo_Datos->setEnabled(true);
        ui->Bot_Img->setChecked(true);
        ui->Bot_Img->setEnabled(true);
        ui->Bot_Descriptor->setChecked(false);
        ui->Bot_Descriptor->setEnabled(false);
        ui->Config_Des->setEnabled(false);
        ui->Tipo_Descrip->setEnabled(false);
        ui->Rees_X->setEnabled(true);
        ui->Rees_Y->setEnabled(true);
        ui->checkBox_Cuadrado->setEnabled(true);
        ui->Num_Clas->setEnabled(false);
        ui->Num_Dat->setEnabled(false);
        ui->Tam_X->setEnabled(false);
        ui->Tam_Y->setEnabled(false);
        ui->Varianza->setEnabled(false);
        ui->Separacion->setEnabled(false);
        ui->Num_frame->setEnabled(true);
        ui->Max_Noise->setEnabled(false);
        ui->Max_Blur->setEnabled(false);
        ui->Max_X->setEnabled(false);
        ui->Max_Y->setEnabled(false);
        ui->Max_Z->setEnabled(false);
        ui->Direccion_Datos->setEnabled(true);
        ui->toolButton->setEnabled(true);
    }
    else if(index==8){
        ui->Config_Herr->setEnabled(true);
        ui->Tipo_Datos->setEnabled(true);
        ui->Bot_Img->setChecked(true);
        ui->Bot_Img->setEnabled(true);
        ui->Bot_Descriptor->setChecked(false);
        ui->Bot_Descriptor->setEnabled(false);
        ui->Config_Des->setEnabled(false);
        ui->Tipo_Descrip->setEnabled(false);
        ui->Rees_X->setEnabled(false);
        ui->Rees_Y->setEnabled(false);
        ui->checkBox_Cuadrado->setEnabled(false);
        ui->Num_Clas->setEnabled(true);
        ui->Num_Dat->setEnabled(true);
        ui->Tam_X->setEnabled(true);
        ui->Tam_Y->setEnabled(true);
        ui->Varianza->setEnabled(true);
        ui->Separacion->setEnabled(true);
        ui->Num_frame->setEnabled(false);
        ui->Max_Noise->setEnabled(false);
        ui->Max_Blur->setEnabled(false);
        ui->Max_X->setEnabled(false);
        ui->Max_Y->setEnabled(false);
        ui->Max_Z->setEnabled(false);
        ui->Direccion_Datos->setEnabled(false);
        ui->toolButton->setEnabled(false);
    }
    else if(index==9){
        ui->Config_Herr->setEnabled(true);
        ui->Tipo_Datos->setEnabled(true);
        ui->Bot_Img->setChecked(true);
        ui->Bot_Img->setEnabled(true);
        ui->Bot_Descriptor->setChecked(false);
        ui->Bot_Descriptor->setEnabled(false);
        ui->Config_Des->setEnabled(false);
        ui->Tipo_Descrip->setEnabled(false);
        ui->Rees_X->setEnabled(false);
        ui->Rees_Y->setEnabled(false);
        ui->checkBox_Cuadrado->setEnabled(false);
        ui->Num_Clas->setEnabled(false);
        ui->Num_Dat->setEnabled(false);
        ui->Tam_X->setEnabled(false);
        ui->Tam_Y->setEnabled(false);
        ui->Varianza->setEnabled(false);
        ui->Separacion->setEnabled(false);
        ui->Num_frame->setEnabled(true);
        ui->Max_Noise->setEnabled(true);
        ui->Max_Blur->setEnabled(true);
        ui->Max_X->setEnabled(true);
        ui->Max_Y->setEnabled(true);
        ui->Max_Z->setEnabled(true);
        ui->Direccion_Datos->setEnabled(true);
        ui->toolButton->setEnabled(true);
    }
    else if(index==10){
        ui->Config_Herr->setEnabled(false);
        ui->Tipo_Datos->setEnabled(true);
        ui->Bot_Img->setChecked(false);
        ui->Bot_Img->setEnabled(false);
        ui->Bot_Descriptor->setChecked(true);
        ui->Bot_Descriptor->setEnabled(true);
        ui->Config_Des->setEnabled(true);
        ui->Tipo_Descrip->setEnabled(true);
        ui->Rees_X->setEnabled(false);
        ui->Rees_Y->setEnabled(false);
        ui->checkBox_Cuadrado->setEnabled(false);
        ui->Num_Clas->setEnabled(false);
        ui->Num_Dat->setEnabled(false);
        ui->Tam_X->setEnabled(false);
        ui->Tam_Y->setEnabled(false);
        ui->Varianza->setEnabled(false);
        ui->Separacion->setEnabled(false);
        ui->Num_frame->setEnabled(false);
        ui->Max_Noise->setEnabled(false);
        ui->Max_Blur->setEnabled(false);
        ui->Max_X->setEnabled(false);
        ui->Max_Y->setEnabled(false);
        ui->Max_Z->setEnabled(false);
        ui->Direccion_Datos->setEnabled(true);
        ui->toolButton->setEnabled(true);
    }
    else if(index==11){
        ui->Config_Herr->setEnabled(false);
        ui->Tipo_Datos->setEnabled(false);
        ui->Bot_Img->setChecked(false);
        ui->Bot_Img->setEnabled(false);
        ui->Bot_Descriptor->setChecked(false);
        ui->Bot_Descriptor->setEnabled(false);
        ui->Config_Des->setEnabled(false);
        ui->Tipo_Descrip->setEnabled(false);
        ui->Rees_X->setEnabled(false);
        ui->Rees_Y->setEnabled(false);
        ui->checkBox_Cuadrado->setEnabled(false);
        ui->Num_Clas->setEnabled(false);
        ui->Num_Dat->setEnabled(false);
        ui->Tam_X->setEnabled(false);
        ui->Tam_Y->setEnabled(false);
        ui->Varianza->setEnabled(false);
        ui->Separacion->setEnabled(false);
        ui->Num_frame->setEnabled(false);
        ui->Max_Noise->setEnabled(false);
        ui->Max_Blur->setEnabled(false);
        ui->Max_X->setEnabled(false);
        ui->Max_Y->setEnabled(false);
        ui->Max_Z->setEnabled(false);
        ui->Direccion_Datos->setEnabled(true);
        ui->toolButton->setEnabled(true);
    }
}

void MainWindow::on_Config_Des_clicked()
{
    if(ui->Tipo_Descrip->currentIndex()==0){
        QMessageBox msgBox;
        msgBox.setText("ERROR: Seleccione un tipo de descriptor");
        msgBox.exec();
        return;
    }
    if(ui->Tipo_Descrip->currentIndex()==10){
        Conf_HOG *conf_HOG= new Conf_HOG(this,this);
        conf_HOG->show();
    }
    else if(ui->Tipo_Descrip->currentIndex()==11){
        Config_PC *conf_PC= new Config_PC(this,this);
        conf_PC->show();
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: El descriptor no tiene configuracion");
        msgBox.exec();
        return;
    }
}

void MainWindow::on_toolButton_clicked()
{
    if(ui->Tipo_Herr->currentIndex()!=0){
        QString filename;
        if(ui->Tipo_Herr->currentIndex()==1){
                    filename= QFileDialog::getExistingDirectory(
                                this,
                                tr("SELECCIONAR CARPETA"),
                                QDir::currentPath()+"/../Data/Imagenes");
        }
        else if(ui->Tipo_Herr->currentIndex()==2){
            filename= QFileDialog::getExistingDirectory(
                        this,
                        tr("SELECCIONAR CARPETA"),
                        QDir::currentPath()+"/../Data/Imagenes");
        }
        else if(ui->Tipo_Herr->currentIndex()==3){
            filename= QFileDialog::getExistingDirectory(
                        this,
                        tr("SELECCIONAR CARPETA"),
                        QDir::currentPath()+"/../Data/Imagenes");
        }
        else if(ui->Tipo_Herr->currentIndex()==4){
            filename = QFileDialog::getOpenFileName(
                         this,
                         tr("- SELECCIONAR EL VIDEO -"),
                         QDir::currentPath()+"/../Data",
                         tr("Document files (*.avi *.mov *.VOB *.ogv *.mp4 *.ogg *.mpg);;All files (*.*)") );
        }
        else if(ui->Tipo_Herr->currentIndex()==5){
            filename = QFileDialog::getOpenFileName(
                         this,
                         tr("- SELECCIONAR EL VIDEO -"),
                         QDir::currentPath()+"/../Data",
                         tr("Document files (*.avi *.mov *.VOB *.ogv *.mp4 *.ogg *.mpg);;All files (*.*)") );
        }
        else if(ui->Tipo_Herr->currentIndex()==6){
            filename = QFileDialog::getOpenFileName(
                         this,
                         tr("- SELECCIONAR EL ARCHIVO -"),
                        QDir::currentPath()+"/../Data/Imagenes/",
                         tr("Document files (*.txt);;All files (*.*)") );
        }
        else if(ui->Tipo_Herr->currentIndex()==7){
            filename = QFileDialog::getOpenFileName(
                         this,
                         tr("- SELECCIONAR EL VIDEO -"),
                         QDir::currentPath()+"/../Data",
                         tr("Document files (*.avi *.mov *.VOB *.ogv *.mp4 *.ogg *.mpg);;All files (*.*)") );
        }
        else if(ui->Tipo_Herr->currentIndex()==9){
            filename = QFileDialog::getOpenFileName(
                         this,
                         tr("- SELECCIONAR EL ARCHIVO -"),
                         QDir::currentPath()+"/../Data/Imagenes/",
                         tr("Document files (*.txt);;All files (*.*)") );
        }
        else if(ui->Tipo_Herr->currentIndex()==10){
            filename= QFileDialog::getExistingDirectory(
                        this,
                        tr("SELECCIONAR CARPETA"),
                        QDir::currentPath()+"/../Data/Imagenes/");
        }
        else if(ui->Tipo_Herr->currentIndex()==11){
            filename= QFileDialog::getExistingDirectory(
                        this,
                        tr("SELECCIONAR CARPETA"),
                        QDir::currentPath()+"/../Data");
        }
        ui->Direccion_Datos->setText(filename);
    }
}

void MainWindow::on_Iniciar_clicked()
{
    int e=0;
    int pos_barra=0;
    LABELS.clear();
    IMAGENES.clear();
    Generacion gen;
    Auxiliares aux;
    QString referencia=ui->Nombre_Datos->displayText();
    QString direccion=ui->Direccion_Datos->displayText();
    std::string ref=referencia.toStdString();
    std::string Dir=direccion.toStdString();
    for(uint i=0; i<ref.size(); i++){
        if(ref[i]==' '){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se permiten espacios en la referencia");
            msgBox.exec();
            return;
        }
    }
    if(ui->Tipo_Herr->currentIndex()!=8){
        if(Dir=="Direccion Archivo, Video o Carpeta"){
            QMessageBox msgBox;
            msgBox.setText("ERROR: Seleccione una direccion valida");
            msgBox.exec();
            return;
        }
    }
    int Rees_X=ui->Rees_X->value();
    int Rees_Y=ui->Rees_Y->value();
    int Num_Clas=ui->Num_Clas->value();
    int Num_Dat=ui->Num_Dat->value();
    int Tam_X=ui->Tam_X->value();
    int Tam_Y=ui->Tam_Y->value();
    float Varianza=ui->Varianza->value();
    float Separacion=ui->Separacion->value();
    int Num_frame=ui->Num_frame->value();
    float Max_Noise=ui->Max_Noise->value();
    float Max_Blur=ui->Max_Blur->value();
    float Max_X=ui->Max_X->value();
    float Max_Y=ui->Max_Y->value();
    float Max_Z=ui->Max_Z->value();
    if(ui->Tipo_Herr->currentIndex()!=0){
        if(ui->Tipo_Herr->currentIndex()==1){
            gen.window=ui;
            e=gen.Datos_Imagenes(ref,Dir,cv::Size2i(Rees_X,Rees_Y),LABELS,IMAGENES,info,save_data);
        }
        else if(ui->Tipo_Herr->currentIndex()==2){
            gen.window=ui;
            e=gen.Etiquetar(ref,Dir,cv::Size2i(Rees_X,Rees_Y),LABELS,IMAGENES,info,save_data);
        }
        else if(ui->Tipo_Herr->currentIndex()==3){
            gen.window=ui;
            e=gen.Recortar_Etiquetar(ref,Dir,ui->checkBox_Cuadrado->isChecked(),Size2i(Rees_X,Rees_Y),LABELS,IMAGENES,info,save_data);
        }
        else if(ui->Tipo_Herr->currentIndex()==4){
            gen.window=ui;
            cv::VideoCapture cap(Dir);
            e=gen.Recortar_Etiquetar(ref,cap,ui->checkBox_Cuadrado->isChecked(),Size2i(Rees_X,Rees_Y),LABELS,IMAGENES,info,save_data);
        }
        else if(ui->Tipo_Herr->currentIndex()==5){
            gen.window=ui;
            cv::VideoCapture cap(Dir);
            e=gen.Autopositivos(ref,cap,ui->checkBox_Cuadrado->isChecked(),Size2i(Rees_X,Rees_Y),LABELS,IMAGENES,info,save_data);
        }
        else if(ui->Tipo_Herr->currentIndex()==6){
            gen.window=ui;
            e=gen.Autonegativos(ref,Dir,Size2i(Rees_X,Rees_Y),Num_frame,IMAGENES,LABELS,info,save_data);
        }
        else if(ui->Tipo_Herr->currentIndex()==7){
            gen.window=ui;
            cv::VideoCapture cap(Dir);
            e=gen.Autogeneracion(ref,cap,Num_frame,ui->checkBox_Cuadrado->isChecked(),Size2i(Rees_X,Rees_Y),LABELS,IMAGENES,info,save_data);
        }
        else if(ui->Tipo_Herr->currentIndex()==8){
            gen.total_progreso=Num_Clas*Num_Dat;
            gen.progreso=0;
            gen.base_progreso=0;
            gen.max_progreso=40;
            gen.window=ui;
            e=gen.Random_Synthetic_Data(ref,Num_Clas,Num_Dat,Size(Tam_X,Tam_Y),Varianza,Separacion,IMAGENES,LABELS,info,save_data);
            pos_barra=40;
        }
        else if(ui->Tipo_Herr->currentIndex()==9){
            vector<Mat> Imagenes;
            vector<float> Lab;
            int pos=0;
            for(uint i=0; i<Dir.size(); i++){
                if(Dir[i]=='/')
                    pos=i;
            }
            std::string archivo_i;
            for(int i=0; i<pos+1; i++)
                archivo_i=archivo_i+Dir[i];
            archivo_i=archivo_i+"Info.xml";
            cv::FileStorage Archivo_i(archivo_i,CV_STORAGE_READ);
            if(!Archivo_i.isOpened()){
                QMessageBox msgBox;
                msgBox.setText("ERROR: La carpeta no tiene la estructura utilizada por el sistema");
                msgBox.exec();
                return;
            }
            int num;
            Archivo_i["Num_Datos"]>>num;
            Archivo_i.release();
            gen.total_progreso=num;
            gen.progreso=0;
            gen.base_progreso=0;
            gen.max_progreso=30;
            gen.window=ui;
            e=gen.Cargar_Fichero(Dir,Imagenes,Lab,info);
            ui->progress_Cargar->setValue(1);
            if(e==1){
                QMessageBox msgBox;
                msgBox.setText("ERROR: No se ha podido cargar el fichero");
                msgBox.exec();
                ui->progress_Clasificar->setValue(100);
                ui->progress_Clasificar->setValue(0);
                ui->progress_generar->setValue(100);
                ui->progress_generar->setValue(0);
                ui->progress_Cargar->setValue(100);
                ui->progress_Cargar->setValue(0);
                ui->progress_Clus->setValue(100);
                ui->progress_Clus->setValue(0);
                ui->progress_Dimensionalidad->setValue(100);
                ui->progress_Dimensionalidad->setValue(0);
                return;
            }
            gen.total_progreso=Imagenes.size();
            gen.progreso=0;
            gen.base_progreso=30;
            gen.max_progreso=20;
            gen.window=ui;
            pos_barra=50;
            e=gen.Synthethic_Data(ref,Imagenes,Lab,IMAGENES,LABELS,Num_frame,Max_Noise,Max_Blur,Max_X,Max_Y,Max_Z,info,save_data);
        }
        else if(ui->Tipo_Herr->currentIndex()==10){
            string archivo_i=Dir+"/Info.xml";
            cv::FileStorage Archivo_i(archivo_i,CV_STORAGE_READ);
            if(!Archivo_i.isOpened()){
                QMessageBox msgBox;
                msgBox.setText("ERROR: La carpeta no tiene la estructura utilizada por el sistema");
                msgBox.exec();
                return;
            }
            else{
                int num;
                Archivo_i["Num_Datos"]>>num;
                Archivo_i.release();
                gen.total_progreso=num;
                gen.progreso=0;
                gen.base_progreso=30;
                gen.max_progreso=20;
                gen.window=ui;
                pos_barra=50;
                string input_directory=Dir+"/Recortes.txt";
                e=gen.Cargar_Fichero(input_directory,IMAGENES,LABELS, info);
                if(e==1){
                    QMessageBox msgBox;
                    msgBox.setText("ERROR: No se han podido cargar los datos");
                    msgBox.exec();
                    ui->progress_Clasificar->setValue(100);
                    ui->progress_Clasificar->setValue(0);
                    ui->progress_generar->setValue(100);
                    ui->progress_generar->setValue(0);
                    ui->progress_Cargar->setValue(100);
                    ui->progress_Cargar->setValue(0);
                    ui->progress_Clus->setValue(100);
                    ui->progress_Clus->setValue(0);
                    ui->progress_Dimensionalidad->setValue(100);
                    ui->progress_Dimensionalidad->setValue(0);
                }
                ui->label_Datos->setText("Datos ref: "+referencia);
                vector<Mat> descriptores;
                if(ui->Tipo_Descrip->currentIndex()==1){
                    Basic_Transformations basic(info.Tipo_Datos,RGB);
                    e=basic.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_RGB";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else if(ui->Tipo_Descrip->currentIndex()==2){
                    Basic_Transformations basic(info.Tipo_Datos,GRAY);
                    e=basic.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_GRAY";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else if(ui->Tipo_Descrip->currentIndex()==3){
                    Basic_Transformations basic(info.Tipo_Datos,HSV);
                    e=basic.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_HSV";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else if(ui->Tipo_Descrip->currentIndex()==4){
                    Basic_Transformations basic(info.Tipo_Datos,H_CHANNEL);
                    e=basic.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_H";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else if(ui->Tipo_Descrip->currentIndex()==5){
                    Basic_Transformations basic(info.Tipo_Datos,S_CHANNEL);
                    e=basic.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_S";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else if(ui->Tipo_Descrip->currentIndex()==6){
                    Basic_Transformations basic(info.Tipo_Datos,V_CHANNEL);
                    e=basic.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_V";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else if(ui->Tipo_Descrip->currentIndex()==7){
                    Basic_Transformations basic(info.Tipo_Datos,THRESHOLD);
                    e=basic.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_THRESHOLD";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else if(ui->Tipo_Descrip->currentIndex()==8){
                    Basic_Transformations basic(info.Tipo_Datos,CANNY);
                    e=basic.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_CANNY";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else if(ui->Tipo_Descrip->currentIndex()==9){
                    Basic_Transformations basic(info.Tipo_Datos,SOBEL);
                    e=basic.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_SOBEL";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else if(ui->Tipo_Descrip->currentIndex()==10){
                    if(Win_Size.height>IMAGENES[0].rows || Win_Size.width>IMAGENES[0].cols){
                        QMessageBox msgBox;
                        msgBox.setText("ERROR: El tamaño de las imagenes es menor que el de la ventana de HOG");
                        msgBox.exec();
                        ui->progress_Clasificar->setValue(100);
                        ui->progress_Clasificar->setValue(0);
                        ui->progress_generar->setValue(100);
                        ui->progress_generar->setValue(0);
                        ui->progress_Cargar->setValue(100);
                        ui->progress_Cargar->setValue(0);
                        ui->progress_Clus->setValue(100);
                        ui->progress_Clus->setValue(0);
                        ui->progress_Dimensionalidad->setValue(100);
                        ui->progress_Dimensionalidad->setValue(0);
                        return;
                    }
                    HOG H(Win_Size,Block_Stride, Win_Sigma,Threshold_L2hys, Gamma_Correction, Nlevels);
                    e=H.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_HOG";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else if(ui->Tipo_Descrip->currentIndex()==11){
                    Puntos_Caracteristicos des(Tipo_Des,Tipo_Ext,Parametro);
                    e=des.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_PC";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else if(ui->Tipo_Descrip->currentIndex()==12){
                    Basic_Transformations basic(info.Tipo_Datos,COLOR_PREDOMINANTE);
                    e=basic.Extract(IMAGENES,descriptores);
                    if(descriptores.size()!=IMAGENES.size()){
                        QMessageBox msgBox;
                        msgBox.setText("WARNING: No se ha podido extraer descriptores de todas las imagenes");
                        msgBox.exec();
                    }
                    ref=ref+"_COLOR_PREDOMINANTE";
                    ui->progress_generar->setValue(50);
                    pos_barra=50;
                }
                else{
                    QMessageBox msgBox;
                    msgBox.setText("ERROR: Seleccione un tipo de descriptor");
                    msgBox.exec();
                    return;
                }
                if(e==1){
                    QMessageBox msgBox;
                    msgBox.setText("ERROR: No se han podido generar los descriptores");
                    msgBox.exec();
                    ui->progress_Clasificar->setValue(100);
                    ui->progress_Clasificar->setValue(0);
                    ui->progress_generar->setValue(100);
                    ui->progress_generar->setValue(0);
                    ui->progress_Cargar->setValue(100);
                    ui->progress_Cargar->setValue(0);
                    ui->progress_Clus->setValue(100);
                    ui->progress_Clus->setValue(0);
                    ui->progress_Dimensionalidad->setValue(100);
                    ui->progress_Dimensionalidad->setValue(0);
                    return;
                }
                int pos_bar_tor;
                for(uint zzz=0; zzz<ref.size(); zzz++){
                    if(ref[zzz]=='_')
                        pos_bar_tor=zzz;
                }
                string tip_dat;
                for(uint zzz=pos_bar_tor+1; zzz<ref.size(); zzz++){
                    tip_dat=tip_dat+ref[zzz];
                }
                if(tip_dat=="RGB")
                    info.Tipo_Datos=RGB;
                else if(tip_dat=="GRAY")
                    info.Tipo_Datos=GRAY;
                else if(tip_dat=="HSV")
                    info.Tipo_Datos=HSV;
                else if(tip_dat=="H")
                    info.Tipo_Datos=H_CHANNEL;
                else if(tip_dat=="S")
                    info.Tipo_Datos=S_CHANNEL;
                else if(tip_dat=="V")
                    info.Tipo_Datos=V_CHANNEL;
                else if(tip_dat=="THRESHOLD")
                    info.Tipo_Datos=THRESHOLD;
                else if(tip_dat=="CANNY")
                    info.Tipo_Datos=CANNY;
                else if(tip_dat=="SOBEL")
                    info.Tipo_Datos=SOBEL;
                else if(tip_dat=="HOG")
                    info.Tipo_Datos=HOG_DES;
                else if(tip_dat=="PC")
                    info.Tipo_Datos=PUNTOS_CARACTERISTICOS;
                else if(tip_dat=="COLOR_PREDOMINANTE")
                    info.Tipo_Datos=COLOR_PREDOMINANTE;
//                if(tip_dat=="FAC")
//                    info.Tipo_Datos=3;
                info.Tam_X=descriptores[0].cols;
                info.Tam_Y=descriptores[0].rows;
                gen.total_progreso=IMAGENES.size();
                gen.progreso=0;
                gen.base_progreso=pos_barra;
                gen.max_progreso=20;
                gen.window=ui;
                pos_barra=70;
                gen.Guardar_Datos(ref,descriptores,LABELS,info);
                referencia=QString::fromStdString(ref);
                Dat_Ref=ref;
                ui->label_Datos->setText("Datos ref: "+referencia);
            }
        }
        else if(ui->Tipo_Herr->currentIndex()==11){
            gen.total_progreso=aux.numero_imagenes(Dir);
            gen.progreso=0;
            gen.base_progreso=0;
            gen.max_progreso=50;
            gen.window=ui;
            pos_barra=50;
            e=gen.Juntar_Recortes(ref,Dir);
        }
        if(e==1){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se han podido generar los datos");
            msgBox.exec();
            ui->progress_Clasificar->setValue(100);
            ui->progress_Clasificar->setValue(0);
            ui->progress_generar->setValue(100);
            ui->progress_generar->setValue(0);
            ui->progress_Cargar->setValue(100);
            ui->progress_Cargar->setValue(0);
            ui->progress_Clus->setValue(100);
            ui->progress_Clus->setValue(0);
            ui->progress_Dimensionalidad->setValue(100);
            ui->progress_Dimensionalidad->setValue(0);
            return;
        }
        string input_directory;
        if(ui->Tipo_Herr->currentIndex()==11){
            string archivo_info=Dir+"/"+ref+"/Info.xml";
            cv::FileStorage Archivo_i(archivo_info,CV_STORAGE_READ);
            Archivo_i["Num_Datos"]>>gen.total_progreso;
            input_directory=Dir+"/"+ref+"/Recortes.txt";
        }
        else{
            gen.total_progreso=IMAGENES.size();
            input_directory="../Data/Imagenes/"+ref+"/Recortes.txt";
        }
        gen.progreso=0;
        gen.base_progreso=pos_barra;
        gen.max_progreso=100-pos_barra;
        gen.window=ui;
        IMAGENES.clear();
        LABELS.clear();
        e=gen.Cargar_Fichero(input_directory,IMAGENES,LABELS,info);
        if(e==1){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se han podido cargar los datos");
            msgBox.exec();
            ui->progress_Clasificar->setValue(100);
            ui->progress_Clasificar->setValue(0);
            ui->progress_generar->setValue(100);
            ui->progress_generar->setValue(0);
            ui->progress_Cargar->setValue(100);
            ui->progress_Cargar->setValue(0);
            ui->progress_Clus->setValue(100);
            ui->progress_Clus->setValue(0);
            ui->progress_Dimensionalidad->setValue(100);
            ui->progress_Dimensionalidad->setValue(0);
            return;
        }
        ui->Dim_X->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Dim_Y->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Dimension->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Num_Imagen->setMaximum(IMAGENES.size());
        bool neg;
        ui->Numero_Clases->setValue(aux.numero_etiquetas(LABELS,neg));
        ui->Dim_X_4->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Dim_Y_4->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Dimension_4->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Num_dimensiones->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Num_dimensiones->setValue(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Dim_X_5->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Dim_Y_5->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Dimension_3->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Dim_X_6->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Dim_Y_6->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Dimension_5->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Dimension_graf->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
        ui->Tam_Folds->setValue(IMAGENES.size()/ui->Num_folds->value());
        ui->progress_Clasificar->setValue(100);
        ui->progress_Clasificar->setValue(0);
        ui->progress_generar->setValue(100);
        ui->progress_generar->setValue(0);
        ui->progress_Cargar->setValue(100);
        ui->progress_Cargar->setValue(0);
        ui->progress_Clus->setValue(100);
        ui->progress_Clus->setValue(0);
        ui->progress_Dimensionalidad->setValue(100);
        ui->progress_Dimensionalidad->setValue(0);
        ui->label_Datos->setText("Datos ref: "+referencia);
    }
}

void MainWindow::on_Representar_2_clicked()
{
    if(IMAGENES.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay datos cargados");
        msgBox.exec();
        return;
    }
    if(LABELS.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay Etiquetas cargadas");
        msgBox.exec();
        return;
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int e=0;
    vector<int> dim;
    if(ui->Dimension->value()>IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: La dimension en Histograma esta fuera de rango");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    if(ui->Dim_X->value()<=IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()){
        dim.push_back(ui->Dim_X->value());
        if(ui->Dim_Y->value()>0 && ui->Dim_Y->value()<=IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels())
            dim.push_back(ui->Dim_Y->value());
        else if(ui->Dim_Y->value()==0){
        }
        else{
            QMessageBox msgBox;
            msgBox.setText("ERROR: La dimension Y esta fuera de rango");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: La dimension X esta fuera de rango");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    Representacion rep;
    if(ui->But_datos->isChecked()){
        e=rep.Data_represent("DATOS "+Dat_Ref,IMAGENES,LABELS,dim,Col);
    }
    else if(ui->But_Elipses->isChecked()){
        e=rep.Ellipse_represent("ELIPSES "+Dat_Ref,IMAGENES,LABELS,dim,Col);
    }
    else if(ui->But_Datos_Elipses->isChecked()){
        e=rep.Data_Ellipse_represent("DATOS CON ELIPSES "+Dat_Ref,IMAGENES,LABELS,dim,Col);
    }
    else if(ui->But_Hist->isChecked()){
        Analisis an;
        vector<vector<Mat> > Histo;
        vector<vector<int> > pos_barras;
        an.Histograma(IMAGENES,LABELS,num_bar,Histo,pos_barras);
        e=rep.Histogram_represent("HISTOGRAMA "+Dat_Ref,Histo,Col, ui->Dimension->value());
    }
    else if(ui->But_Imagen->isChecked() && (info.Tipo_Datos!= HOG_DES && info.Tipo_Datos!=PUNTOS_CARACTERISTICOS))
        e=rep.Imagen(IMAGENES,ui->Num_Imagen->value()-1);
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: Los datos no son de tipo Imagen");
        msgBox.exec();
    }
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido representar los datos");
        msgBox.exec();
    }
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_toolButton_2_clicked()
{
    QString filename= QFileDialog::getExistingDirectory(
                this,
                tr("SELECCIONAR CARPETA"),
                QDir::currentPath()+"/../Data/Imagenes/");
    ui->Direccion_Carga->setText(filename);
}

void MainWindow::on_Cargar_2_clicked()
{
	ui->progress_Clasificar->setValue(1);
    ui->progress_generar->setValue(1);
    ui->progress_Cargar->setValue(1);
    ui->progress_Clus->setValue(1);
    ui->progress_Dimensionalidad->setValue(1);
    int e=0;
    Generacion gen;
    Auxiliares aux;
    QString direccion=ui->Direccion_Carga->displayText();
    std::string Dir=direccion.toStdString();
    int pos=0;
    for(uint i=0; i<Dir.size(); i++){
        if(Dir[i]=='/')
            pos=i;
    }
    std::string ref;
    for(uint i=pos+1; i<Dir.size(); i++)
        ref=ref+Dir[i];
    QString referencia=QString::fromStdString(ref);
    string archivo_i=Dir+"/Info.xml";
    cv::FileStorage Archivo_i(archivo_i,CV_STORAGE_READ);
    if(!Archivo_i.isOpened()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: La carpeta no tiene la estructura utilizada por el sistema");
        msgBox.exec();
        return;
    }
    int num;
    Archivo_i["Num_Datos"]>>num;
    Archivo_i.release();
    gen.total_progreso=num;
    gen.progreso=0;
    gen.base_progreso=0;
    gen.max_progreso=100;
    gen.window=ui;
    string input_directory=Dir+"/Recortes.txt";
    e=gen.Cargar_Fichero(input_directory,IMAGENES,LABELS,info);
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido cargar los datos");
        msgBox.exec();
        ui->progress_Clasificar->setValue(100);
        ui->progress_Clasificar->setValue(0);
        ui->progress_generar->setValue(100);
        ui->progress_generar->setValue(0);
        ui->progress_Cargar->setValue(100);
        ui->progress_Cargar->setValue(0);
        ui->progress_Clus->setValue(100);
        ui->progress_Clus->setValue(0);
        ui->progress_Dimensionalidad->setValue(100);
        ui->progress_Dimensionalidad->setValue(0);
        return;
    }
    ui->Dim_X->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Dim_Y->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Dimension->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Num_Imagen->setMaximum(IMAGENES.size());
    bool neg;
    ui->Numero_Clases->setValue(aux.numero_etiquetas(LABELS,neg));
    ui->Dim_X_4->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Dim_Y_4->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Dimension_4->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Num_dimensiones->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Num_dimensiones->setValue(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Dim_X_5->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Dim_Y_5->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Dimension_3->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Dim_X_6->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Dim_Y_6->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Dimension_5->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Dimension_graf->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    ui->Tam_Folds->setValue(IMAGENES.size()/ui->Num_folds->value());
    ui->progress_Clasificar->setValue(100);
    ui->progress_Clasificar->setValue(0);
    ui->progress_generar->setValue(100);
    ui->progress_generar->setValue(0);
    ui->progress_Cargar->setValue(100);
    ui->progress_Cargar->setValue(0);
    ui->progress_Clus->setValue(100);
    ui->progress_Clus->setValue(0);
    ui->progress_Dimensionalidad->setValue(100);
    ui->progress_Dimensionalidad->setValue(0);
    ui->progress_Dimensionalidad->setValue(0);
    Dat_Ref=ref;
    ui->label_Datos->setText("Datos ref: "+referencia);
}

void MainWindow::on_Analizar_2_clicked()
{
    if(IMAGENES.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay datos cargados");
        msgBox.exec();
        return;
    }
    if(LABELS.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay Etiquetas cargadas");
        msgBox.exec();
        return;
    }
    int e=0;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    ui->progress_Analisis->setValue(1);
    Analisis an;
    Auxiliares aux;
    Mat Datos;
    bool negativa;
    int num=aux.numero_etiquetas(LABELS,negativa);
    vector<int> numero(num);
    for(int i=0; i<num; i++)
        numero[i]=0;
    for(uint i=0; i<LABELS.size(); i++){
        if(negativa && LABELS[i]==-1.0)
            numero[0]++;
        else if(negativa && LABELS[i]>0)
            numero[LABELS[i]]++;
        else if(negativa==false)
            numero[LABELS[i]-1]++;
    }
    aux.Image2Lexic(IMAGENES,Datos);
    vector<Mat> Medias_Mat,Des_tipics_Mat;
    vector<vector<Mat> > D_Prime;
    e=an.Estadisticos(Datos,LABELS,Medias_Mat,Des_tipics_Mat,D_Prime);
    ui->progress_Analisis->setValue(15);
    vector<Mat> Covarianzas;
    int maximo_progreso;
    if(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()<=1024){
        e=an.Covarianza(Datos,LABELS,Covarianzas);
        maximo_progreso=num*(Medias_Mat[0].cols+Des_tipics_Mat[0].cols+D_Prime.size()*D_Prime[0].size()+Covarianzas[0].rows*Covarianzas[0].cols);
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("El numero de dimensiones es mayor a 1024 por lo que no se mostrara la covarianza");
        msgBox.exec();
        maximo_progreso=num*(Medias_Mat[0].cols+Des_tipics_Mat[0].cols+D_Prime.size()*D_Prime[0].size());
    }
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido generar los datos");
        msgBox.exec();
        ui->progress_Analisis->setValue(0);
        return;
    }
    ui->progress_Analisis->setValue(30);
    QStandardItemModel *model=new QStandardItemModel(0,0);
    int progreso=0;
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
        QStandardItem *Num_Datos = new QStandardItem(QString("Numero Datos"));
        QStandardItem *Num = new QStandardItem(QString("%1").arg(numero[i]));
        Num_Datos->appendRow(Num);
        Lab->appendRow(Num_Datos);
        QStandardItem *Dimensiones = new QStandardItem(QString("Dimensiones"));
        QStandardItem *Dim = new QStandardItem(QString("%1").arg(Datos.cols));
        Dimensiones->appendRow(Dim);
        Lab->appendRow(Dimensiones);
        QStandardItem *Medias = new QStandardItem(QString("Media"));
        for(int j=0; j<Medias_Mat[i].cols; j++){
            stringstream media;
            media<<fixed<<Medias_Mat[i].at<float>(0,j);
            QString valor=QString::fromStdString(media.str());
            QStandardItem *Media = new QStandardItem(QString(valor));
            Medias->appendRow(Media);
            progreso++;
            ui->progress_Analisis->setValue(30+(70*progreso/maximo_progreso));
        }
        Lab->appendRow(Medias);
        QStandardItem *Des = new QStandardItem(QString("Desviacion Tipica"));
        for(int j=0; j<Des_tipics_Mat[i].cols; j++){
            stringstream desviacion;
            desviacion<<fixed<<Des_tipics_Mat[i].at<float>(0,j);
            QString valor=QString::fromStdString(desviacion.str());
            QStandardItem *desvi = new QStandardItem(QString(valor));
            Des->appendRow(desvi);
            progreso++;
            ui->progress_Analisis->setValue(30+(70*progreso/maximo_progreso));
        }
        Lab->appendRow(Des);
        QStandardItem *DPrime = new QStandardItem(QString("D-Prime"));
        for(int j=0; j<num; j++){
            int etiqueta2;
            if(negativa){
                if(j==0)
                    etiqueta2=-1;
                else
                    etiqueta2=j;
            }
            else
                etiqueta2=j+1;
            if(etiqueta!=etiqueta2){
                QStandardItem *Etiqueta_Etiqueta = new QStandardItem(QString("Etiqueta %1").arg(etiqueta2));
                for(int k=0; k<D_Prime[i][j].cols; k++){
                    stringstream dprime;
                    dprime<<fixed<<D_Prime[i][j].at<float>(0,k);
                    QString valor=QString::fromStdString(dprime.str());
                    QStandardItem *Dprime = new QStandardItem(QString(valor));
                    Etiqueta_Etiqueta->appendRow(Dprime);
                }
                DPrime->appendRow(Etiqueta_Etiqueta);
                progreso++;
                ui->progress_Analisis->setValue(30+(70*progreso/maximo_progreso));
            }
        }
        Lab->appendRow(DPrime);
        if(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()<=1024){
            QStandardItem *Cov = new QStandardItem(QString("Covarianza"));
            for(int j=0; j<Covarianzas[i].rows; j++){
                stringstream linea_covarianza;
                for(int k=0; k<Covarianzas[i].cols; k++){
                    linea_covarianza<<fixed<<Covarianzas[i].at<float>(j,k);
                    linea_covarianza<<" ";
                    progreso++;
                    ui->progress_Analisis->setValue(30+(70*progreso/maximo_progreso));
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
    ui->Estadisticos->setModel(model);
    ui->progress_Analisis->setValue(100);
    ui->progress_Analisis->setValue(0);
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_Tipo_Clus_activated(int index)
{
    if(index==0){
        ui->label_6->setEnabled(true);
        ui->Numero_Clases->setEnabled(true);
        ui->label_7->setEnabled(true);
        ui->Inicializaciones->setEnabled(true);
        ui->label_8->setEnabled(true);
        ui->Repeticiones->setEnabled(true);
        ui->label_9->setEnabled(false);
        ui->Max_Dist->setEnabled(false);
        ui->label_20->setEnabled(false);
        ui->Tam_Celda_Clus->setEnabled(false);
        ui->label_21->setEnabled(false);
        ui->Tipo_Cov->setEnabled(false);
    }
    else if(index==1){
        ui->label_6->setEnabled(false);
        ui->Numero_Clases->setEnabled(false);
        ui->label_7->setEnabled(false);
        ui->Inicializaciones->setEnabled(false);
        ui->label_8->setEnabled(false);
        ui->Repeticiones->setEnabled(false);
        ui->label_9->setEnabled(true);
        ui->Max_Dist->setEnabled(true);
        ui->label_20->setEnabled(false);
        ui->Tam_Celda_Clus->setEnabled(false);
        ui->label_21->setEnabled(false);
        ui->Tipo_Cov->setEnabled(false);
    }
    else if(index==2){
        ui->label_6->setEnabled(false);
        ui->Numero_Clases->setEnabled(false);
        ui->label_7->setEnabled(false);
        ui->Inicializaciones->setEnabled(false);
        ui->label_8->setEnabled(false);
        ui->Repeticiones->setEnabled(false);
        ui->label_9->setEnabled(true);
        ui->Max_Dist->setEnabled(true);
        ui->label_20->setEnabled(false);
        ui->Tam_Celda_Clus->setEnabled(false);
        ui->label_21->setEnabled(false);
        ui->Tipo_Cov->setEnabled(false);
    }
    else if(index==3){
        ui->label_6->setEnabled(false);
        ui->Numero_Clases->setEnabled(false);
        ui->label_7->setEnabled(false);
        ui->Inicializaciones->setEnabled(false);
        ui->label_8->setEnabled(false);
        ui->Repeticiones->setEnabled(false);
        ui->label_9->setEnabled(false);
        ui->Max_Dist->setEnabled(false);
        ui->label_20->setEnabled(true);
        ui->Tam_Celda_Clus->setEnabled(true);
        ui->label_21->setEnabled(false);
        ui->Tipo_Cov->setEnabled(false);
    }
    else if(index==4){
        ui->label_6->setEnabled(true);
        ui->Numero_Clases->setEnabled(true);
        ui->label_7->setEnabled(false);
        ui->Inicializaciones->setEnabled(false);
        ui->label_8->setEnabled(false);
        ui->Repeticiones->setEnabled(false);
        ui->label_9->setEnabled(false);
        ui->Max_Dist->setEnabled(false);
        ui->label_20->setEnabled(false);
        ui->Tam_Celda_Clus->setEnabled(false);
        ui->label_21->setEnabled(true);
        ui->Tipo_Cov->setEnabled(true);
    }
}

void MainWindow::on_Generar_2_clicked()
{
    if(IMAGENES.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay datos cargados");
        msgBox.exec();
        return;
    }
    QString referencia=ui->Nombre_Datos_2->displayText();
    std::string ref=referencia.toStdString();
    for(uint i=0; i<ref.size(); i++){
        if(ref[i]==' '){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se permiten espacios en la referencia");
            msgBox.exec();
            return;
        }
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    Clustering clus;
    Mat Centers;
    Labels.clear();
    if(ui->Tipo_Clus->currentIndex()==0){
        if(ui->Inicializaciones->currentIndex()==1)
            clus.K_mean(IMAGENES,ui->Numero_Clases->value(),Labels,Centers,ui->Repeticiones->value(),KMEANS_RANDOM_CENTERS);
        else if(ui->Inicializaciones->currentIndex()==2)
            clus.K_mean(IMAGENES,ui->Numero_Clases->value(),Labels,Centers,ui->Repeticiones->value(),KMEANS_PP_CENTERS);
        else{
            QMessageBox msgBox;
            msgBox.setText("ERROR: Elige una opcion de Attempts");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
    }
    else if(ui->Tipo_Clus->currentIndex()==1){
        clus.Distancias_Encadenadas(IMAGENES,ui->Max_Dist->value(),Labels,Centers);
    }
    else if(ui->Tipo_Clus->currentIndex()==2){
        clus.Min_Max(IMAGENES,ui->Max_Dist->value(),Labels,Centers);
    }
    else if(ui->Tipo_Clus->currentIndex()==3){
        clus.Histograma(IMAGENES,ui->Tam_Celda_Clus->value(),Labels,Centers);
    }
    else if(ui->Tipo_Clus->currentIndex()==4){
        if(ui->Tipo_Cov->currentIndex()==0)
            clus.EXP_MAX(IMAGENES,Labels,Centers,ui->Numero_Clases->value(),ml::EM::COV_MAT_SPHERICAL);
        else if(ui->Tipo_Cov->currentIndex()==1)
            clus.EXP_MAX(IMAGENES,Labels,Centers,ui->Numero_Clases->value(),ml::EM::COV_MAT_DIAGONAL);
        else if(ui->Tipo_Cov->currentIndex()==2)
            clus.EXP_MAX(IMAGENES,Labels,Centers,ui->Numero_Clases->value(),ml::EM::COV_MAT_GENERIC);
    }
    QApplication::restoreOverrideCursor();
    ui->Clus_Representacion->setEnabled(true);
    ui->Guardar->setEnabled(true);
}


void MainWindow::on_Guardar_clicked()
{
    LABELS=Labels;
    QString referencia=ui->Nombre_Datos_2->displayText();
    std::string ref=referencia.toStdString();
    for(uint i=0; i<ref.size(); i++){
        if(ref[i]==' '){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se permiten espacios en la referencia");
            msgBox.exec();
            return;
        }
    }
    Generacion gen;
    gen.total_progreso=IMAGENES.size();
    gen.progreso=0;
    gen.base_progreso=0;
    gen.max_progreso=100;
    gen.window=ui;
    gen.Guardar_Datos(ref,IMAGENES,LABELS,info);
    ui->progress_Clasificar->setValue(100);
    ui->progress_Clasificar->setValue(0);
    ui->progress_generar->setValue(100);
    ui->progress_generar->setValue(0);
    ui->progress_Cargar->setValue(100);
    ui->progress_Cargar->setValue(0);
    ui->progress_Clus->setValue(100);
    ui->progress_Clus->setValue(0);
    ui->progress_Dimensionalidad->setValue(100);
    ui->progress_Dimensionalidad->setValue(0);
    ui->progress_Dimensionalidad->setValue(0);
    Dat_Ref=ref;
    ui->label_Datos->setText("Datos ref: "+referencia);
}

void MainWindow::on_Clus_Representar_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int e=0;
    vector<int> dim;
    if(ui->Dimension_4->value()>IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: La dimension en Histograma esta fuera de rango");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    if(ui->Dim_X_4->value()<=IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()){
        dim.push_back(ui->Dim_X_4->value());
        if(ui->Dim_Y_4->value()>0 && ui->Dim_Y_4->value()<=IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels())
            dim.push_back(ui->Dim_Y_4->value());
        else if(ui->Dim_Y_4->value()==0){
        }
        else{
            QMessageBox msgBox;
            msgBox.setText("ERROR: La dimension Y esta fuera de rango");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: La dimension X esta fuera de rango");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    Representacion rep;
    if(ui->But_datos_5->isChecked()){
        e=rep.Data_represent("DATOS "+Dat_Ref+" "+ui->Nombre_Datos_2->text().toStdString(),IMAGENES,Labels,dim,Col);
    }
    else if(ui->But_Elipses_5->isChecked()){
        e=rep.Ellipse_represent("ELIPSES "+Dat_Ref+" "+ui->Nombre_Datos_2->text().toStdString(),IMAGENES,Labels,dim,Col);
    }
    else if(ui->But_Datos_Elipses_5->isChecked()){
        e=rep.Data_Ellipse_represent("DATOS CON ELIPSES "+Dat_Ref+" "+ui->Nombre_Datos_2->text().toStdString(),IMAGENES,Labels,dim,Col);
    }
    else if(ui->But_Hist_4->isChecked()){
        Analisis an;
        vector<vector<Mat> > Histo;
        vector<vector<int> > pos_barras;
        an.Histograma(IMAGENES,Labels,num_bar,Histo,pos_barras);
        e=rep.Histogram_represent("HISTOGRAMA "+Dat_Ref+" "+ui->Nombre_Datos_2->text().toStdString(),Histo,Col, ui->Dimension_4->value());
    }
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido representar los datos. Generalmente este error es debido a que el numero de etiquetas excede al número de colores disponibles para la representacion");
        msgBox.exec();
    }
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_Button_Calidad_clicked()
{
    int e=0;
    if(IMAGENES.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay datos cargados");
        msgBox.exec();
        return;
    }
    if(LABELS.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay Etiquetas cargadas");
        msgBox.exec();
        return;
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if(ui->Num_dimensiones->value()>IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: El numero de dimensiones es mayor que el de los datos");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    QString referencia=ui->Nombre_Datos_3->displayText();
    std::string ref=referencia.toStdString();
    Representacion rep;
    Dimensionalidad dim(ref);
    Mat SEPARABILIDAD,SEPARABILIDAD_ACUM;
    int MEJ_DIM;
    if(ui->Radio_Dist->isChecked()){
        if(ui->But_LDA->isChecked())
            e=dim.Calidad_dimensiones_distancia(IMAGENES,LABELS,LDA_DIM,ui->Num_dimensiones->value(),SEPARABILIDAD,SEPARABILIDAD_ACUM,MEJ_DIM);
        else if(ui->But_PCA->isChecked())
            e=dim.Calidad_dimensiones_distancia(IMAGENES,LABELS,PCA_DIM,ui->Num_dimensiones->value(),SEPARABILIDAD,SEPARABILIDAD_ACUM,MEJ_DIM);
        else if(ui->But_distmax->isChecked())
            e=dim.Calidad_dimensiones_distancia(IMAGENES,LABELS,MAXDIST_DIM,ui->Num_dimensiones->value(),SEPARABILIDAD,SEPARABILIDAD_ACUM,MEJ_DIM);
        else if(ui->But_D_Prime->isChecked())
            e=dim.Calidad_dimensiones_distancia(IMAGENES,LABELS,D_PRIME_DIM,ui->Num_dimensiones->value(),SEPARABILIDAD,SEPARABILIDAD_ACUM,MEJ_DIM);
    }
    else if(ui->Radio_Dprime->isChecked()){
        if(ui->But_LDA->isChecked())
            e=dim.Calidad_dimensiones_d_prime(IMAGENES,LABELS,LDA_DIM,ui->Num_dimensiones->value(),SEPARABILIDAD,SEPARABILIDAD_ACUM,MEJ_DIM);
        else if(ui->But_PCA->isChecked())
            e=dim.Calidad_dimensiones_d_prime(IMAGENES,LABELS,PCA_DIM,ui->Num_dimensiones->value(),SEPARABILIDAD,SEPARABILIDAD_ACUM,MEJ_DIM);
        else if(ui->But_distmax->isChecked())
            e=dim.Calidad_dimensiones_d_prime(IMAGENES,LABELS,MAXDIST_DIM,ui->Num_dimensiones->value(),SEPARABILIDAD,SEPARABILIDAD_ACUM,MEJ_DIM);
        else if(ui->But_D_Prime->isChecked())
            e=dim.Calidad_dimensiones_d_prime(IMAGENES,LABELS,D_PRIME_DIM,ui->Num_dimensiones->value(),SEPARABILIDAD,SEPARABILIDAD_ACUM,MEJ_DIM);
    }
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se ha podido calcular la calidad de las dimensiones");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    QMessageBox msgBox;
    stringstream txt;
    txt<<"INFO: El numero de dimensiones optimo es "<<MEJ_DIM;
    msgBox.setText(QString::fromStdString(txt.str()));
    msgBox.exec();
    vector<float> GRAP;
    for(int i=0; i<SEPARABILIDAD.rows; i++)
        GRAP.push_back(1.0);
    for(int i=0; i<SEPARABILIDAD_ACUM.rows; i++)
        GRAP.push_back(2.0);
    vector<Scalar> COLORES;
    Scalar COLOR;
    COLOR[0]=0;
    COLOR[1]=0;
    COLOR[2]=255;
    COLORES.push_back(COLOR);
    COLOR[0]=0;
    COLOR[1]=255;
    COLOR[2]=0;
    COLORES.push_back(COLOR);
    Mat Separabilidad=Mat::zeros(SEPARABILIDAD.rows+SEPARABILIDAD_ACUM.rows,2,CV_32F);
    for(int i=0; i<SEPARABILIDAD.rows; i++)
        SEPARABILIDAD.row(i).copyTo(Separabilidad.row(i));
    for(int i=1; i<SEPARABILIDAD_ACUM.rows+1; i++)
        SEPARABILIDAD_ACUM.row(i-1).copyTo(Separabilidad.row(SEPARABILIDAD.rows-1+i));
    Mat most=Mat::zeros(150,400,CV_8UC3);
    most=most+Scalar(255,255,255);
    String texto="SEPARABILIDAD";
    putText(most,texto,Point(10,50),1,1.5,COLORES[0],2);
    texto="SEPARABILIDAD ACUMULADA";
    putText(most,texto,Point(10,100),1,1.5,COLORES[1],2);
    imshow("Leyenda",most);
    rep.Continuous_data_represent("CALIDAD DIMENSIONES "+Dat_Ref, Separabilidad,GRAP,COLORES);
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_Generar_3_clicked()
{
    if(IMAGENES.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay datos cargados");
        msgBox.exec();
        return;
    }
    if(LABELS.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay Etiquetas cargadas");
        msgBox.exec();
        return;
    }
    QString referencia=ui->Nombre_Datos_3->displayText();
    std::string ref=referencia.toStdString();
    for(uint i=0; i<ref.size(); i++){
        if(ref[i]==' '){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se permiten espacios en la referencia");
            msgBox.exec();
            return;
        }
    }
    if(ui->Num_dimensiones->value()>IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: El numero de dimensiones es mayor que el de los datos");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int e=0;
    Imagenes.clear();
    Dimensionalidad::Reducciones reduc;
    reduc.tam_reduc=ui->Num_dimensiones->value();
    if(ui->But_LDA->isChecked())
        reduc.si_lda=true;
    else if(ui->But_PCA->isChecked())
        reduc.si_pca=true;
    else if(ui->But_distmax->isChecked())
        reduc.si_dist=true;
    else if(ui->But_D_Prime->isChecked())
        reduc.si_d_prime=true;
    Dimensionalidad dim(ref);
    e=dim.Reducir(IMAGENES,Imagenes,LABELS,reduc,Info,save_other);
    if(e==0){
        ui->Representar_Dim->setEnabled(true);
        ui->Guardar_2->setEnabled(true);
        ui->Dimension_3->setMaximum((int)(Imagenes[0].rows*Imagenes[0].cols));
        ui->Dim_X_5->setMaximum((int)(Imagenes[0].rows*Imagenes[0].cols));
        ui->Dim_Y_5->setMaximum((int)(Imagenes[0].rows*Imagenes[0].cols));
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se ha podido hacer la reduccion");
        msgBox.exec();
    }
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_pushButton_4_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int e=0;
    vector<int> dim;
    if(ui->Dimension_3->value()>IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: La dimension en Histograma esta fuera de rango");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    if(ui->Dim_X_5->value()<=IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()){
        dim.push_back(ui->Dim_X_5->value());
        if(ui->Dim_Y_5->value()>0 && ui->Dim_Y_5->value()<=IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels())
            dim.push_back(ui->Dim_Y_5->value());
        else if(ui->Dim_Y_5->value()==0){
        }
        else{
            QMessageBox msgBox;
            msgBox.setText("ERROR: La dimension Y esta fuera de rango");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: La dimension X esta fuera de rango");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    Representacion rep;
    if(ui->But_datos_6->isChecked()){
        e=rep.Data_represent("DATO "+Dat_Ref+" "+ui->Nombre_Datos_3->text().toStdString(),Imagenes,LABELS,dim,Col);
    }
    else if(ui->But_Elipses_6->isChecked()){
        e=rep.Ellipse_represent("ELIPSES "+Dat_Ref+" "+ui->Nombre_Datos_3->text().toStdString(),Imagenes,LABELS,dim,Col);
    }
    else if(ui->But_Datos_Elipses_6->isChecked()){
        e=rep.Data_Ellipse_represent("DATOS CON ELIPSES "+Dat_Ref+" "+ui->Nombre_Datos_3->text().toStdString(),Imagenes,LABELS,dim,Col);
    }
    else if(ui->But_Hist_3->isChecked()){
        Analisis an;
        vector<vector<Mat> > Histo;
        vector<vector<int> > pos_barras;
        an.Histograma(Imagenes,LABELS,num_bar,Histo,pos_barras);
        e=rep.Histogram_represent("HISTOGRAMA "+Dat_Ref+" "+ui->Nombre_Datos_3->text().toStdString(),Histo,Col, ui->Dimension_3->value());
    }
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido representar los datos");
        msgBox.exec();
    }
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_Guardar_2_clicked()
{
    info.si_pca=Info.si_pca;
    info.si_lda=Info.si_lda;
    info.si_d_prime=Info.si_d_prime;
    info.si_dist=Info.si_dist;
    info.PCA=Info.PCA;
    info.LDA=Info.LDA;
    info.D_PRIME=Info.D_PRIME;
    info.DS=Info.DS;
    info.Tam_X=Info.Tam_X;
    info.Tam_Y=Info.Tam_Y;
    IMAGENES=Imagenes;
    QString referencia=ui->Nombre_Datos_3->displayText();
    std::string ref=referencia.toStdString();
    for(uint i=0; i<ref.size(); i++){
        if(ref[i]==' '){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se permiten espacios en la referencia");
            msgBox.exec();
            return;
        }
    }
    Generacion gen;
    gen.total_progreso=IMAGENES.size();
    gen.progreso=0;
    gen.base_progreso=0;
    gen.max_progreso=100;
    gen.window=ui;
    gen.Guardar_Datos(ref,IMAGENES,LABELS,info);
    ui->progress_Clasificar->setValue(100);
    ui->progress_Clasificar->setValue(0);
    ui->progress_generar->setValue(100);
    ui->progress_generar->setValue(0);
    ui->progress_Cargar->setValue(100);
    ui->progress_Cargar->setValue(0);
    ui->progress_Clus->setValue(100);
    ui->progress_Clus->setValue(0);
    ui->progress_Dimensionalidad->setValue(100);
    ui->progress_Dimensionalidad->setValue(0);
    ui->progress_Dimensionalidad->setValue(0);
    Dat_Ref=ref;
    ui->label_Datos->setText("Datos ref: "+referencia);
}


void MainWindow::on_toolButton_4_clicked()
{
    QString filename= QFileDialog::getExistingDirectory(
                this,
                tr("SELECCIONAR CARPETA"),
                QDir::currentPath()+"/../Data/Configuracion");
    ui->Direccion_Carga_2->setText(filename);
}

void MainWindow::on_toolButton_14_clicked()
{
    if(ui->Tipo_Clasif_2->currentIndex()==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: El clasificador Distancias no tiene parametros");
        msgBox.exec();
        return;
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==2){
        QMessageBox msgBox;
        msgBox.setText("ERROR: El clasificador Bayesiano(Gauss) no tiene parametros");
        msgBox.exec();
        return;
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==3){
        Conf_Histograma *nueva_ventana=new Conf_Histograma(this,this);
        nueva_ventana->show();
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==4){
        Conf_KNN *nueva_ventana=new Conf_KNN(this,this);
        nueva_ventana->show();
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==5){
        Conf_neuronal *nueva_ventana=new Conf_neuronal(this,this);
        nueva_ventana->show();
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==6){
        Conf_SVM *nueva_ventana=new Conf_SVM(this,this);
        nueva_ventana->show();
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==7){
        Conf_DTrees *nueva_ventana=new Conf_DTrees(this,this);
        nueva_ventana->show();
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==8){
        Conf_RTrees *nueva_ventana=new Conf_RTrees(this,this);
        nueva_ventana->show();
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==9){
        Conf_Boosting *nueva_ventana=new Conf_Boosting(this,this);
        nueva_ventana->show();
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==10){
        Conf_HAAR *nueva_ventana=new Conf_HAAR(this,this);
        nueva_ventana->show();
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==11){
        Conf_HAAR *nueva_ventana=new Conf_HAAR(this,this);
        nueva_ventana->show();
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==12){
        Conf_EM *nueva_ventana=new Conf_EM(this,this);
        nueva_ventana->show();
    }
//        else if(ui->Tipo_Clasif->currentIndex()==14){
//                Conf_Histograma *nueva_ventana=new Conf_Histograma(this,this);
//                nueva_ventana->show();
//        }
//        else if(ui->Tipo_Clasif->currentIndex()==15){
//            Conf_Histograma *nueva_ventana=new Conf_Histograma(this,this);
//            nueva_ventana->show();
//        }

}

void MainWindow::on_Iniciar_2_clicked()
{
    if(IMAGENES.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay datos cargados");
        msgBox.exec();
        return;
    }
    if(LABELS.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay Etiquetas cargadas");
        msgBox.exec();
        return;
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QString nom=ui->Nombre_Clasif->displayText();
    std::string nombre=nom.toStdString();
    for(uint i=0; i<nombre.size(); i++){
        if(nombre[i]==' '){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se permiten espacios en la referencia");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
    }
    int e=0;
    if(ui->Tipo_Clasif_2->currentIndex()==1){
        ID=DISTANCIAS;
        D.nombre=nombre;
        e=D.Parametrizar();
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=D.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==2){
        ID=GAUSSIANO;
        G.nombre=nombre;
        e=G.Parametrizar();
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=G.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==3){
        ID=HISTOGRAMA;
        H.nombre=nombre;
        e=H.Parametrizar(Hist_tam_celda);
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=H.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==4){
        ID=KNN;
        K.nombre=nombre;
        e=K.Parametrizar(KNN_k,KNN_regression);
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=K.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==5){
        ID=NEURONAL;
        N.nombre=nombre;
        if(Neuronal_layerSize.rows<3){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se ha configurado el clasificador adecuadamente");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
        Neuronal_layerSize.row(0)=IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels();
        Auxiliares aux;
        bool negativa;
        int numero=aux.numero_etiquetas(LABELS,negativa);
        Neuronal_layerSize.row(Neuronal_layerSize.rows-1)=numero;
        e=N.Parametrizar(Neuronal_layerSize,Neuronal_Method,Neuronal_Function,Neuronal_bp_dw_scale,Neuronal_bp_moment_scale,Neuronal_rp_dw0,Neuronal_rp_dw_max,Neuronal_rp_dw_min,Neuronal_rp_dw_minus,Neuronal_rp_dw_plus,Neuronal_fparam1,Neuronal_fparam2);
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=N.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==6){
        ID=C_SVM;
        S.nombre=nombre;
        e=S.Parametrizar(SVM_train,SVM_Type,SVM_kernel_type,Mat(),SVM_degree,SVM_gamma,SVM_coef0,SVM_C,SVM_nu,SVM_p);
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=S.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==7){
        ID=DTREES;
        DT.nombre=nombre;
        e=DT.Parametrizar(DTrees_max_depth,DTrees_min_sample_count,DTrees_regression_accuracy,DTrees_use_surrogates,DTrees_max_categories,DTrees_cv_folds,DTrees_use_1se_rule,DTrees_truncate_pruned_tree,DTrees_priors);
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=DT.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==8){
        ID=RTREES;
        RT.nombre=nombre;
        e=RT.Parametrizar(RTrees_max_depth,RTrees_min_sample_count,RTrees_regression_accuracy,RTrees_use_surrogates,RTrees_max_categories,RTrees_cv_folds,RTrees_use_1se_rule,RTrees_truncate_pruned_tree,RTrees_priors,RTrees_calc_var_importance,RTrees_native_vars);
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=RT.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==9){
        Auxiliares aux;
        bool neg;
        if(aux.numero_etiquetas(LABELS,neg)!=2){
            QMessageBox msgBox;
            msgBox.setText("ERROR: Boosting solo se puede usar con dos clases");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
        ID=BOOSTING;
        B.nombre=nombre;
        e=B.Parametrizar(Boosting_boost_type,Boosting_weak_count,Boosting_weight_trim_rate,Boosting_max_depth,Boosting_use_surrogates,Boosting_priors);
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=B.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==10){
        ID=CASCADA_CLAS;
        HA.nombre=nombre;
        e=HA.Parametrizar("HAAR",si_entrenar,Cascada_NumPos,Cascada_NumNeg,Cascada_Mode,Cascada_NumStage,Cascada_MinHitRate,Cascada_MaxFalseAlarmRate,Cascada_WeightTrimRate,Cascada_MaxWeakCount,Cascada_MaxDepth,Cascada_Bt,Cascada_PrecalcValBufSize,Cascada_PrecalcidxBufSize);
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=HA.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==11){
        ID=CASCADA_CLAS;
        HA.nombre=nombre;
        e=HA.Parametrizar("LBP",si_entrenar,Cascada_NumPos,Cascada_NumNeg,Cascada_Mode,Cascada_NumStage,Cascada_MinHitRate,Cascada_MaxFalseAlarmRate,Cascada_WeightTrimRate,Cascada_MaxWeakCount,Cascada_MaxDepth,Cascada_Bt,Cascada_PrecalcValBufSize,Cascada_PrecalcidxBufSize);
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=HA.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
    else if(ui->Tipo_Clasif_2->currentIndex()==12){
        ID=EXP_MAX;
        E.nombre=nombre;
        e=E.Parametrizar(EM_nclusters,EM_covMatType);
        if(e==0){
            Dimensionalidad::Reducciones reduc;
            e=E.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
        }
    }
//    else if(ui->Tipo_Clasif_2->currentIndex()==13){
//        Auxiliares aux;
//        bool neg;
//        if(aux.numero_etiquetas(LABELS,neg)!=2){
//            QMessageBox msgBox;
//            msgBox.setText("ERROR: GBTrees solo se puede usar con dos clases");
//            msgBox.exec();
//            QApplication::restoreOverrideCursor();
//            return;
//        }
//        ID=GBT;
//        GB.nombre=nombre;
//        e=GB.Parametrizar(GBT_loss_function_type,GBT_weak_count,GBT_shrinkage,GBT_subsample_portion,GBT_max_depth,GBT_use_surrogates);
//        if(e==0){
//            Dimensionalidad::Reducciones reduc;
//            e=GB.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
//        }
//    }
//    else if(ui->Tipo_Clasif_2->currentIndex()==14){
//        ID=ERTREES;
//        ER.nombre=nombre;
//        e=ER.Parametrizar(ERTrees_max_depth,ERTrees_min_sample_count,ERTrees_regression_accuracy,ERTrees_use_surrogates,ERTrees_max_categories,ERTrees_cv_folds,ERTrees_use_1se_rule,ERTrees_truncate_pruned_tree,ERTrees_priors,ERTrees_calc_var_importance,ERTrees_native_vars);
//        if(e==0){
//            Dimensionalidad::Reducciones reduc;
//            e=ER.Autotrain(IMAGENES,LABELS,reduc,info,save_clasif);
//        }
//    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: Selecciona un clasificador");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se ha podido entrenar el clasificador");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    ui->label_Clasif->setText("Clasificador ref: "+nom);
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_Cargar_3_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int e=0;
    QString direccion=ui->Direccion_Carga_2->displayText();
    std::string Dir=direccion.toStdString();
    int pos=0;
    for(uint i=0; i<Dir.size(); i++){
        if(Dir[i]=='/')
            pos=i;
    }
    std::string nombre;
    for(uint i=pos+1; i<Dir.size(); i++)
        nombre=nombre+Dir[i];
    QString nom=QString::fromStdString(nombre);
    string archivo=Dir+"/Clasificador.xml";
    cv::FileStorage archivo_r(archivo,CV_STORAGE_READ);
    int id;
    if(archivo_r.isOpened()){
        archivo_r["Tipo"]>>id;
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: La carpeta no tiene la estructura utilizada por el sistema");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    if(id==DISTANCIAS){
        D.nombre=nombre;
        e=D.Read_Data();
    }
    else if(id==GAUSSIANO){
        G.nombre=nombre;
        e=G.Read_Data();
    }
    else if(id==CASCADA_CLAS){
        HA.nombre=nombre;
        HA.Read_Data();
    }
    else if(id==HISTOGRAMA){
        H.nombre=nombre;
        e=H.Read_Data();
    }
    else if(id==KNN){
        K.nombre=nombre;
        e=K.Read_Data();
    }
    else if(id==NEURONAL){
        N.nombre=nombre;
        e=N.Read_Data();
    }
    else if(id==C_SVM){
        S.nombre=nombre;
        e=S.Read_Data();
    }
    else if(id==RTREES){
        RT.nombre=nombre;
        e=RT.Read_Data();
    }
    else if(id==DTREES){
        DT.nombre=nombre;
        e=DT.Read_Data();
    }
    else if(id==BOOSTING){
        B.nombre=nombre;
        e=B.Read_Data();
    }
    else if(id==EXP_MAX){
        E.nombre=nombre;
        e=E.Read_Data();
    }
//    else if(id==GBT){
//        GB.nombre=nombre;
//        e=GB.Read_Data();
//    }
//    else if(id==ERTREES){
//        ER.nombre=nombre;
//        e=ER.Read_Data();
//    }
    else if(id==MICLASIFICADOR){
        MC.nombre=nombre;
        e=MC.Read_Data();
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: La carpeta no tiene la estructura utilizada por el sistema");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    ID=id;
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido cargar los datos");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    ui->label_Clasif->setText("Clasificador ref: "+nom);
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_Iniciar_3_clicked()
{
    resultado.clear();
    if(IMAGENES.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay Datos cargados");
        msgBox.exec();
        return;
    }
    QString nom=ui->Nombre_Clasif_2->displayText();
    std::string nombre=nom.toStdString();
    for(uint i=0; i<nombre.size(); i++){
        if(nombre[i]==' '){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se permiten espacios en la referencia");
            msgBox.exec();
            return;
        }
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int e=0;
    if(ui->Clasif_Cargado->isChecked()){
        if(ID==DISTANCIAS){
            D.progreso=0;
            D.max_progreso=100;
            D.base_progreso=0;
            D.total_progreso=IMAGENES.size();
            D.window=ui;
            e=D.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
        else if(ID==GAUSSIANO){
            G.progreso=0;
            G.max_progreso=100;
            G.base_progreso=0;
            G.total_progreso=IMAGENES.size();
            G.window=ui;
            e=G.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
        else if(ID==CASCADA_CLAS){
            HA.progreso=0;
            HA.max_progreso=100;
            HA.base_progreso=0;
            HA.total_progreso=IMAGENES.size();
            HA.window=ui;
            e=HA.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
        else if(ID==HISTOGRAMA){
            H.progreso=0;
            H.max_progreso=100;
            H.base_progreso=0;
            H.total_progreso=IMAGENES.size();
            H.window=ui;
            e=H.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
        else if(ID==KNN){
            K.progreso=0;
            K.max_progreso=100;
            K.base_progreso=0;
            K.total_progreso=IMAGENES.size();
            K.window=ui;
            e=K.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
        else if(ID==NEURONAL){
            N.progreso=0;
            N.max_progreso=100;
            N.base_progreso=0;
            N.total_progreso=IMAGENES.size();
            N.window=ui;
            e=N.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
        else if(ID==C_SVM){
            S.progreso=0;
            S.max_progreso=100;
            S.base_progreso=0;
            S.total_progreso=IMAGENES.size();
            S.window=ui;
            e=S.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
        else if(ID==RTREES){
            RT.progreso=0;
            RT.max_progreso=100;
            RT.base_progreso=0;
            RT.total_progreso=IMAGENES.size();
            RT.window=ui;
            e=RT.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
        else if(ID==DTREES){
            DT.progreso=0;
            DT.max_progreso=100;
            DT.base_progreso=0;
            DT.total_progreso=IMAGENES.size();
            DT.window=ui;
            e=DT.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
        else if(ID==BOOSTING){
            B.progreso=0;
            B.max_progreso=100;
            B.base_progreso=0;
            B.total_progreso=IMAGENES.size();
            B.window=ui;
            e=B.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
        else if(ID==EXP_MAX){
            E.progreso=0;
            E.max_progreso=100;
            E.base_progreso=0;
            E.total_progreso=IMAGENES.size();
            E.window=ui;
            e=E.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
//        else if(ID==GBT){
//            GB.progreso=0;
//            GB.max_progreso=100;
//            GB.base_progreso=0;
//            GB.total_progreso=IMAGENES.size();
//            GB.window=ui;
//            e=GB.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
//        }
//        else if(ID==ERTREES){
//            ER.progreso=0;
//            ER.max_progreso=100;
//            ER.base_progreso=0;
//            ER.total_progreso=IMAGENES.size();
//            ER.window=ui;
//            e=ER.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
//        }
        else if(ID==MICLASIFICADOR){
            MC.progreso=0;
            MC.max_progreso=100;
            MC.base_progreso=0;
            MC.total_progreso=IMAGENES.size();
            MC.window=ui;
            e=MC.Autoclasificacion(IMAGENES,resultado,ifreduc,read);
        }
        else{
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se ha cargado ningun clasificador");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
    }
    else if(ui->Multiclasif->isChecked()){
        vector<Clasificador*> clasificadores;
        for(uint i=0; i<id_clasificadores.size(); i++){
            if(id_clasificadores[i]==DISTANCIAS){
                Clasificador_Distancias *clasi=new Clasificador_Distancias(nombres[i]);
                clasi->Read_Data();
                clasificadores.push_back(clasi);
            }
            else if(id_clasificadores[i]==GAUSSIANO){
                Clasificador_Gaussiano *clasi=new Clasificador_Gaussiano(nombres[i]);
                clasi->Read_Data();
                clasificadores.push_back(clasi);
            }
            else if(id_clasificadores[i]==CASCADA_CLAS){
                Clasificador_Cascada *clasi=new Clasificador_Cascada(nombres[i]);
                clasi->Read_Data();
                clasificadores.push_back(clasi);
            }
            else if(id_clasificadores[i]==HISTOGRAMA){
                Clasificador_Histograma *clasi=new Clasificador_Histograma(nombres[i]);
                clasi->Read_Data();
                clasificadores.push_back(clasi);
            }
            else if(id_clasificadores[i]==KNN){
                Clasificador_KNN *clasi=new Clasificador_KNN(nombres[i]);
                clasi->Read_Data();
                clasificadores.push_back(clasi);
            }
            else if(id_clasificadores[i]==NEURONAL){
                Clasificador_Neuronal *clasi=new Clasificador_Neuronal(nombres[i]);
                clasi->Read_Data();
                clasificadores.push_back(clasi);
            }
            else if(id_clasificadores[i]==C_SVM){
                Clasificador_SVM *clasi=new Clasificador_SVM(nombres[i]);
                clasi->Read_Data();
                clasificadores.push_back(clasi);
            }
            else if(id_clasificadores[i]==RTREES){
                Clasificador_RTrees *clasi=new Clasificador_RTrees(nombres[i]);
                clasi->Read_Data();
                clasificadores.push_back(clasi);
            }
            else if(id_clasificadores[i]==DTREES){
                Clasificador_DTrees *clasi=new Clasificador_DTrees(nombres[i]);
                clasi->Read_Data();
                clasificadores.push_back(clasi);
            }
            else if(id_clasificadores[i]==BOOSTING){
                Clasificador_Boosting *clasi=new Clasificador_Boosting(nombres[i]);
                clasi->Read_Data();
                clasificadores.push_back(clasi);
            }
            else if(id_clasificadores[i]==EXP_MAX){
                Clasificador_EM *clasi=new Clasificador_EM(nombres[i]);
                clasi->Read_Data();
                clasificadores.push_back(clasi);
            }
    //        else if(id_clasificadores[i]==GBT){
    //            Clasificador_GBTrees *clasi=new Clasificador_GBTrees(nombres[i]);
//                clasi->Read_Data();
//                clasificadores.push_back(clasi);
    //        }
    //        else if(id_clasificadores[i]==ERTREES){
    //            Clasificador_ERTrees *clasi=new Clasificador_ERTrees(nombres[i]);
//                clasi->Read_Data();
//                clasificadores.push_back(clasi);
    //        }
        }
        MultiClasificador multi(clasificadores);
        multi.progreso=0;
        multi.max_progreso=100;
        multi.base_progreso=0;
        multi.total_progreso=IMAGENES.size();
        multi.window=ui;
        if(Multi_tipo.tipo==CASCADA){
            e=multi.Cascada(IMAGENES,Multi_tipo.tipo_regla,Multi_tipo.label_ref,resultado);
        }
        else if(Multi_tipo.tipo==VOTACION){
            e=multi.Votacion(IMAGENES,Multi_tipo.w_clasif,resultado);
        }
    }
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido clasificar los datos");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    stringstream txt;
    if(LABELS.size()==resultado.size()){
        txt<<"Num Dato      Etiqueta Original      Resultado"<<endl;
        for(uint i=0; i<resultado.size(); i++){
            txt<<i+1<<"                               "<<LABELS[i]<<"                       "<<resultado[i]<<endl;
            if(resultado[i]==0)
                e=1;
        }
    }
    else{
        txt<<"Num Dato      Resultado"<<endl;
        for(uint i=0; i<resultado.size(); i++){
            txt<<i+1<<"                       "<<resultado[i]<<endl;
            if(resultado[i]==0)
                e=1;
        }
    }
    if(e==1){
        ui->progress_Clasificar->setValue(100);
        ui->progress_Clasificar->setValue(0);
        ui->progress_generar->setValue(100);
        ui->progress_generar->setValue(0);
        ui->progress_Cargar->setValue(100);
        ui->progress_Cargar->setValue(0);
        ui->progress_Clus->setValue(100);
        ui->progress_Clus->setValue(0);
        ui->progress_Dimensionalidad->setValue(100);
        ui->progress_Dimensionalidad->setValue(0);
        QMessageBox msgBox;
        msgBox.setText("WARNING: Algunos datos no se han podido clasificar -> Etiquetas con valor 0");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
    }
    ui->Result_Clas->setText(QString::fromStdString(txt.str()));
    ui->label_Resultado->setText("Resultado Ref: "+nom);
    result_ref=nom.toStdString();
    ui->Clas_Representacion->setEnabled(true);
    ui->progress_Clasificar->setValue(100);
    ui->progress_Clasificar->setValue(0);
    ui->progress_generar->setValue(100);
    ui->progress_generar->setValue(0);
    ui->progress_Cargar->setValue(100);
    ui->progress_Cargar->setValue(0);
    ui->progress_Clus->setValue(100);
    ui->progress_Clus->setValue(0);
    ui->progress_Dimensionalidad->setValue(100);
    ui->progress_Dimensionalidad->setValue(0);
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_Analizar_3_clicked()
{
    if(resultado.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay datos clasificados");
        msgBox.exec();
        return;
    }
    if(LABELS.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay Etiquetas originales cargadas");
        msgBox.exec();
        return;
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int e=0;
    Auxiliares aux;
    bool negativa;
    int num=aux.numero_etiquetas(LABELS,negativa);
    ui->progress_Analisis_2->setValue(10);
    Analisis an;
    Mat Confusion;
    float error;
    e=an.Confusion(LABELS,resultado,Confusion,error);
    vector<Analisis::Ratios_data> Rat;
    e=an.Ratios(LABELS,resultado,Rat);
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido obtener los datos estadisticos");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        ui->progress_Analisis_2->setValue(0);
        return;
    }
    ui->progress_Analisis_2->setValue(30);
    QStandardItemModel *model=new QStandardItemModel(0,0);
    QStandardItem *ER= new QStandardItem(QString("Error"));
    QStandardItem *Er = new QStandardItem(QString("%1").arg(error));
    ER->appendRow(Er);
    model->appendRow(ER);
    QStandardItem *Conf = new QStandardItem(QString("Confusion"));
    for(int j=0; j<Confusion.rows; j++){
        stringstream linea_conf;
        for(int k=0; k<Confusion.cols; k++){
            linea_conf<<Confusion.at<float>(j,k);
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
    ui->progress_Analisis_2->setValue(50);
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
        QStandardItem *vp = new QStandardItem(QString("%1").arg(Rat[i].VP));
        VP->appendRow(vp);
        Lab->appendRow(VP);
        QStandardItem *VN = new QStandardItem(QString("VN"));
        QStandardItem *vn = new QStandardItem(QString("%1").arg(Rat[i].VN));
        VN->appendRow(vn);
        Lab->appendRow(VN);
        QStandardItem *FP = new QStandardItem(QString("FP"));
        QStandardItem *fp = new QStandardItem(QString("%1").arg(Rat[i].FP));
        FP->appendRow(fp);
        Lab->appendRow(FP);
        QStandardItem *FN = new QStandardItem(QString("FN"));
        QStandardItem *fn = new QStandardItem(QString("%1").arg(Rat[i].FN));
        FN->appendRow(fn);
        Lab->appendRow(FN);
        QStandardItem *TAR = new QStandardItem(QString("TAR"));
        QStandardItem *tar = new QStandardItem(QString("%1").arg(Rat[i].TAR));
        TAR->appendRow(tar);
        Lab->appendRow(TAR);
        QStandardItem *TRR = new QStandardItem(QString("TRR"));
        QStandardItem *trr = new QStandardItem(QString("%1").arg(Rat[i].TRR));
        TRR->appendRow(trr);
        Lab->appendRow(TRR);
        QStandardItem *FAR = new QStandardItem(QString("FAR"));
        QStandardItem *far = new QStandardItem(QString("%1").arg(Rat[i].FAR));
        FAR->appendRow(far);
        Lab->appendRow(FAR);
        QStandardItem *FRR = new QStandardItem(QString("FRR"));
        QStandardItem *frr = new QStandardItem(QString("%1").arg(Rat[i].FRR));
        FRR->appendRow(frr);
        Lab->appendRow(FRR);
        QStandardItem *PPV = new QStandardItem(QString("PPV"));
        QStandardItem *ppv = new QStandardItem(QString("%1").arg(Rat[i].PPV));
        PPV->appendRow(ppv);
        Lab->appendRow(PPV);
        QStandardItem *NPV = new QStandardItem(QString("NPV"));
        QStandardItem *npv = new QStandardItem(QString("%1").arg(Rat[i].NPV));
        NPV->appendRow(npv);
        Lab->appendRow(NPV);
        QStandardItem *FDR = new QStandardItem(QString("FDR"));
        QStandardItem *fdr = new QStandardItem(QString("%1").arg(Rat[i].FDR));
        FDR->appendRow(fdr);
        Lab->appendRow(FDR);
        QStandardItem *F1 = new QStandardItem(QString("F1"));
        QStandardItem *f1 = new QStandardItem(QString("%1").arg(Rat[i].F1));
        F1->appendRow(f1);
        Lab->appendRow(F1);
        QStandardItem *INFORMEDNESS = new QStandardItem(QString("INFORMEDNESS"));
        QStandardItem *informedness = new QStandardItem(QString("%1").arg(Rat[i].INFORMEDNESS));
        INFORMEDNESS->appendRow(informedness);
        Lab->appendRow(INFORMEDNESS);
        QStandardItem *MARKEDNESS = new QStandardItem(QString("MARKEDNESS"));
        QStandardItem *markedness = new QStandardItem(QString("%1").arg(Rat[i].MARKEDNESS));
        MARKEDNESS->appendRow(markedness);
        Lab->appendRow(MARKEDNESS);
        QStandardItem *EXP_ERROR = new QStandardItem(QString("EXP_ERROR"));
        QStandardItem *exp_error = new QStandardItem(QString("%1").arg(Rat[i].EXP_ERROR));
        EXP_ERROR->appendRow(exp_error);
        Lab->appendRow(EXP_ERROR);
        QStandardItem *LR_NEG = new QStandardItem(QString("LR_NEG"));
        QStandardItem *lr_neg = new QStandardItem(QString("%1").arg(Rat[i].LR_NEG));
        LR_NEG->appendRow(lr_neg);
        Lab->appendRow(LR_NEG);
        QStandardItem *LR_POS = new QStandardItem(QString("LR_POS"));
        QStandardItem *lr_pos = new QStandardItem(QString("%1").arg(Rat[i].LR_POS));
        LR_POS->appendRow(lr_pos);
        Lab->appendRow(LR_POS);
        QStandardItem *DOR = new QStandardItem(QString("DOR"));
        QStandardItem *dor = new QStandardItem(QString("%1").arg(Rat[i].DOR));
        DOR->appendRow(dor);
        Lab->appendRow(DOR);
        QStandardItem *ACC = new QStandardItem(QString("ACC"));
        QStandardItem *acc = new QStandardItem(QString("%1").arg(Rat[i].ACC));
        ACC->appendRow(acc);
        Lab->appendRow(ACC);
        QStandardItem *PREVALENCE = new QStandardItem(QString("PREVALENCE"));
        QStandardItem *prevalence = new QStandardItem(QString("%1").arg(Rat[i].PREVALENCE));
        PREVALENCE->appendRow(prevalence);
        Lab->appendRow(PREVALENCE);
        Rati->appendRow(Lab);
        ui->progress_Analisis_2->setValue(50+(50*i/num));
    }
    ui->Estadisticos_2->setModel(model);
    ui->progress_Analisis_2->setValue(100);
    ui->progress_Analisis_2->setValue(0);
    QApplication::restoreOverrideCursor();
}


void MainWindow::on_Clus_Representar_2_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int e=0;
    vector<int> dim;
    if(ui->Dimension_5->value()>IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: La dimension en Histograma esta fuera de rango");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    if(ui->Dim_X_6->value()<=IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels()){
        dim.push_back(ui->Dim_X_6->value());
        if(ui->Dim_Y_6->value()>0 && ui->Dim_Y_6->value()<=IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels())
            dim.push_back(ui->Dim_Y_6->value());
        else if(ui->Dim_Y_6->value()==0){
        }
        else{
            QMessageBox msgBox;
            msgBox.setText("ERROR: La dimension Y esta fuera de rango");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: La dimension X esta fuera de rango");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    Representacion rep;
    if(ui->But_datos_7->isChecked()){
        e=rep.Data_represent("DATOS "+result_ref,IMAGENES,resultado,dim,Col);
    }
    else if(ui->But_Elipses_7->isChecked()){
        e=rep.Ellipse_represent("ELIPSES "+result_ref,IMAGENES,resultado,dim,Col);
    }
    else if(ui->But_Datos_Elipses_7->isChecked()){
        e=rep.Data_Ellipse_represent("DATOS CON ELIPSES "+result_ref,IMAGENES,resultado,dim,Col);
    }
    else if(ui->But_Hist_5->isChecked()){
        Analisis an;
        vector<vector<Mat> > Histo;
        vector<vector<int> > pos_barras;
        an.Histograma(IMAGENES,resultado,num_bar,Histo,pos_barras);
        e=rep.Histogram_represent("HISTOGRAMA "+result_ref,Histo,Col, ui->Dimension_5->value());
    }
    else if(ui->But_Aciertos->isChecked()){
        vector<Scalar> colores_aciertos;
        bool todo_aciertos=true;
        vector<float> aciertos;
        for(int i=0; i<resultado.size(); i++){
            if(resultado[i]==LABELS[i])
                aciertos.push_back(1.);
            else{
                aciertos.push_back(-1.);
                todo_aciertos=false;
            }
        }
        if(todo_aciertos){
            Scalar col_acierto2=Scalar(0,255,0);
            colores_aciertos.push_back(col_acierto2);
        }
        else{
            Scalar col_acierto1=Scalar(0,0,255);
            colores_aciertos.push_back(col_acierto1);
            Scalar col_acierto2=Scalar(0,255,0);
            colores_aciertos.push_back(col_acierto2);
            vector<float> aciertos;
        }
        e=rep.Data_represent("ACIERTOS "+result_ref,IMAGENES,aciertos,dim,colores_aciertos);
    }
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido representar los datos");
        msgBox.exec();
    }
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_Iniciar_4_clicked()
{
    if(IMAGENES.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay datos cargados");
        msgBox.exec();
        return;
    }
    if(LABELS.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay Etiquetas cargadas");
        msgBox.exec();
        return;
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int e=0;
    Optimizacion op;
    op.total_progreso=IMAGENES.size();
    op.progreso=0;
    op.base_progreso=0;
    op.max_progreso=100;
    op.window=ui;
    if(ui->Validation->isChecked()){
        int id_clasificador;
        if(ui->Tipo_Clasif->currentIndex()==1)
            id_clasificador=DISTANCIAS;
        else if(ui->Tipo_Clasif->currentIndex()==2)
            id_clasificador=GAUSSIANO;
        else if(ui->Tipo_Clasif->currentIndex()==3)
            id_clasificador=HISTOGRAMA;
        else if(ui->Tipo_Clasif->currentIndex()==4)
            id_clasificador=KNN;
        else if(ui->Tipo_Clasif->currentIndex()==5){
            if(inicio.Neuronal_layerSize.rows<3){
                QMessageBox msgBox;
                msgBox.setText("ERROR: No se ha configurado el clasificador Neuronal adecuadamente");
                msgBox.exec();
                QApplication::restoreOverrideCursor();
                ui->progress_Clasificar->setValue(0);
                return;
            }
            inicio.Neuronal_layerSize.row(0)=IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels();
            Auxiliares aux;
            bool negativa;
            int numero=aux.numero_etiquetas(LABELS,negativa);
            Neuronal_layerSize.row(Neuronal_layerSize.rows-1)=numero;
            id_clasificador=NEURONAL;
        }
        else if(ui->Tipo_Clasif->currentIndex()==6)
            id_clasificador=C_SVM;
        else if(ui->Tipo_Clasif->currentIndex()==7)
            id_clasificador=RTREES;
        else if(ui->Tipo_Clasif->currentIndex()==8)
            id_clasificador=DTREES;
        else if(ui->Tipo_Clasif->currentIndex()==9)
            id_clasificador=BOOSTING;
        else if(ui->Tipo_Clasif->currentIndex()==10)
            id_clasificador=EXP_MAX;
//        else if(ui->Tipo_Clasif->currentIndex()==11)
//            id_clasificador=GBT;
//        else if(ui->Tipo_Clasif->currentIndex()==12)
//            id_clasificador=ERTREES;
        else{
            QMessageBox msgBox;
            msgBox.setText("ERROR: Selecciona un clasificador");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        float Error;
        Mat Confusion;
        vector<Analisis::Ratios_data> Ratios;
        e=op.Validation(IMAGENES,LABELS,ui->Porcentaje->value(),id_clasificador,inicio,Error,Confusion,Ratios);
        if(e==1){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se han podido obtener los datos estadisticos");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        stringstream texto;
        texto<<"Error=";
        texto<<Error;
        texto<<endl;
        texto<<"Confusion="<<endl;
        for(int i=0; i<Confusion.cols; i++){
            for(int j=0; j<Confusion.rows; j++){
                texto<<Confusion.at<float>(j,i);
                texto<<"    ";
            }
            texto<<endl;
        }
        Auxiliares aux;
        bool negativa;
        int num=aux.numero_etiquetas(LABELS,negativa);
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
            texto<<"Etiqueta "<<etiqueta<<":"<<endl;
            texto<<"VP="<<Ratios[i].VP;
            texto<<"    VN="<<Ratios[i].VN;
            texto<<"    FP="<<Ratios[i].FP;
            texto<<"    FN="<<Ratios[i].FN;
            texto<<"    TAR="<<Ratios[i].TAR;
            texto<<"    TRR="<<Ratios[i].TRR;
            texto<<"    FAR="<<Ratios[i].FAR;
            texto<<"    FRR="<<Ratios[i].FRR;
            texto<<"    PPV="<<Ratios[i].PPV;
            texto<<"    NPV="<<Ratios[i].NPV;
            texto<<"    FDR="<<Ratios[i].FDR;
            texto<<"    F1="<<Ratios[i].F1;
            texto<<"    INFORMEDNESS="<<Ratios[i].INFORMEDNESS;
            texto<<"    MARKEDNESS="<<Ratios[i].MARKEDNESS;
            texto<<"    EXP_ERROR="<<Ratios[i].EXP_ERROR;
            texto<<"    LR_POS="<<Ratios[i].LR_POS;
            texto<<"    LR_NEG="<<Ratios[i].LR_NEG;
            texto<<"    DOR="<<Ratios[i].DOR;
            texto<<"    ACC="<<Ratios[i].ACC;
            texto<<"    PREVALENCE="<<Ratios[i].PREVALENCE<<endl;
        }
        ui->Texto_Result->setText(QString::fromStdString(texto.str()));
    }
    else if(ui->Validation2->isChecked()){
        float Error;
        Mat Confusion;
        vector<Analisis::Ratios_data> Ratios;
        e=op.Validation(IMAGENES,LABELS,ui->Porcentaje->value(),id_clasificadores,inicio,Multi_tipo,Error,Confusion,Ratios);
        if(e==1){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se han podido obtener los datos estadisticos");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        stringstream texto;
        texto<<"Error=";
        texto<<Error;
        texto<<endl;
        texto<<"Confusion="<<endl;
        for(int i=0; i<Confusion.cols; i++){
            for(int j=0; j<Confusion.rows; j++){
                texto<<Confusion.at<float>(j,i);
                texto<<"    ";
            }
            texto<<endl;
        }
        Auxiliares aux;
        bool negativa;
        int num=aux.numero_etiquetas(LABELS,negativa);
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
            texto<<"Etiqueta "<<etiqueta<<":"<<endl;
            texto<<"VP="<<Ratios[i].VP;
            texto<<"    VN="<<Ratios[i].VN;
            texto<<"    FP="<<Ratios[i].FP;
            texto<<"    FN="<<Ratios[i].FN;
            texto<<"    TAR="<<Ratios[i].TAR;
            texto<<"    TRR="<<Ratios[i].TRR;
            texto<<"    FAR="<<Ratios[i].FAR;
            texto<<"    FRR="<<Ratios[i].FRR;
            texto<<"    PPV="<<Ratios[i].PPV;
            texto<<"    NPV="<<Ratios[i].NPV;
            texto<<"    FDR="<<Ratios[i].FDR;
            texto<<"    F1="<<Ratios[i].F1;
            texto<<"    INFORMEDNESS="<<Ratios[i].INFORMEDNESS;
            texto<<"    MARKEDNESS="<<Ratios[i].MARKEDNESS;
            texto<<"    EXP_ERROR="<<Ratios[i].EXP_ERROR;
            texto<<"    LR_POS="<<Ratios[i].LR_POS;
            texto<<"    LR_NEG="<<Ratios[i].LR_NEG;
            texto<<"    DOR="<<Ratios[i].DOR;
            texto<<"    ACC="<<Ratios[i].ACC;
            texto<<"    PREVALENCE="<<Ratios[i].PREVALENCE<<endl;
        }
        ui->Texto_Result->setText(QString::fromStdString(texto.str()));
    }
    else if(ui->C_Validation->isChecked()){
        int id_clasificador;
        if(ui->Tipo_Clasif->currentIndex()==1)
            id_clasificador=DISTANCIAS;
        else if(ui->Tipo_Clasif->currentIndex()==2)
            id_clasificador=GAUSSIANO;
        else if(ui->Tipo_Clasif->currentIndex()==3)
            id_clasificador=HISTOGRAMA;
        else if(ui->Tipo_Clasif->currentIndex()==4)
            id_clasificador=KNN;
        else if(ui->Tipo_Clasif->currentIndex()==5){
            if(inicio.Neuronal_layerSize.rows<3){
                QMessageBox msgBox;
                msgBox.setText("ERROR: No se ha configurado el clasificador Neuronal adecuadamente");
                msgBox.exec();
                QApplication::restoreOverrideCursor();
                ui->progress_Clasificar->setValue(0);
                return;
            }
            inicio.Neuronal_layerSize.row(0)=IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels();
            Auxiliares aux;
            bool negativa;
            int numero=aux.numero_etiquetas(LABELS,negativa);
            Neuronal_layerSize.row(Neuronal_layerSize.rows-1)=numero;
            id_clasificador=NEURONAL;
        }
        else if(ui->Tipo_Clasif->currentIndex()==6)
            id_clasificador=C_SVM;
        else if(ui->Tipo_Clasif->currentIndex()==7)
            id_clasificador=RTREES;
        else if(ui->Tipo_Clasif->currentIndex()==8)
            id_clasificador=DTREES;
        else if(ui->Tipo_Clasif->currentIndex()==9)
            id_clasificador=BOOSTING;
        else if(ui->Tipo_Clasif->currentIndex()==10)
            id_clasificador=EXP_MAX;
//        else if(ui->Tipo_Clasif->currentIndex()==11)
//            id_clasificador=GBT;
//        else if(ui->Tipo_Clasif->currentIndex()==12)
//            id_clasificador=ERTREES;
        else{
            QMessageBox msgBox;
            msgBox.setText("ERROR: Selecciona un clasificador");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        if((uint)ui->Num_folds->value()*ui->Tam_Folds->value()>IMAGENES.size()){
            QMessageBox msgBox;
            msgBox.setText("ERROR: Numero de datos menor de lo que se pide para el proceso");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        float Error=0;
        Mat Confus;
        Optimizacion::Parametros parametros;
        e=op.Cross_Validation(IMAGENES,LABELS,ui->Num_folds->value(),ui->Tam_Folds->value(),id_clasificador,inicio,fin,salto,parametros,Error,Confus);
        if(e==1){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se han podido obtener los datos estadisticos");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        stringstream texto;
        if(id_clasificador==DISTANCIAS){
            QMessageBox msgBox;
            msgBox.setText("ERROR: El clasificador Distancias no tiene parametros");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        else if(id_clasificador==GAUSSIANO){
            QMessageBox msgBox;
            msgBox.setText("ERROR: El clasificador Bayesiano(Gauss) no tiene parametros");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        else if(id_clasificador==CASCADA_CLAS){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No implementado");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        else if(id_clasificador==HISTOGRAMA){
            texto<<"Parametros optimizados"<<endl;
            texto<<"Hist_tam_celdea= "<<parametros.Hist_tam_celda<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificador==KNN){
            texto<<"Parametros optimizados"<<endl;
            texto<<"KNN_k= "<<parametros.KNN_k<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificador==NEURONAL){
            texto<<"Parametros optimizados"<<endl;
            texto<<"Neuronal_bp_dw_scale= "<<parametros.Neuronal_bp_dw_scale<<endl;
            texto<<"Neuronal_bp_moment_scale= "<<parametros.Neuronal_bp_moment_scale<<endl;
            texto<<"Neuronal_rp_dw0= "<<parametros.Neuronal_rp_dw0<<endl;
            texto<<"Neuronal_rp_dw_max= "<<parametros.Neuronal_rp_dw_max<<endl;
            texto<<"Neuronal_rp_dw_min= "<<parametros.Neuronal_rp_dw_min<<endl;
            texto<<"Neuronal_rp_dw_minus= "<<parametros.Neuronal_rp_dw_minus<<endl;
            texto<<"Neuronal_rp_dw_plus= "<<parametros.Neuronal_rp_dw_plus<<endl;
            texto<<"Neuronal_fparam1= "<<parametros.Neuronal_fparam1<<endl;
            texto<<"Neuronal_fparam2= "<<parametros.Neuronal_fparam2<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificador==C_SVM){
            texto<<"Parametros optimizados"<<endl;
            texto<<"SVM_C= "<<parametros.SVM_C<<endl;
            texto<<"SVM_gamma= "<<parametros.SVM_gamma<<endl;
            texto<<"SVM_p= "<<parametros.SVM_p<<endl;
            texto<<"SVM_nu= "<<parametros.SVM_nu<<endl;
            texto<<"SVM_coef0= "<<parametros.SVM_coef0<<endl;
            texto<<"SVM_degree= "<<parametros.SVM_degree<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificador==RTREES){
            texto<<"Parametros optimizados"<<endl;
            texto<<"RTrees_max_depth= "<<parametros.RTrees_max_depth<<endl;
            texto<<"RTrees_min_sample_count= "<<parametros.RTrees_min_sample_count<<endl;
            texto<<"RTrees_regression_accuracy= "<<parametros.RTrees_regression_accuracy<<endl;
            texto<<"RTrees_max_categories= "<<parametros.RTrees_max_categories<<endl;
            texto<<"RTrees_cv_folds= "<<parametros.RTrees_cv_folds<<endl;
            texto<<"RTrees_native_vars= "<<parametros.RTrees_native_vars<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificador==DTREES){
            texto<<"Parametros optimizados"<<endl;
            texto<<"DTrees_max_depth= "<<parametros.DTrees_max_depth<<endl;
            texto<<"DTrees_min_sample_count= "<<parametros.DTrees_min_sample_count<<endl;
            texto<<"DTrees_regression_accuracy= "<<parametros.DTrees_regression_accuracy<<endl;
            texto<<"DTrees_max_categories= "<<parametros.DTrees_max_categories<<endl;
            texto<<"DTrees_cv_folds= "<<parametros.DTrees_cv_folds<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificador==BOOSTING){
            texto<<"Parametros optimizados"<<endl;
            texto<<"Boosting_max_depth= "<<parametros.Boosting_max_depth<<endl;
            texto<<"Boosting_weak_count= "<<parametros.Boosting_weak_count<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificador==EXP_MAX){
            texto<<"Parametros optimizados"<<endl;
            texto<<"EM_nclusters= "<<parametros.EM_nclusters<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
//        else if(id_clasificador==GBT){
//            texto<<"Parametros optimizados"<<endl;
//            texto<<"GBT_weak_count= "<<parametros.GBT_weak_count<<endl;
//            texto<<"GBT_shrinkage= "<<parametros.GBT_shrinkage<<endl;
//            texto<<"GBT_max_depth= "<<parametros.GBT_max_depth<<endl;
//            texto<<"Error= "<<Error<<endl;
//            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
//        }
//        else if(id_clasificador==ERTREES){
//            texto<<"Parametros optimizados"<<endl;
//            texto<<"ERTrees_max_depth= "<<parametros.ERTrees_max_depth<<endl;
//            texto<<"ERTrees_min_sample_count= "<<parametros.ERTrees_min_sample_count<<endl;
//            texto<<"ERTrees_regression_accuracy= "<<parametros.ERTrees_regression_accuracy<<endl;
//            texto<<"ERTrees_max_categories= "<<parametros.ERTrees_max_categories<<endl;
//            texto<<"ERTrees_cv_folds= "<<parametros.ERTrees_cv_folds<<endl;
//            texto<<"ERTrees_native_vars= "<<parametros.ERTrees_native_vars<<endl;
//            texto<<"Error= "<<Error<<endl;
//            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
//        }
        ui->Texto_Result->setText(QString::fromStdString(texto.str()));
    }
    else if(ui->SC_Validation->isChecked()){
        if((uint)ui->Num_folds->value()*ui->Tam_Folds->value()>IMAGENES.size()){
            QMessageBox msgBox;
            msgBox.setText("ERROR: Numero de datos menor de lo que se pide para el proceso");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        float Error=0;
        Mat Confus;
        Optimizacion::Parametros parametros;
        e=op.Super_Cross_Validation(IMAGENES,LABELS,ui->Num_folds->value(),ui->Tam_Folds->value(),id_clasificadores,inicio,fin,salto,parametros,Error,Confus);
        if(e==1){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se han podido obtener los datos estadisticos");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        stringstream texto;
        if(id_clasificadores[0]==DISTANCIAS){
            texto<<"El mejor clasificador es Clasificador_Distancias"<<endl;
        }
        else if(id_clasificadores[0]==GAUSSIANO){
            texto<<"El mejor clasificador es Clasificador_Gaussiano"<<endl;
        }
        else if(id_clasificadores[0]==CASCADA_CLAS){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No implementado");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        else if(id_clasificadores[0]==HISTOGRAMA){
            texto<<"El mejor clasificador es Clasificador_Histograma"<<endl;
            texto<<"Parametros optimizados"<<endl;
            texto<<"Hist_tam_celdea= "<<parametros.Hist_tam_celda<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificadores[0]==KNN){
            texto<<"El mejor clasificador es Clasificador_KNN"<<endl;
            texto<<"Parametros optimizados"<<endl;
            texto<<"KNN_k= "<<parametros.KNN_k<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificadores[0]==NEURONAL){
            texto<<"El mejor clasificador es Clasificador_Neuronal"<<endl;
            texto<<"Parametros optimizados"<<endl;
            texto<<"Neuronal_bp_dw_scale= "<<parametros.Neuronal_bp_dw_scale<<endl;
            texto<<"Neuronal_bp_moment_scale= "<<parametros.Neuronal_bp_moment_scale<<endl;
            texto<<"Neuronal_rp_dw0= "<<parametros.Neuronal_rp_dw0<<endl;
            texto<<"Neuronal_rp_dw_max= "<<parametros.Neuronal_rp_dw_max<<endl;
            texto<<"Neuronal_rp_dw_min= "<<parametros.Neuronal_rp_dw_min<<endl;
            texto<<"Neuronal_rp_dw_minus= "<<parametros.Neuronal_rp_dw_minus<<endl;
            texto<<"Neuronal_rp_dw_plus= "<<parametros.Neuronal_rp_dw_plus<<endl;
            texto<<"Neuronal_fparam1= "<<parametros.Neuronal_fparam1<<endl;
            texto<<"Neuronal_fparam2= "<<parametros.Neuronal_fparam2<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificadores[0]==C_SVM){
            texto<<"El mejor clasificador es Clasificador_SVM"<<endl;
            texto<<"Parametros optimizados"<<endl;
            texto<<"SVM_C= "<<parametros.SVM_C<<endl;
            texto<<"SVM_gamma= "<<parametros.SVM_gamma<<endl;
            texto<<"SVM_p= "<<parametros.SVM_p<<endl;
            texto<<"SVM_nu= "<<parametros.SVM_nu<<endl;
            texto<<"SVM_coef0= "<<parametros.SVM_coef0<<endl;
            texto<<"SVM_degree= "<<parametros.SVM_degree<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificadores[0]==RTREES){
            texto<<"El mejor clasificador es Clasificador_RTrees"<<endl;
            texto<<"Parametros optimizados"<<endl;
            texto<<"RTrees_max_depth= "<<parametros.RTrees_max_depth<<endl;
            texto<<"RTrees_min_sample_count= "<<parametros.RTrees_min_sample_count<<endl;
            texto<<"RTrees_regression_accuracy= "<<parametros.RTrees_regression_accuracy<<endl;
            texto<<"RTrees_max_categories= "<<parametros.RTrees_max_categories<<endl;
            texto<<"RTrees_cv_folds= "<<parametros.RTrees_cv_folds<<endl;
            texto<<"RTrees_native_vars= "<<parametros.RTrees_native_vars<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificadores[0]==DTREES){
            texto<<"El mejor clasificador es Clasificador_DTrees"<<endl;
            texto<<"Parametros optimizados"<<endl;
            texto<<"DTrees_max_depth= "<<parametros.DTrees_max_depth<<endl;
            texto<<"DTrees_min_sample_count= "<<parametros.DTrees_min_sample_count<<endl;
            texto<<"DTrees_regression_accuracy= "<<parametros.DTrees_regression_accuracy<<endl;
            texto<<"DTrees_max_categories= "<<parametros.DTrees_max_categories<<endl;
            texto<<"DTrees_cv_folds= "<<parametros.DTrees_cv_folds<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificadores[0]==BOOSTING){
            texto<<"El mejor clasificador es Clasificador_Boosting"<<endl;
            texto<<"Parametros optimizados"<<endl;
            texto<<"Boosting_max_depth= "<<parametros.Boosting_max_depth<<endl;
            texto<<"Boosting_weak_count= "<<parametros.Boosting_weak_count<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
        else if(id_clasificadores[0]==EXP_MAX){
            texto<<"El mejor clasificador es Clasificador_EM"<<endl;
            texto<<"Parametros optimizados"<<endl;
            texto<<"EM_nclusters= "<<parametros.EM_nclusters<<endl;
            texto<<"Error= "<<Error<<endl;
            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
        }
//        else if(id_clasificadores[0]==GBT){
//            cout<<"El mejor clasificador es Clasificador_GBT"<<endl;
//            texto<<"Parametros optimizados"<<endl;
//            texto<<"GBT_weak_count= "<<parametros.GBT_weak_count<<endl;
//            texto<<"GBT_shrinkage= "<<parametros.GBT_shrinkage<<endl;
//            texto<<"GBT_max_depth= "<<parametros.GBT_max_depth<<endl;
//            texto<<"Error= "<<Error<<endl;
//            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
//        }
//        else if(id_clasificadores[0]==ERTREES){
//            texto<<"El mejor clasificador es Clasificador_ERTrees"<<endl;
//            texto<<"Parametros optimizados"<<endl;
//            texto<<"ERTrees_max_depth= "<<parametros.ERTrees_max_depth<<endl;
//            texto<<"ERTrees_min_sample_count= "<<parametros.ERTrees_min_sample_count<<endl;
//            texto<<"ERTrees_regression_accuracy= "<<parametros.ERTrees_regression_accuracy<<endl;
//            texto<<"ERTrees_max_categories= "<<parametros.ERTrees_max_categories<<endl;
//            texto<<"ERTrees_cv_folds= "<<parametros.ERTrees_cv_folds<<endl;
//            texto<<"ERTrees_native_vars= "<<parametros.ERTrees_native_vars<<endl;
//            texto<<"Error= "<<Error<<endl;
//            texto<<"Matriz Confusion= "<<endl<<Confus<<endl;
//        }
        ui->Texto_Result->setText(QString::fromStdString(texto.str()));
    }
    else if(ui->Ratios_Parametro->isChecked()){
        vector<float> valor_x,valor_y;
        vector<vector<Analisis::Ratios_data> > RAT;
        Optimizacion::Parametros param_aux=inicio;
        e=op.Ratios_parametro(IMAGENES,LABELS,ui->Porcentaje->value(),parametro,inicio,fin,salto,RAT);
        if(e==1){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se han podido obtener los datos estadisticos");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            ui->progress_Clasificar->setValue(0);
            return;
        }
        stringstream texto;
        Auxiliares aux;
        bool negativa;
        aux.numero_etiquetas(LABELS,negativa);
        vector<Analisis::Ratios_data> suma_parametro(RAT.size());
        for(uint j=0; j<RAT[0].size(); j++){
            int etiqueta;
            if(negativa){
                if(j==0)
                    etiqueta=-1;
                else
                    etiqueta=j;
            }
            else
                etiqueta=j+1;
            texto<<"ETIQUETA "<<etiqueta<<":"<<endl;
            for(uint i=0; i<RAT.size(); i++){
                if(parametro=="Hist_tam_celda"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Hist_tam_celda);
                        if(Y==0)
                            valor_y.push_back(inicio.Hist_tam_celda);
                    }
                    texto<<"Hist_tam_celda= "<<inicio.Hist_tam_celda<<endl;
                    inicio.Hist_tam_celda=inicio.Hist_tam_celda+salto.Hist_tam_celda;
                }
                else if(parametro=="KNN_k"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.KNN_k);
                        if(Y==0)
                            valor_y.push_back(inicio.KNN_k);
                    }
                    texto<<"KNN_k= "<<inicio.KNN_k<<endl;
                    inicio.KNN_k=inicio.KNN_k+salto.KNN_k;
                }
                else if(parametro=="Neuronal_bp_dw_scale"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Neuronal_bp_dw_scale);
                        if(Y==0)
                            valor_y.push_back(inicio.Neuronal_bp_dw_scale);
                    }
                    texto<<"Neuronal_bp_dw_scale= "<<inicio.Neuronal_bp_dw_scale<<endl;
                    inicio.Neuronal_bp_dw_scale=inicio.Neuronal_bp_dw_scale+salto.Neuronal_bp_dw_scale;
                }
                else if(parametro=="Neuronal_bp_moment_scale"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Neuronal_bp_moment_scale);
                        if(Y==0)
                            valor_y.push_back(inicio.Neuronal_bp_moment_scale);
                    }
                    texto<<"Neuronal_bp_moment_scale= "<<inicio.Neuronal_bp_moment_scale<<endl;
                    inicio.Neuronal_bp_moment_scale=inicio.Neuronal_bp_moment_scale+salto.Neuronal_bp_moment_scale;
                }
                else if(parametro=="Neuronal_fparam1"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Neuronal_fparam1);
                        if(Y==0)
                            valor_y.push_back(inicio.Neuronal_fparam1);
                    }
                    texto<<"Neuronal_fparam1= "<<inicio.Neuronal_fparam1<<endl;
                    inicio.Neuronal_fparam1=inicio.Neuronal_fparam1+salto.Neuronal_fparam1;
                }
                else if(parametro=="Neuronal_fparam2"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Neuronal_fparam2);
                        if(Y==0)
                            valor_y.push_back(inicio.Neuronal_fparam2);
                    }
                    texto<<"Neuronal_fparam2= "<<inicio.Neuronal_fparam2<<endl;
                    inicio.Neuronal_fparam2=inicio.Neuronal_fparam2+salto.Neuronal_fparam2;
                }
                else if(parametro=="Neuronal_rp_dw0"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Neuronal_rp_dw0);
                        if(Y==0)
                            valor_y.push_back(inicio.Neuronal_rp_dw0);
                    }
                    texto<<"Neuronal_rp_dw0= "<<inicio.Neuronal_rp_dw0<<endl;
                    inicio.Neuronal_rp_dw0=inicio.Neuronal_rp_dw0+salto.Neuronal_rp_dw0;
                }
                else if(parametro=="Neuronal_rp_dw_max"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Neuronal_rp_dw_max);
                        if(Y==0)
                            valor_y.push_back(inicio.Neuronal_rp_dw_max);
                    }
                    texto<<"Neuronal_rp_dw_max= "<<inicio.Neuronal_rp_dw_max<<endl;
                    inicio.Neuronal_rp_dw_max=inicio.Neuronal_rp_dw_max+salto.Neuronal_rp_dw_max;
                }
                else if(parametro=="Neuronal_rp_dw_min"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Neuronal_rp_dw_min);
                        if(Y==0)
                            valor_y.push_back(inicio.Neuronal_rp_dw_min);
                    }
                    texto<<"Neuronal_rp_dw_min= "<<inicio.Neuronal_rp_dw_min<<endl;
                    inicio.Neuronal_rp_dw_min=inicio.Neuronal_rp_dw_min+salto.Neuronal_rp_dw_min;
                }
                else if(parametro=="Neuronal_rp_dw_minus"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Neuronal_rp_dw_minus);
                        if(Y==0)
                            valor_y.push_back(inicio.Neuronal_rp_dw_minus);
                    }
                    texto<<"Neuronal_rp_dw_minus= "<<inicio.Neuronal_rp_dw_minus<<endl;
                    inicio.Neuronal_rp_dw_minus=inicio.Neuronal_rp_dw_minus+salto.Neuronal_rp_dw_minus;
                }
                else if(parametro=="Neuronal_rp_dw_plus"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Neuronal_rp_dw_plus);
                        if(Y==0)
                            valor_y.push_back(inicio.Neuronal_rp_dw_plus);
                    }
                    texto<<"Neuronal_rp_dw_plus= "<<inicio.Neuronal_rp_dw_plus<<endl;
                    inicio.Neuronal_rp_dw_plus=inicio.Neuronal_rp_dw_plus+salto.Neuronal_rp_dw_plus;
                }
                else if(parametro=="SVM_degree"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.SVM_degree);
                        if(Y==0)
                            valor_y.push_back(inicio.SVM_degree);
                    }
                    texto<<"SVM_degree= "<<inicio.SVM_degree<<endl;
                    inicio.SVM_degree=inicio.SVM_degree+salto.SVM_degree;
                }
                else if(parametro=="SVM_gamma"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.SVM_gamma);
                        if(Y==0)
                            valor_y.push_back(inicio.SVM_gamma);
                    }
                    texto<<"SVM_gamma= "<<inicio.SVM_gamma<<endl;
                    inicio.SVM_gamma=inicio.SVM_gamma+salto.SVM_gamma;
                }
                else if(parametro=="SVM_coef0"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.SVM_coef0);
                        if(Y==0)
                            valor_y.push_back(inicio.SVM_coef0);
                    }
                    texto<<"SVM_coef0= "<<inicio.SVM_coef0<<endl;
                    inicio.SVM_coef0=inicio.SVM_coef0+salto.SVM_coef0;
                }
                else if(parametro=="SVM_C"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.SVM_C);
                        if(Y==0)
                            valor_y.push_back(inicio.SVM_C);
                    }
                    texto<<"SVM_C= "<<inicio.SVM_C<<endl;
                    inicio.SVM_C=inicio.SVM_C+salto.SVM_C;
                }
                else if(parametro=="SVM_nu"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.SVM_nu);
                        if(Y==0)
                            valor_y.push_back(inicio.SVM_nu);
                    }
                    texto<<"SVM_nu= "<<inicio.SVM_nu<<endl;
                    inicio.SVM_nu=inicio.SVM_nu+salto.SVM_nu;
                }
                else if(parametro=="SVM_p"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.SVM_p);
                        if(Y==0)
                            valor_y.push_back(inicio.SVM_p);
                    }
                    texto<<"SVM_p= "<<inicio.SVM_p<<endl;
                    inicio.SVM_p=inicio.SVM_p+salto.SVM_p;
                }
                else if(parametro=="RTrees_cv_folds"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.RTrees_cv_folds);
                        if(Y==0)
                            valor_y.push_back(inicio.RTrees_cv_folds);
                    }
                    texto<<"RTrees_cv_folds= "<<inicio.RTrees_cv_folds<<endl;
                    inicio.RTrees_cv_folds=inicio.RTrees_cv_folds+salto.RTrees_cv_folds;
                }
                else if(parametro=="RTrees_max_categories"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.RTrees_max_categories);
                        if(Y==0)
                            valor_y.push_back(inicio.RTrees_max_categories);
                    }
                    texto<<"RTrees_max_categories= "<<inicio.RTrees_max_categories<<endl;
                    inicio.RTrees_max_categories=inicio.RTrees_max_categories+salto.RTrees_max_categories;
                }
                else if(parametro=="RTrees_max_depth"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.RTrees_max_depth);
                        if(Y==0)
                            valor_y.push_back(inicio.RTrees_max_depth);
                    }
                    texto<<"RTrees_max_depth= "<<inicio.RTrees_max_depth<<endl;
                    inicio.RTrees_max_depth=inicio.RTrees_max_depth+salto.RTrees_max_depth;
                }
                else if(parametro=="RTrees_min_sample_count"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.RTrees_min_sample_count);
                        if(Y==0)
                            valor_y.push_back(inicio.RTrees_min_sample_count);
                    }
                    texto<<"RTrees_min_sample_count= "<<inicio.RTrees_min_sample_count<<endl;
                    inicio.RTrees_min_sample_count=inicio.RTrees_min_sample_count+salto.RTrees_min_sample_count;
                }
                else if(parametro=="RTrees_native_vars"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.RTrees_native_vars);
                        if(Y==0)
                            valor_y.push_back(inicio.RTrees_native_vars);
                    }
                    texto<<"RTrees_native_vars= "<<inicio.RTrees_native_vars<<endl;
                    inicio.RTrees_native_vars=inicio.RTrees_native_vars+salto.RTrees_native_vars;
                }
                else if(parametro=="RTrees_regression_accuracy"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.RTrees_regression_accuracy);
                        if(Y==0)
                            valor_y.push_back(inicio.RTrees_regression_accuracy);
                    }
                    texto<<"RTrees_regression_accuracy= "<<inicio.RTrees_regression_accuracy<<endl;
                    inicio.RTrees_regression_accuracy=inicio.RTrees_regression_accuracy+salto.RTrees_regression_accuracy;
                }
                else if(parametro=="DTrees_cv_folds"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.DTrees_cv_folds);
                        if(Y==0)
                            valor_y.push_back(inicio.DTrees_cv_folds);
                    }
                    texto<<"DTrees_cv_folds= "<<inicio.DTrees_cv_folds<<endl;
                    inicio.DTrees_cv_folds=inicio.DTrees_cv_folds+salto.DTrees_cv_folds;
                }
                else if(parametro=="DTrees_max_categories"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.DTrees_max_categories);
                        if(Y==0)
                            valor_y.push_back(inicio.DTrees_max_categories);
                    }
                    texto<<"DTrees_max_categories= "<<inicio.DTrees_max_categories<<endl;
                    inicio.DTrees_max_categories=inicio.DTrees_max_categories+salto.DTrees_max_categories;
                }
                else if(parametro=="DTrees_max_depth"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.DTrees_max_depth);
                        if(Y==0)
                            valor_y.push_back(inicio.DTrees_max_depth);
                    }
                    texto<<"DTrees_max_depth= "<<inicio.DTrees_max_depth<<endl;
                    inicio.DTrees_max_depth=inicio.DTrees_max_depth+salto.DTrees_max_depth;
                }
                else if(parametro=="DTrees_min_sample_count"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.DTrees_min_sample_count);
                        if(Y==0)
                            valor_y.push_back(inicio.DTrees_min_sample_count);
                    }
                    texto<<"DTrees_min_sample_count= "<<inicio.DTrees_min_sample_count<<endl;
                    inicio.DTrees_min_sample_count=inicio.DTrees_min_sample_count+salto.DTrees_min_sample_count;
                }
                else if(parametro=="DTrees_regression_accuracy"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.DTrees_regression_accuracy);
                        if(Y==0)
                            valor_y.push_back(inicio.DTrees_regression_accuracy);
                    }
                    texto<<"DTrees_regression_accuracy= "<<inicio.DTrees_regression_accuracy<<endl;
                    inicio.DTrees_regression_accuracy=inicio.DTrees_regression_accuracy+salto.DTrees_regression_accuracy;
                }
                else if(parametro=="Boosting_max_depth"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Boosting_max_depth);
                        if(Y==0)
                            valor_y.push_back(inicio.Boosting_max_depth);
                    }
                    texto<<"Boosting_max_depth= "<<inicio.Boosting_max_depth<<endl;
                    inicio.Boosting_max_depth=inicio.Boosting_max_depth+salto.Boosting_max_depth;
                }
                else if(parametro=="Boosting_weak_count"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.Boosting_weak_count);
                        if(Y==0)
                            valor_y.push_back(inicio.Boosting_weak_count);
                    }
                    texto<<"Boosting_weak_count= "<<inicio.Boosting_weak_count<<endl;
                    inicio.Boosting_weak_count=inicio.Boosting_weak_count+salto.Boosting_weak_count;
                }
                else if(parametro=="EM_nclusters"){
                    if(j==0){
                        if(X==0)
                            valor_x.push_back(inicio.EM_nclusters);
                        if(Y==0)
                            valor_y.push_back(inicio.EM_nclusters);
                    }
                    texto<<"EM_nclusters= "<<inicio.EM_nclusters<<endl;
                    inicio.EM_nclusters=inicio.EM_nclusters+salto.EM_nclusters;
                }
//                else if(parametro=="GBT_weak_count"){
//                    if(j==0){
//                        if(X==0)
//                            valor_x.push_back(inicio.GBT_weak_count);
//                        if(Y==0)
//                            valor_y.push_back(inicio.GBT_weak_count);
//                    }
//                    texto<<"GBT_weak_count= "<<inicio.GBT_weak_count<<endl;
//                    inicio.GBT_weak_count=inicio.GBT_weak_count+salto.GBT_weak_count;
//                }
//                else if(parametro=="GBT_shrinkage"){
//                    if(j==0){
//                        if(X==0)
//                            valor_x.push_back(inicio.GBT_shrinkage);
//                        if(Y==0)
//                            valor_y.push_back(inicio.GBT_shrinkage);
//                    }
//                    texto<<"GBT_shrinkage= "<<inicio.GBT_shrinkage<<endl;
//                    inicio.GBT_shrinkage=inicio.GBT_shrinkage+salto.GBT_shrinkage;
//                }
//                else if(parametro=="GBT_max_depth"){
//                    if(j==0){
//                        if(X==0)
//                            valor_x.push_back(inicio.GBT_max_depth);
//                        if(Y==0)
//                            valor_y.push_back(inicio.GBT_max_depth);
//                    }
//                    texto<<"GBT_max_depth= "<<inicio.GBT_max_depth<<endl;
//                    inicio.GBT_max_depth=inicio.GBT_max_depth+salto.GBT_max_depth;
//                }
//                else if(parametro=="ERTrees_cv_folds"){
//                    if(j==0){
//                        if(X==0)
//                            valor_x.push_back(inicio.ERTrees_cv_folds);
//                        if(Y==0)
//                            valor_y.push_back(inicio.ERTrees_cv_folds);
//                    }
//                    texto<<"ERTrees_cv_folds= "<<inicio.ERTrees_cv_folds<<endl;
//                    inicio.ERTrees_cv_folds=inicio.ERTrees_cv_folds+salto.ERTrees_cv_folds;
//                }
//                else if(parametro=="ERTrees_max_categories"){
//                    if(j==0){
//                        if(X==0)
//                            valor_x.push_back(inicio.ERTrees_max_categories);
//                        if(Y==0)
//                            valor_y.push_back(inicio.ERTrees_max_categories);
//                    }
//                    texto<<"ERTrees_max_categories= "<<inicio.ERTrees_max_categories<<endl;
//                    inicio.ERTrees_max_categories=inicio.ERTrees_max_categories+salto.ERTrees_max_categories;
//                }
//                else if(parametro=="ERTrees_max_depth"){
//                    if(j==0){
//                        if(X==0)
//                            valor_x.push_back(inicio.ERTrees_max_depth);
//                        if(Y==0)
//                            valor_y.push_back(inicio.ERTrees_max_depth);
//                    }
//                    texto<<"ERTrees_max_depth= "<<inicio.ERTrees_max_depth<<endl;
//                    inicio.ERTrees_max_depth=inicio.ERTrees_max_depth+salto.ERTrees_max_depth;
//                }
//                else if(parametro=="ERTrees_min_sample_count"){
//                    if(j==0){
//                        if(X==0)
//                            valor_x.push_back(inicio.ERTrees_min_sample_count);
//                        if(Y==0)
//                            valor_y.push_back(inicio.ERTrees_min_sample_count);
//                    }
//                    texto<<"ERTrees_min_sample_count= "<<inicio.ERTrees_min_sample_count<<endl;
//                    inicio.ERTrees_min_sample_count=inicio.ERTrees_min_sample_count+salto.ERTrees_min_sample_count;
//                }
//                else if(parametro=="ERTrees_native_vars"){
//                    if(j==0){
//                        if(X==0)
//                            valor_x.push_back(inicio.ERTrees_native_vars);
//                        if(Y==0)
//                            valor_y.push_back(inicio.ERTrees_native_vars);
//                    }
//                    texto<<"ERTrees_native_vars= "<<inicio.ERTrees_native_vars<<endl;
//                    inicio.ERTrees_native_vars=inicio.ERTrees_native_vars+salto.ERTrees_native_vars;
//                }
//                else if(parametro=="ERTrees_regression_accuracy"){
//                    if(j==0){
//                        if(X==0)
//                            valor_x.push_back(inicio.ERTrees_regression_accuracy);
//                        if(Y==0)
//                            valor_y.push_back(inicio.ERTrees_regression_accuracy);
//                    }
//                    texto<<"ERTrees_regression_accuracy= "<<inicio.ERTrees_regression_accuracy<<endl;
//                    inicio.ERTrees_regression_accuracy=inicio.ERTrees_regression_accuracy+salto.ERTrees_regression_accuracy;
//                }
                suma_parametro[i].VP=suma_parametro[i].VP+RAT[i][j].VP;
                suma_parametro[i].VN=suma_parametro[i].VN+RAT[i][j].VN;
                suma_parametro[i].FN=suma_parametro[i].FN+RAT[i][j].FN;
                suma_parametro[i].FP=suma_parametro[i].FP+RAT[i][j].FP;
                texto<<"VP="<<RAT[i][j].VP;
                texto<<"    VN="<<RAT[i][j].VN;
                texto<<"    FP="<<RAT[i][j].FP;
                texto<<"    FN="<<RAT[i][j].FN;
                texto<<"    TAR="<<RAT[i][j].TAR;
                texto<<"    TRR="<<RAT[i][j].TRR;
                texto<<"    FAR="<<RAT[i][j].FAR;
                texto<<"    FRR="<<RAT[i][j].FRR;
                texto<<"    PPV="<<RAT[i][j].PPV;
                texto<<"    NPV="<<RAT[i][j].NPV;
                texto<<"    FDR="<<RAT[i][j].FDR;
                texto<<"    F1="<<RAT[i][j].F1;
                texto<<"    INFORMEDNESS="<<RAT[i][j].INFORMEDNESS;
                texto<<"    MARKEDNESS="<<RAT[i][j].MARKEDNESS;
                texto<<"    EXP_ERROR="<<RAT[i][j].EXP_ERROR;
                texto<<"    LR_NEG="<<RAT[i][j].LR_NEG;
                texto<<"    LR_POS="<<RAT[i][j].LR_POS;
                texto<<"    DOR="<<RAT[i][j].DOR;
                texto<<"    ACC="<<RAT[i][j].ACC;
                texto<<"    PREVALENCE="<<RAT[i][j].PREVALENCE<<endl;
            }
            texto<<endl;
            inicio=param_aux;
        }
        for(uint i=0; i<suma_parametro.size(); i++){
            if(suma_parametro[i].FP==0)
                suma_parametro[i].FAR=0;
            else
                suma_parametro[i].FAR=suma_parametro[i].FP/(suma_parametro[i].FP+suma_parametro[i].VN);
            if(suma_parametro[i].FN==0)
                suma_parametro[i].FRR=0;
            else
                suma_parametro[i].FRR=suma_parametro[i].FN/(suma_parametro[i].FN+suma_parametro[i].VP);
            suma_parametro[i].TAR=1-suma_parametro[i].FRR;
            suma_parametro[i].TRR=1-suma_parametro[i].FAR;
            suma_parametro[i].PPV=suma_parametro[i].VP/(suma_parametro[i].VP+suma_parametro[i].FP);
            suma_parametro[i].NPV=suma_parametro[i].VN/(suma_parametro[i].VN+suma_parametro[i].FN);
            suma_parametro[i].FDR=1-suma_parametro[i].PPV;
            suma_parametro[i].F1=(2*suma_parametro[i].VP)/(2*suma_parametro[i].VP+suma_parametro[i].FP+suma_parametro[i].FN);
            suma_parametro[i].INFORMEDNESS=suma_parametro[i].TAR+suma_parametro[i].TAR-1;
            suma_parametro[i].MARKEDNESS=suma_parametro[i].PPV+suma_parametro[i].NPV-1;
            suma_parametro[i].EXP_ERROR=0.5*suma_parametro[i].FAR+0.5*suma_parametro[i].FRR;
            if(suma_parametro[i].FAR!=0)
                suma_parametro[i].LR_POS=suma_parametro[i].TAR/suma_parametro[i].FAR;
            if(suma_parametro[i].TRR!=0)
                suma_parametro[i].LR_NEG=suma_parametro[i].FRR/suma_parametro[i].TRR;
            if(suma_parametro[i].LR_NEG!=0)
                suma_parametro[i].DOR=suma_parametro[i].LR_POS/suma_parametro[i].LR_NEG;
            suma_parametro[i].ACC=(suma_parametro[i].VP+suma_parametro[i].VN)/(suma_parametro[i].VP+suma_parametro[i].VN+suma_parametro[i].FN+suma_parametro[i].FP);
            suma_parametro[i].PREVALENCE=(suma_parametro[i].VP+suma_parametro[i].FN)/(suma_parametro[i].VP+suma_parametro[i].VN+suma_parametro[i].FN+suma_parametro[i].FP);
            suma_parametro[i].EXP_ERROR=0.5*suma_parametro[i].FAR+0.5*suma_parametro[i].FRR;
            if(Y==1)
                valor_y.push_back(suma_parametro[i].VP);
            else if(Y==2)
                valor_y.push_back(suma_parametro[i].VN);
            else if(Y==3)
                valor_y.push_back(suma_parametro[i].FN);
            else if(Y==4)
                valor_y.push_back(suma_parametro[i].FP);
            else if(Y==5)
                valor_y.push_back(suma_parametro[i].TAR);
            else if(Y==6)
                valor_y.push_back(suma_parametro[i].TRR);
            else if(Y==7)
                valor_y.push_back(suma_parametro[i].FAR);
            else if(Y==8)
                valor_y.push_back(suma_parametro[i].FRR);
            else if(Y==9)
                valor_y.push_back(suma_parametro[i].PPV);
            else if(Y==10)
                valor_y.push_back(suma_parametro[i].NPV);
            else if(Y==11)
                valor_y.push_back(suma_parametro[i].FDR);
            else if(Y==12)
                valor_y.push_back(suma_parametro[i].F1);
            else if(Y==13)
                valor_y.push_back(suma_parametro[i].INFORMEDNESS);
            else if(Y==14)
                valor_y.push_back(suma_parametro[i].MARKEDNESS);
            else if(Y==15)
                valor_y.push_back(suma_parametro[i].EXP_ERROR);
            else if(Y==16)
                valor_y.push_back(suma_parametro[i].LR_POS);
            else if(Y==17)
                valor_y.push_back(suma_parametro[i].LR_NEG);
            else if(Y==18)
                valor_y.push_back(suma_parametro[i].DOR);
            else if(Y==19)
                valor_y.push_back(suma_parametro[i].ACC);
            else if(Y==20)
                valor_y.push_back(suma_parametro[i].PREVALENCE);
            if(X==1)
                valor_x.push_back(suma_parametro[i].VP);
            else if(Y==2)
                valor_x.push_back(suma_parametro[i].VN);
            else if(Y==3)
                valor_x.push_back(suma_parametro[i].FN);
            else if(Y==4)
                valor_x.push_back(suma_parametro[i].FP);
            else if(Y==5)
                valor_x.push_back(suma_parametro[i].TAR);
            else if(Y==6)
                valor_x.push_back(suma_parametro[i].TRR);
            else if(Y==7)
                valor_x.push_back(suma_parametro[i].FAR);
            else if(Y==8)
                valor_x.push_back(suma_parametro[i].FRR);
            else if(Y==9)
                valor_x.push_back(suma_parametro[i].PPV);
            else if(Y==10)
                valor_x.push_back(suma_parametro[i].NPV);
            else if(Y==11)
                valor_x.push_back(suma_parametro[i].FDR);
            else if(Y==12)
                valor_x.push_back(suma_parametro[i].F1);
            else if(Y==13)
                valor_x.push_back(suma_parametro[i].INFORMEDNESS);
            else if(Y==14)
                valor_x.push_back(suma_parametro[i].MARKEDNESS);
            else if(Y==15)
                valor_x.push_back(suma_parametro[i].EXP_ERROR);
            else if(Y==16)
                valor_x.push_back(suma_parametro[i].LR_POS);
            else if(Y==17)
                valor_x.push_back(suma_parametro[i].LR_NEG);
            else if(Y==18)
                valor_x.push_back(suma_parametro[i].DOR);
            else if(Y==19)
                valor_x.push_back(suma_parametro[i].ACC);
            else if(Y==20)
                valor_x.push_back(suma_parametro[i].PREVALENCE);
        }
        ui->Texto_Result->setText(QString::fromStdString(texto.str()));
        vector<cv::Scalar> Col;
        Scalar color;
        color[0]=0;
        color[1]=0;
        color[2]=255;
        Col.push_back(color);
        Representacion rep;
        stringstream text;
        if(X==0)
            text<<parametro+" Parametro";
        else if(X==1)
            text<<parametro+" VP";
        else if(X==2)
            text<<parametro+" VN";
        else if(X==3)
            text<<parametro+" FP";
        else if(X==4)
            text<<parametro+" FN";
        else if(X==5)
            text<<parametro+" TAR";
        else if(X==6)
            text<<parametro+" TRR";
        else if(X==7)
            text<<parametro+" FAR";
        else if(X==8)
            text<<parametro+" FRR";
        else if(X==9)
            text<<parametro+" PPV";
        else if(X==10)
            text<<parametro+" NPV";
        else if(X==11)
            text<<parametro+" FDR";
        else if(X==12)
            text<<parametro+" F1";
        else if(X==13)
            text<<parametro+" INFORMEDNESS";
        else if(X==14)
            text<<parametro+" MARKEDNESS";
        else if(X==15)
            text<<parametro+" ERROR_EXP";
        else if(X==16)
            text<<parametro+" LR_NEG";
        else if(X==17)
            text<<parametro+" LR_POS";
        else if(X==18)
            text<<parametro+" DOR";
        else if(X==19)
            text<<parametro+" ACC";
        else if(X==20)
            text<<parametro+" PREVALENCE";
        if(Y==0)
            text<<"/PREVALENCE";
        else if(Y==1)
            text<<"/VP";
        else if(Y==2)
            text<<"/VN";
        else if(Y==3)
            text<<"/FP";
        else if(Y==4)
            text<<"/FN";
        else if(Y==5)
            text<<"/TAR";
        else if(Y==6)
            text<<"/TRR";
        else if(Y==7)
            text<<"/FAR";
        else if(Y==8)
            text<<"/FRR";
        else if(Y==9)
            text<<"/PPV";
        else if(Y==10)
            text<<"/NPV";
        else if(Y==11)
            text<<"/FDR";
        else if(Y==12)
            text<<"/F1";
        else if(Y==13)
            text<<"/INFORMEDNESS";
        else if(Y==14)
            text<<"/MARKEDNESS";
        else if(Y==15)
            text<<"/ERROR_EXP";
        else if(Y==16)
            text<<"/LR_NEG";
        else if(Y==17)
            text<<"/LR_POS";
        else if(Y==18)
            text<<"/DOR";
        else if(Y==19)
            text<<"/ACC";
        else if(Y==20)
            text<<"/PREVALENCE";
        vector<float> lab;
        Mat Dat=Mat::zeros(valor_x.size(),2,CV_32F);
        for(uint i=0; i<valor_x.size(); i++){
            Dat.at<float>(i,0)=valor_x[i];
            Dat.at<float>(i,1)=valor_y[i];
            lab.push_back(1);
        }
        rep.Continuous_data_represent(text.str(),Dat,lab,Col);
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se ha podido ejecutar");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        ui->progress_Clasificar->setValue(0);
        return;
    }
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se ha podido ejecutar la herramienta");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        ui->progress_Clasificar->setValue(0);
        return;
    }
    ui->progress_Clasificar->setValue(0);
    QApplication::restoreOverrideCursor();
}


void MainWindow::on_Parametros_clicked()
{
    Conf_Parametros *conf=new Conf_Parametros(&inicio,this);
    conf->show();
}

void MainWindow::on_Salto_clicked()
{
    Conf_Parametros *conf=new Conf_Parametros(&salto,this);
    conf->show();
}

void MainWindow::on_Parada_clicked()
{
    Conf_Parametros *conf=new Conf_Parametros(&fin,this);
    conf->show();
}

void MainWindow::on_toolButton_7_clicked()
{
    Con_Multi *mul=new Con_Multi(this,this);
    mul->show();
}

void MainWindow::on_toolButton_20_clicked()
{
    Conf_SC_Vali *SC=new Conf_SC_Vali(this,this);
    SC->show();
}

void MainWindow::on_toolButton_9_clicked()
{
    Conf_Multi *mul=new Conf_Multi(this,this);
    mul->show();
}


void MainWindow::on_Validation_clicked()
{
    ui->Parametros->setEnabled(true);
    ui->Salto->setEnabled(false);
    ui->Parada->setEnabled(false);
    ui->label_18->setEnabled(true);
    ui->label_19->setEnabled(false);
    ui->label_28->setEnabled(false);
    ui->Porcentaje->setEnabled(true);
    ui->Num_folds->setEnabled(false);
    ui->Tam_Folds->setEnabled(false);
    ui->Tipo_Clasif->setEnabled(true);
}

void MainWindow::on_Validation2_clicked()
{
    ui->Parametros->setEnabled(true);
    ui->Salto->setEnabled(false);
    ui->Parada->setEnabled(false);
    ui->label_18->setEnabled(true);
    ui->label_19->setEnabled(false);
    ui->label_28->setEnabled(false);
    ui->Porcentaje->setEnabled(true);
    ui->Num_folds->setEnabled(false);
    ui->Tam_Folds->setEnabled(false);
    ui->Tipo_Clasif->setEnabled(false);
}

void MainWindow::on_C_Validation_clicked()
{
    ui->Parametros->setEnabled(true);
    ui->Salto->setEnabled(true);
    ui->Parada->setEnabled(true);
    ui->label_18->setEnabled(false);
    ui->label_19->setEnabled(true);
    ui->label_28->setEnabled(true);
    ui->Porcentaje->setEnabled(false);
    ui->Num_folds->setEnabled(true);
    ui->Tam_Folds->setEnabled(true);
    ui->Tipo_Clasif->setEnabled(true);
}

void MainWindow::on_SC_Validation_clicked()
{
    ui->Parametros->setEnabled(true);
    ui->Salto->setEnabled(true);
    ui->Parada->setEnabled(true);
    ui->label_18->setEnabled(false);
    ui->label_19->setEnabled(true);
    ui->label_28->setEnabled(true);
    ui->Porcentaje->setEnabled(false);
    ui->Num_folds->setEnabled(true);
    ui->Tam_Folds->setEnabled(true);
    ui->Tipo_Clasif->setEnabled(false);
}

void MainWindow::on_Ratios_Parametro_clicked()
{
    ui->Parametros->setEnabled(true);
    ui->Salto->setEnabled(true);
    ui->Parada->setEnabled(true);
    ui->label_18->setEnabled(true);
    ui->label_19->setEnabled(false);
    ui->label_28->setEnabled(false);
    ui->Porcentaje->setEnabled(true);
    ui->Num_folds->setEnabled(false);
    ui->Tam_Folds->setEnabled(false);
    ui->Tipo_Clasif->setEnabled(false);
}

void MainWindow::on_toolButton_22_clicked()
{
    Selec_Param *par=new Selec_Param(this,this);
    par->show();
}


void MainWindow::on_Representar_3_clicked()
{
    if(IMAGENES.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay datos cargados");
        msgBox.exec();
        return;
    }
    if(LABELS.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay Etiquetas cargadas");
        msgBox.exec();
        return;
    }
    if(resultado.empty()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No hay datos clasificados");
        msgBox.exec();
        return;
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    X=ui->X->currentIndex();
    Y=ui->Y->currentIndex();
    int e=0;
    Analisis an;
    vector<vector<Analisis::Ratios_data> > Ratios;
    e=an.Ratios_Histograma(IMAGENES,LABELS,resultado,num_bar,Ratios);
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido obtener los datos estadísticos");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    Mat datos_dibujo=Mat::zeros(Ratios[ui->Dimension_graf->value()].size(),2,CV_32F);
    vector<float> Etiq;
    for(uint i=0; i<Ratios[ui->Dimension_graf->value()].size(); i++){
        Etiq.push_back(1.0);
        if(X==0)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].valor_x;
        if(X==1)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].VP;
        if(X==2)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].VN;
        if(X==3)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].FP;
        if(X==4)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].FN;
        if(X==5)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].TAR;
        if(X==6)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].TRR;
        if(X==7)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].FAR;
        if(X==8)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].FRR;
        if(X==9)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].PPV;
        if(X==10)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].NPV;
        if(X==11)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].FDR;
        if(X==12)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].F1;
        if(X==13)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].INFORMEDNESS;
        if(X==14)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].MARKEDNESS;
        if(X==15)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].EXP_ERROR;    
        if(X==16)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].LR_NEG;
        if(X==17)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].LR_POS;
        if(X==18)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].DOR;
        if(X==19)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].ACC;
        if(X==20)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].PREVALENCE;
        if(Y==0)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].valor_x;
        if(Y==1)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].VP;
        if(Y==2)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].VN;
        if(Y==3)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].FP;
        if(Y==4)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].FN;
        if(Y==5)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].TAR;
        if(Y==6)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].TRR;
        if(Y==7)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].FAR;
        if(Y==8)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].FRR;
        if(Y==9)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].PPV;
        if(Y==10)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].NPV;
        if(Y==11)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].FDR;
        if(Y==12)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].F1;
        if(Y==13)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].INFORMEDNESS;
        if(Y==14)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].MARKEDNESS;
        if(Y==15)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].EXP_ERROR;
        if(Y==16)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].LR_NEG;
        if(Y==17)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].LR_POS;
        if(Y==18)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].DOR;
        if(Y==19)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].ACC;
        if(Y==20)
            datos_dibujo.at<float>(i,0)=Ratios[ui->Dimension_graf->value()][i].PREVALENCE;
    }
    stringstream text;
    if(X==0)
        text<<result_ref+" Parametro";
    else if(X==1)
        text<<result_ref+" VP";
    else if(X==2)
        text<<result_ref+" VN";
    else if(X==3)
        text<<result_ref+" FP";
    else if(X==4)
        text<<result_ref+" FN";
    else if(X==5)
        text<<result_ref+" TAR";
    else if(X==6)
        text<<result_ref+" TRR";
    else if(X==7)
        text<<result_ref+" FAR";
    else if(X==8)
        text<<result_ref+" FRR";
    else if(X==9)
        text<<result_ref+" PPV";
    else if(X==10)
        text<<result_ref+" NPV";
    else if(X==11)
        text<<result_ref+" FDR";
    else if(X==12)
        text<<result_ref+" F1";
    else if(X==13)
        text<<result_ref+" INFORMEDNESS";
    else if(X==14)
        text<<result_ref+" MARKEDNESS";
    else if(X==15)
        text<<result_ref+" ERROR_EXP";
    else if(X==16)
        text<<result_ref+" LR_NEG";
    else if(X==17)
        text<<result_ref+" LR_POS";
    else if(X==18)
        text<<result_ref+" DOR";
    else if(X==19)
        text<<result_ref+" ACC";
    else if(X==20)
        text<<result_ref+" PREVALENCE";
    if(Y==0)
        text<<"/PREVALENCE Dimension "<<ui->Dimension_graf->value();
    else if(Y==1)
        text<<"/VP Dimension "<<ui->Dimension_graf->value();
    else if(Y==2)
        text<<"/VN Dimension "<<ui->Dimension_graf->value();
    else if(Y==3)
        text<<"/FP Dimension "<<ui->Dimension_graf->value();
    else if(Y==4)
        text<<"/FN Dimension "<<ui->Dimension_graf->value();
    else if(Y==5)
        text<<"/TAR Dimension "<<ui->Dimension_graf->value();
    else if(Y==6)
        text<<"/TRR Dimension "<<ui->Dimension_graf->value();
    else if(Y==7)
        text<<"/FAR Dimension "<<ui->Dimension_graf->value();
    else if(Y==8)
        text<<"/FRR Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==9)
        text<<"/PPV Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==10)
        text<<"/NPV Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==11)
        text<<"/FDR Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==12)
        text<<"/F1 Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==13)
        text<<"/INFORMEDNESS Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==14)
        text<<"/MARKEDNESS Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==15)
        text<<"/ERROR_EXP Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==16)
        text<<"/LR_NEG Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==17)
        text<<"/LR_POS Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==18)
        text<<"/DOR Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==19)
        text<<"/ACC Etiqueta "<<ui->Dimension_graf->value();
    else if(Y==20)
        text<<"/PREVALENCE Etiqueta "<<ui->Dimension_graf->value();
    vector<cv::Scalar> Colores;
    Colores.push_back(Scalar(0,0,255));
    Representacion rep;
    e=rep.Continuous_data_represent(text.str(),datos_dibujo,Etiq,Colores);
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido representar los datos estadísticos");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_toolButton_busqueda_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                             this,
                             tr("- SELECCIONAR IMAGEN -"),
                             QDir::currentPath()+"..",
                             tr("Document files (*.bmp *.dib *.jpeg *.jpg *.jpe *.jp2 *.png *.pbm *.pgm *.ppm *.sr *.ras *.tiff *.tif);;All files (*.*)") );
    ui->Direccion_Carga_4->setText(filename);
}

void MainWindow::on_toolButton_19_clicked()
{
    Con_Multi *mul=new Con_Multi(this,this);
    mul->show();
}

void MainWindow::on_Config_Des_2_clicked()
{
    if(ui->Tipo_Descrip_2->currentIndex()==0){
        QMessageBox msgBox;
        msgBox.setText("ERROR: Seleccione un tipo de descriptor");
        msgBox.exec();
        return;
    }
    if(ui->Tipo_Descrip_2->currentIndex()==10){
        Conf_HOG *conf_HOG= new Conf_HOG(this,this);
        conf_HOG->show();
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==11){
        Config_PC *conf_PC= new Config_PC(this,this);
        conf_PC->show();
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: El descriptor no tiene configuracion");
        msgBox.exec();
        return;
    }
}

void MainWindow::on_Iniciar_6_clicked()
{
    int e=0;
    cv::Mat imagen;
    cv::Mat salida;
    vector<cv::RotatedRect> recuadros;
    vector<float> labels_recuadros;
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int current_type;
    if(ui->Aleatoria->isChecked()){
        Generacion gen;
        gen.Random_Synthetic_Image(ui->Clases_aleatoria->value(),Size(500,500),ui->Varianza_2->value(),ui->Separacion_2->value(),imagen);
        current_type=GRAY;
    }
    else{
        imagen=cv::imread(ui->Direccion_Carga_4->text().toStdString());
        if(imagen.empty()){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se ha podido abrir la imagen o la ruta es erronea");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
        imagen.convertTo(imagen,CV_32F);
        if(imagen.cols<ui->Vent_X->value() || imagen.rows<ui->Vent_Y->value()){
            QMessageBox msgBox;
            msgBox.setText("ERROR: El tamaño de la ventana es mayor que el de la imagen");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
        current_type=RGB;
    }
    int tipo_dato=-1;
    Descriptor *Tipo_Descriptor;
    if(ui->Tipo_Descrip_2->currentIndex()==1){
        tipo_dato=RGB;
        Tipo_Descriptor=0;
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==2){
        tipo_dato=GRAY;
        Basic_Transformations *basic=new Basic_Transformations(current_type,GRAY);
        Tipo_Descriptor=basic;
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==3){
        tipo_dato=HSV;
        Basic_Transformations *basic=new Basic_Transformations(current_type,HSV);
        Tipo_Descriptor=basic;
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==4){
        tipo_dato=H_CHANNEL;
        Basic_Transformations *basic=new Basic_Transformations(current_type,H_CHANNEL);
        Tipo_Descriptor=basic;
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==5){
        tipo_dato=S_CHANNEL;
        Basic_Transformations *basic=new Basic_Transformations(current_type,S_CHANNEL);
        Tipo_Descriptor=basic;
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==6){
        tipo_dato=V_CHANNEL;
        Basic_Transformations *basic=new Basic_Transformations(current_type,V_CHANNEL);
        Tipo_Descriptor=basic;
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==7){
        tipo_dato=THRESHOLD;
        Basic_Transformations *basic=new Basic_Transformations(current_type,THRESHOLD);
        Tipo_Descriptor=basic;
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==8){
        tipo_dato=CANNY;
        Basic_Transformations *basic=new Basic_Transformations(current_type,CANNY);
        Tipo_Descriptor=basic;
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==9){
        tipo_dato=SOBEL;
        Basic_Transformations *basic=new Basic_Transformations(current_type,SOBEL);
        Tipo_Descriptor=basic;
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==10){
        if(Win_Size.height>ui->Vent_Y->value()|| Win_Size.width>ui->Vent_X->value()){
            QMessageBox msgBox;
            msgBox.setText("ERROR: El tamaño de la ventana de busqueda es menor que el de la ventana de HOG");
            msgBox.exec();
            ui->progress_Clasificar->setValue(100);
            ui->progress_Clasificar->setValue(0);
            ui->progress_generar->setValue(100);
            ui->progress_generar->setValue(0);
            ui->progress_Cargar->setValue(100);
            ui->progress_Cargar->setValue(0);
            ui->progress_Clus->setValue(100);
            ui->progress_Clus->setValue(0);
            ui->progress_Dimensionalidad->setValue(100);
            ui->progress_Dimensionalidad->setValue(0);
            QApplication::restoreOverrideCursor();
            return;
        }
        tipo_dato=HOG_DES;
        HOG *Hoog=new HOG(Win_Size,Block_Stride, Win_Sigma,Threshold_L2hys, Gamma_Correction, Nlevels);
        Tipo_Descriptor= Hoog;
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==11){
        tipo_dato=PUNTOS_CARACTERISTICOS;
        Puntos_Caracteristicos *des=new Puntos_Caracteristicos(Tipo_Des,Tipo_Ext,Parametro);
        Tipo_Descriptor= des;
    }
    else if(ui->Tipo_Descrip_2->currentIndex()==12){
        tipo_dato=COLOR_PREDOMINANTE;
        Basic_Transformations *basic=new Basic_Transformations(current_type,COLOR_PREDOMINANTE);
        Tipo_Descriptor=basic;
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: Seleccione un tipo de descriptor");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    if(ui->radioPosicion->isChecked()){
        if(ui->Clasif_Cargado_2->isChecked()){
            if(ID==DISTANCIAS){
                D.progreso=0;
                D.max_progreso=100;
                D.base_progreso=0;
                D.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                D.window=ui;
                Busqueda bus(&D,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
            else if(ID==GAUSSIANO){
                G.progreso=0;
                G.max_progreso=100;
                G.base_progreso=0;
                G.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                G.window=ui;
                Busqueda bus(&G,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
            else if(ID==CASCADA_CLAS){
                HA.progreso=0;
                HA.max_progreso=100;
                HA.base_progreso=0;
                HA.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                HA.window=ui;
                Busqueda bus(&HA,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
            else if(ID==HISTOGRAMA){
                H.progreso=0;
                H.max_progreso=100;
                H.base_progreso=0;
                H.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                H.window=ui;
                Busqueda bus(&H,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
            else if(ID==KNN){
                K.progreso=0;
                K.max_progreso=100;
                K.base_progreso=0;
                K.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                K.window=ui;
                Busqueda bus(&K,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
            else if(ID==NEURONAL){
                N.progreso=0;
                N.max_progreso=100;
                N.base_progreso=0;
                N.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                N.window=ui;
                Busqueda bus(&N,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
            else if(ID==C_SVM){
                S.progreso=0;
                S.max_progreso=100;
                S.base_progreso=0;
                S.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                S.window=ui;
                Busqueda bus(&S,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
            else if(ID==RTREES){
                RT.progreso=0;
                RT.max_progreso=100;
                RT.base_progreso=0;
                RT.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                RT.window=ui;
                Busqueda bus(&RT,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
            else if(ID==DTREES){
                DT.progreso=0;
                DT.max_progreso=100;
                DT.base_progreso=0;
                DT.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                DT.window=ui;
                Busqueda bus(&DT,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
            else if(ID==BOOSTING){
                B.progreso=0;
                B.max_progreso=100;
                B.base_progreso=0;
                B.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                B.window=ui;
                Busqueda bus(&B,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
            else if(ID==EXP_MAX){
                E.progreso=0;
                E.max_progreso=100;
                E.base_progreso=0;
                E.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                E.window=ui;
                Busqueda bus(&E,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
//            else if(ID==GBT){
//                GB.progreso=0;
//                GB.max_progreso=100;
//                GB.base_progreso=0;
//                GB.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                GB.window=ui;
//                Busqueda bus(GBT,&GB,tipo_dato,Tipo_Descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
//            else if(ID==ERTREES){
//                ER.progreso=0;
//                ER.max_progreso=100;
//                ER.base_progreso=0;
//                ER.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                ER.window=ui;
//                Busqueda bus(ERTREES,&ER,tipo_dato,Tipo_Descriptor);
//                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
//            }
            else if(ID==MICLASIFICADOR){
                MC.progreso=0;
                MC.max_progreso=100;
                MC.base_progreso=0;
                MC.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                MC.window=ui;
                Busqueda bus(&MC,tipo_dato,Tipo_Descriptor);
                e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
            }
            else{
                QMessageBox msgBox;
                msgBox.setText("ERROR: No se ha cargado ningun clasificador");
                msgBox.exec();
                QApplication::restoreOverrideCursor();
                return;
            }
            if(e==1){
                QMessageBox msgBox;
                msgBox.setText("ERROR: No se ha podido clasificar la imagen");
                msgBox.exec();
                QApplication::restoreOverrideCursor();
                return;
            }
        }
        else if(ui->Multiclasif_2->isChecked()){
            vector<Clasificador*> clasificadores;
            for(uint i=0; i<id_clasificadores.size(); i++){
                if(id_clasificadores[i]==DISTANCIAS){
                    Clasificador_Distancias *clasi=new Clasificador_Distancias(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==GAUSSIANO){
                    Clasificador_Gaussiano *clasi=new Clasificador_Gaussiano(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==CASCADA_CLAS){
                    Clasificador_Cascada *clasi=new Clasificador_Cascada(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==HISTOGRAMA){
                    Clasificador_Histograma *clasi=new Clasificador_Histograma(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==KNN){
                    Clasificador_KNN *clasi=new Clasificador_KNN(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==NEURONAL){
                    Clasificador_Neuronal *clasi=new Clasificador_Neuronal(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==C_SVM){
                    Clasificador_SVM *clasi=new Clasificador_SVM(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==RTREES){
                    Clasificador_RTrees *clasi=new Clasificador_RTrees(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==DTREES){
                    Clasificador_DTrees *clasi=new Clasificador_DTrees(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==BOOSTING){
                    Clasificador_Boosting *clasi=new Clasificador_Boosting(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
        //        else if(id_clasificadores[i]==GBT){
        //            Clasificador_GBTrees *clasi=new Clasificador_GBTrees(nombres[i]);
    //                clasi->Read_Data();
    //                clasificadores.push_back(clasi);
        //        }
        //        else if(id_clasificadores[i]==ERTREES){
        //            Clasificador_ERTrees *clasi=new Clasificador_ERTrees(nombres[i]);
    //                clasi->Read_Data();
    //                clasificadores.push_back(clasi);
        //        }
                else if(id_clasificadores[i]==EXP_MAX){
                    Clasificador_EM *clasi=new Clasificador_EM(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
            }
            MultiClasificador multi(clasificadores);
            multi.progreso=0;
            multi.max_progreso=100;
            multi.base_progreso=0;
            multi.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
            multi.window=ui;
            Busqueda bus(&multi,tipo_dato,Tipo_Descriptor,&Multi_tipo);
            e=bus.Posicion(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),ui->Solapamiento->isChecked(),ui->Filtro_aislados->isChecked(),ui->Dist_cuadros->value(),ui->Rotacion_2->value(),recuadros,labels_recuadros);
        }
        Representacion rep;
        Mat mostrar;
        imagen.convertTo(imagen,CV_32F);
        double minval,maxval;
        cv::minMaxLoc(imagen,&minval,&maxval);
        imagen=(imagen-minval)/(maxval-minval);
        imshow("Imagen",imagen);
        rep.Recuadros(imagen,recuadros,labels_recuadros,Col,mostrar,show_graphics);
    }
    else if(ui->radioTextura->isChecked()){
        if(ui->Clasif_Cargado_2->isChecked()){
            if(ID==DISTANCIAS){
                D.progreso=0;
                D.max_progreso=100;
                D.base_progreso=0;
                D.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                D.window=ui;
                Busqueda bus(&D,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
            else if(ID==GAUSSIANO){
                G.progreso=0;
                G.max_progreso=100;
                G.base_progreso=0;
                G.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                G.window=ui;
                Busqueda bus(&G,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
            else if(ID==CASCADA_CLAS){
                HA.progreso=0;
                HA.max_progreso=100;
                HA.base_progreso=0;
                HA.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                HA.window=ui;
                Busqueda bus(&HA,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
            else if(ID==HISTOGRAMA){
                H.progreso=0;
                H.max_progreso=100;
                H.base_progreso=0;
                H.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                H.window=ui;
                Busqueda bus(&H,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
            else if(ID==KNN){
                K.progreso=0;
                K.max_progreso=100;
                K.base_progreso=0;
                K.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                K.window=ui;
                Busqueda bus(&K,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
            else if(ID==NEURONAL){
                N.progreso=0;
                N.max_progreso=100;
                N.base_progreso=0;
                N.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                N.window=ui;
                Busqueda bus(&N,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
            else if(ID==C_SVM){
                S.progreso=0;
                S.max_progreso=100;
                S.base_progreso=0;
                S.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                S.window=ui;
                Busqueda bus(&S,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
            else if(ID==RTREES){
                RT.progreso=0;
                RT.max_progreso=100;
                RT.base_progreso=0;
                RT.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                RT.window=ui;
                Busqueda bus(&RT,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
            else if(ID==DTREES){
                DT.progreso=0;
                DT.max_progreso=100;
                DT.base_progreso=0;
                DT.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                DT.window=ui;
                Busqueda bus(&DT,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
            else if(ID==BOOSTING){
                B.progreso=0;
                B.max_progreso=100;
                B.base_progreso=0;
                B.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                B.window=ui;
                Busqueda bus(&B,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
            else if(ID==EXP_MAX){
                E.progreso=0;
                E.max_progreso=100;
                E.base_progreso=0;
                E.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                E.window=ui;
                Busqueda bus(&E,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
//            else if(ID==GBT){
//                GB.progreso=0;
//                GB.max_progreso=100;
//                GB.base_progreso=0;
//                GB.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                GB.window=ui;
//                Busqueda bus(GBT,&GB,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
//            else if(ID==ERTREES){
//                ER.progreso=0;
//                ER.max_progreso=100;
//                ER.base_progreso=0;
//                ER.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
//                ER.window=ui;
//                Busqueda bus(ERTREES,&ER,tipo_dato,Tipo_Descriptor);
//                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
//            }
            else if(ID==MICLASIFICADOR){
                MC.progreso=0;
                MC.max_progreso=100;
                MC.base_progreso=0;
                MC.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
                MC.window=ui;
                Busqueda bus(&MC,tipo_dato,Tipo_Descriptor);
                e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
            }
            else{
                QMessageBox msgBox;
                msgBox.setText("ERROR: No se ha cargado ningun clasificador");
                msgBox.exec();
                QApplication::restoreOverrideCursor();
                return;
            }
            if(e==1){
                QMessageBox msgBox;
                msgBox.setText("ERROR: No se ha podido clasificar la imagen");
                msgBox.exec();
                QApplication::restoreOverrideCursor();
                return;
            }
        }
        else if(ui->Multiclasif_2->isChecked()){
            vector<Clasificador*> clasificadores;
            for(uint i=0; i<id_clasificadores.size(); i++){
                if(id_clasificadores[i]==DISTANCIAS){
                    Clasificador_Distancias *clasi=new Clasificador_Distancias(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==GAUSSIANO){
                    Clasificador_Gaussiano *clasi=new Clasificador_Gaussiano(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==CASCADA_CLAS){
                    Clasificador_Cascada *clasi=new Clasificador_Cascada(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==HISTOGRAMA){
                    Clasificador_Histograma *clasi=new Clasificador_Histograma(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==KNN){
                    Clasificador_KNN *clasi=new Clasificador_KNN(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==NEURONAL){
                    Clasificador_Neuronal *clasi=new Clasificador_Neuronal(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==C_SVM){
                    Clasificador_SVM *clasi=new Clasificador_SVM(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==RTREES){
                    Clasificador_RTrees *clasi=new Clasificador_RTrees(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==DTREES){
                    Clasificador_DTrees *clasi=new Clasificador_DTrees(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==BOOSTING){
                    Clasificador_Boosting *clasi=new Clasificador_Boosting(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
                else if(id_clasificadores[i]==EXP_MAX){
                    Clasificador_EM *clasi=new Clasificador_EM(nombres[i]);
                    clasi->Read_Data();
                    clasificadores.push_back(clasi);
                }
        //        else if(id_clasificadores[i]==GBT){
        //            Clasificador_GBTrees *clasi=new Clasificador_GBTrees(nombres[i]);
    //                clasi->Read_Data();
    //                clasificadores.push_back(clasi);
        //        }
        //        else if(id_clasificadores[i]==ERTREES){
        //            Clasificador_ERTrees *clasi=new Clasificador_ERTrees(nombres[i]);
    //                clasi->Read_Data();
    //                clasificadores.push_back(clasi);
        //        }
            }
            MultiClasificador multi(clasificadores);
            multi.progreso=0;
            multi.max_progreso=100;
            multi.base_progreso=0;
            multi.total_progreso=((imagen.cols-ui->Vent_X->value())/ui->Salto_2->value())*((imagen.rows-ui->Vent_Y->value())/ui->Salto_2->value());
            multi.window=ui;
            Busqueda bus(&multi,tipo_dato,Tipo_Descriptor,&Multi_tipo);
            e=bus.Textura(imagen,Size(ui->Vent_X->value(),ui->Vent_Y->value()),ui->Escalas->value(),ui->Salto_2->value(),ui->Rotacion->value(),ui->Postproceso->isChecked(),salida);
        }
        if(!salida.empty()){
            Mat mostrar;
            Representacion rep;
            imagen.convertTo(imagen,CV_32F);
            double minval,maxval;
            cv::minMaxLoc(imagen,&minval,&maxval);
            imagen=(imagen-minval)/(maxval-minval);
            imshow("Imagen",imagen);
            e=rep.Color(salida,Col,mostrar,show_graphics);
            if(e==1){
                QMessageBox msgBox;
                msgBox.setText("ERROR: No se ha podido representar la imagen clasificada");
                msgBox.exec();
                QApplication::restoreOverrideCursor();
                return;
            }
            QApplication::restoreOverrideCursor();
        }
        else{
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se ha podido clasificar la imagen");
            msgBox.exec();
            QApplication::restoreOverrideCursor();
            return;
        }
    }
    ui->progress_Clasificar->setValue(100);
    ui->progress_Clasificar->setValue(0);
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_Postproceso_clicked(bool checked)
{
    if(checked){
        if(ui->radioPosicion->isChecked()){
            ui->label_38->setEnabled(true);
            ui->label_40->setEnabled(true);
            ui->Dist_cuadros->setEnabled(true);
            ui->Rotacion_2->setEnabled(true);
            ui->Solapamiento->setEnabled(true);
            ui->Filtro_aislados->setEnabled(true);
        }
        else if(ui->radioTextura->isChecked()){
            ui->label_38->setEnabled(false);
            ui->label_40->setEnabled(false);
            ui->Dist_cuadros->setEnabled(false);
            ui->Rotacion_2->setEnabled(false);
            ui->Solapamiento->setEnabled(false);
            ui->Filtro_aislados->setEnabled(false);
        }
    }
    else{
        ui->label_38->setEnabled(false);
        ui->label_40->setEnabled(false);
        ui->Dist_cuadros->setEnabled(false);
        ui->Rotacion_2->setEnabled(false);
        ui->Solapamiento->setEnabled(false);
        ui->Filtro_aislados->setEnabled(false);
    }
}

void MainWindow::on_radioPosicion_clicked(bool checked)
{
    if(checked){
        if(ui->Postproceso->isChecked()){
            ui->label_38->setEnabled(true);
            ui->label_40->setEnabled(true);
            ui->Dist_cuadros->setEnabled(true);
            ui->Rotacion_2->setEnabled(true);
            ui->Solapamiento->setEnabled(true);
            ui->Filtro_aislados->setEnabled(true);
        }
        else{
            ui->label_38->setEnabled(false);
            ui->label_40->setEnabled(false);
            ui->Dist_cuadros->setEnabled(false);
            ui->Rotacion_2->setEnabled(false);
            ui->Solapamiento->setEnabled(false);
            ui->Filtro_aislados->setEnabled(false);
        }
    }
    else{
        ui->label_38->setEnabled(false);
        ui->label_40->setEnabled(false);
        ui->Dist_cuadros->setEnabled(false);
        ui->Rotacion_2->setEnabled(false);
        ui->Solapamiento->setEnabled(false);
        ui->Filtro_aislados->setEnabled(false);
    }
}

void MainWindow::on_radioTextura_clicked(bool checked)
{
    if(checked){
        ui->label_38->setEnabled(false);
        ui->label_40->setEnabled(false);
        ui->Dist_cuadros->setEnabled(false);
        ui->Rotacion_2->setEnabled(false);
        ui->Solapamiento->setEnabled(false);
        ui->Filtro_aislados->setEnabled(false);
    }
}

void MainWindow::on_Num_folds_valueChanged(int arg1)
{
    if(!IMAGENES.empty())
        ui->Tam_Folds->setValue(IMAGENES.size()/arg1);
}
