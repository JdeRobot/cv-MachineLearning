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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <iostream>
#include <QMainWindow>
#include <QStandardItem>
#include <qmessagebox.h>
#include <qfiledialog.h>
#include "ui_mainwindow.h"

#include "conf_hog.h"
#include "config_pc.h"

#include "conf_histograma.h"
#include "conf_knn.h"
#include "conf_svm.h"
#include "conf_haar.h"
#include "conf_neuronal.h"
#include "conf_rtrees.h"
//#include "conf_ertrees.h"
#include "conf_dtrees.h"
#include "conf_em.h"
//#include "conf_gb.h"
#include "conf_boosting.h"

#include "conf_parametros.h"
#include "con_multi.h"
#include "conf_multi.h"
#include "conf_sc_vali.h"
#include "selec_param.h"

#include "../ClasificationSystem.h"
#include "../Clasificadores/miclasificador.h"

#include <fstream>
#include <opencv2/opencv.hpp>

using namespace MLT;


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    
    Ui::MainWindow *ui;

    string Dat_Ref;
    string result_ref;

    String Tipo_Des,Tipo_Ext;
    float Parametro;

    Size Win_Size, Block_Stride;
    double Win_Sigma, Threshold_L2hys;
    bool Gamma_Correction;
    int Nlevels;

    float Hist_tam_celda;

    int KNN_k;
    bool KNN_regression;

    Size Cascada_Tam;
    int Cascada_NumPos;
    int Cascada_NumNeg;
    string Cascada_Mode;
    int Cascada_NumStage;
    float Cascada_MinHitRate;
    float Cascada_MaxFalseAlarmRate;
    float Cascada_WeightTrimRate;
    int Cascada_MaxWeakCount;
    int Cascada_MaxDepth;
    string Cascada_Bt;
    int Cascada_PrecalcValBufSize;
    int Cascada_PrecalcidxBufSize;
    bool si_entrenar;

    Mat Neuronal_layerSize;
    int Neuronal_Method;
    int Neuronal_Function;
    double Neuronal_bp_dw_scale;
    double Neuronal_bp_moment_scale;
    double Neuronal_rp_dw0;
    double Neuronal_rp_dw_max;
    double Neuronal_rp_dw_min;
    double Neuronal_rp_dw_minus;
    double Neuronal_rp_dw_plus;
    double Neuronal_fparam1;
    double Neuronal_fparam2;

    int SVM_train;
    int SVM_Type;
    int SVM_kernel_type;
    CvMat *SVM_class_weights;
    double SVM_degree;
    double SVM_gamma;
    double SVM_coef0;
    double SVM_C;
    double SVM_nu;
    double SVM_p;

    int RTrees_max_depth;
    int RTrees_min_sample_count;
    float RTrees_regression_accuracy;
    bool RTrees_use_surrogates;
    int RTrees_max_categories;
    int RTrees_cv_folds;
    bool RTrees_use_1se_rule;
    bool RTrees_truncate_pruned_tree;
    Mat RTrees_priors;
    bool RTrees_calc_var_importance;
    int RTrees_native_vars;

    int DTrees_max_depth;
    int DTrees_min_sample_count;
    float DTrees_regression_accuracy;
    bool DTrees_use_surrogates;
    int DTrees_max_categories;
    int DTrees_cv_folds;
    bool DTrees_use_1se_rule;
    bool DTrees_truncate_pruned_tree;
    Mat DTrees_priors;

    int Boosting_boost_type;
    int Boosting_weak_count;
    double Boosting_weight_trim_rate;
    int Boosting_max_depth;
    bool Boosting_use_surrogates;
    Mat Boosting_priors;

    int EM_nclusters;
    int EM_covMatType;

//    int GBT_loss_function_type;
//    int GBT_weak_count;
//    float GBT_shrinkage;
//    float GBT_subsample_portion;
//    int GBT_max_depth;
//    bool GBT_use_surrogates;

//    int ERTrees_max_depth;
//    int ERTrees_min_sample_count;
//    float ERTrees_regression_accuracy;
//    bool ERTrees_use_surrogates;
//    int ERTrees_max_categories;
//    int ERTrees_cv_folds;
//    bool ERTrees_use_1se_rule;
//    bool ERTrees_truncate_pruned_tree;
//    Mat ERTrees_priors;
//    bool ERTrees_calc_var_importance;
//    int ERTrees_native_vars;

    vector<int> id_clasificadores;
    vector<string> nombres;
    MultiClasificador::Multi_type Multi_tipo;

    Optimizacion::Parametros inicio,salto,fin;

    int X,Y;

    string parametro;

private slots:
    void on_Tipo_Herr_activated(int index);

    void on_Iniciar_clicked();

    void on_toolButton_clicked();

    void on_Representar_2_clicked();

    void on_toolButton_2_clicked();

    void on_Cargar_2_clicked();

    void on_Analizar_2_clicked();

    void on_Generar_2_clicked();

    void on_Guardar_clicked();

    void on_Clus_Representar_clicked();

    void on_Button_Calidad_clicked();

    void on_Generar_3_clicked();

    void on_pushButton_4_clicked();

    void on_Guardar_2_clicked();

    void on_Tipo_Clus_activated(int index);

    void on_toolButton_4_clicked();

    void on_Iniciar_2_clicked();

    void on_Cargar_3_clicked();

    void on_Iniciar_3_clicked();

    void on_Analizar_3_clicked();

    void on_Clus_Representar_2_clicked();

    void on_Iniciar_4_clicked();

    void on_toolButton_14_clicked();

    void on_Parametros_clicked();

    void on_Salto_clicked();

    void on_Parada_clicked();

    void on_C_Validation_clicked();

    void on_Validation_clicked();

    void on_Validation2_clicked();

    void on_SC_Validation_clicked();

    void on_toolButton_7_clicked();

    void on_toolButton_20_clicked();

    void on_toolButton_9_clicked();

    void on_Ratios_Parametro_clicked();

    void on_toolButton_22_clicked();

    void on_Representar_3_clicked();

    void on_Iniciar_6_clicked();

    void on_toolButton_19_clicked();

    void on_toolButton_busqueda_clicked();

    void on_Postproceso_clicked(bool checked);

    void on_radioPosicion_clicked(bool checked);

    void on_radioTextura_clicked(bool checked);

    void on_Num_folds_valueChanged(int arg1);

    void on_Config_Des_clicked();

    void on_Config_Des_2_clicked();

private:
    vector<cv::Scalar> Col;
    Scalar color;

    Generacion::Info_Datos info,Info;
    std::vector<float> LABELS;
    std::vector<cv::Mat> IMAGENES;
    std::vector<float> Labels;
    std::vector<cv::Mat> Imagenes;
    std::vector<float> resultado;
    int ID;

    int num_bar;
    bool show_graphics;
    bool save_clasif;
    bool save_data;
    bool save_other;
    bool read;
    bool ifreduc;

    Clasificador_Distancias D;
    Clasificador_Gaussiano G;
    Clasificador_Cascada HA;
    Clasificador_Histograma H;
    Clasificador_KNN K;
    Clasificador_Neuronal N;
    Clasificador_SVM S;
    Clasificador_RTrees RT;
    Clasificador_DTrees DT;
    Clasificador_Boosting B;
//    Clasificador_GBT GB;
    Clasificador_EM E;
//    Clasificador_ERTrees ER;
    MiClasificador MC;
};
#endif // MAINWINDOW_H
