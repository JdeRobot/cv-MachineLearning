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

#ifndef OPTIMIZACION_H
#define OPTIMIZACION_H

#include <opencv2/opencv.hpp>
#include <Herramientas/analisis.h>
#include <Herramientas/auxiliares.h>
#include <Clasificadores/multiclasificador.h>
#ifdef GUI
#include <ui_mainwindow.h>
#endif

using namespace std;
using namespace cv;

namespace MLT {
    class Optimizacion
    {
    public:
        struct Parametros {
            float Hist_tam_celda;

            int KNN_k;
            bool KNN_regression;

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
            Mat SVM_class_weights;
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

//            int GBT_loss_function_type;
//            int GBT_weak_count;
//            float GBT_shrinkage;
//            float GBT_subsample_portion;
//            int GBT_max_depth;
//            bool GBT_use_surrogates;

            int EM_nclusters;
            int EM_covMatType;

//            int ERTrees_max_depth;
//            int ERTrees_min_sample_count;
//            float ERTrees_regression_accuracy;
//            bool ERTrees_use_surrogates;
//            int ERTrees_max_categories;
//            int ERTrees_cv_folds;
//            bool ERTrees_use_1se_rule;
//            bool ERTrees_truncate_pruned_tree;
//            const float* ERTrees_priors;
//            bool ERTrees_calc_var_importance;
//            int ERTrees_native_vars;

            Parametros(){Hist_tam_celda=1;
                         KNN_k=1;KNN_regression=false;
                         Neuronal_Method=ml::ANN_MLP::RPROP; Neuronal_Function=ml::ANN_MLP::SIGMOID_SYM; Neuronal_bp_dw_scale=0.1; Neuronal_bp_moment_scale=0.1; Neuronal_rp_dw0=0.1;
                         Neuronal_rp_dw_max=50.0; Neuronal_rp_dw_min=FLT_EPSILON; Neuronal_rp_dw_minus=0.5; Neuronal_rp_dw_plus=1.2;
                         Neuronal_fparam1=0; Neuronal_fparam2=0; SVM_train=TRAIN; SVM_Type=ml::SVM::NU_SVC; SVM_kernel_type=ml::SVM::RBF; SVM_class_weights=0;
                         SVM_degree=0; SVM_gamma=1; SVM_coef0=0; SVM_C=1; SVM_nu=0.00001; SVM_p=0;
                         RTrees_max_depth=5; RTrees_min_sample_count=10; RTrees_regression_accuracy=0; RTrees_use_surrogates=false; RTrees_max_categories=10;
                         RTrees_cv_folds=0; RTrees_use_1se_rule=false; RTrees_truncate_pruned_tree=false; RTrees_priors=Mat(); RTrees_calc_var_importance=false;
                         RTrees_native_vars=0; DTrees_max_depth=INT_MAX; DTrees_min_sample_count=10; DTrees_regression_accuracy=0.01; DTrees_use_surrogates=true;
                         DTrees_max_categories=10; DTrees_cv_folds=10; DTrees_use_1se_rule=true; DTrees_truncate_pruned_tree=true; DTrees_priors=Mat();
                         Boosting_boost_type=ml::Boost::REAL; Boosting_weak_count=100; Boosting_weight_trim_rate=0.95; Boosting_max_depth=1; Boosting_use_surrogates=false; Boosting_priors=Mat();
//                         GBT_loss_function_type=CvGBTrees::SQUARED_LOSS; GBT_weak_count=200; GBT_shrinkage=0.8f; GBT_subsample_portion=0.01f; GBT_max_depth=3; GBT_use_surrogates=false;
                         EM_nclusters=ml::EM::DEFAULT_NCLUSTERS; EM_covMatType=ml::EM::COV_MAT_DIAGONAL;
//                         ERTrees_max_depth=5; ERTrees_min_sample_count=10; ERTrees_regression_accuracy=0; ERTrees_use_surrogates=false; ERTrees_max_categories=10;
//                         ERTrees_cv_folds=0; ERTrees_use_1se_rule=false; ERTrees_truncate_pruned_tree=false; ERTrees_priors=0; ERTrees_calc_var_importance=false;
//                         ERTrees_native_vars=0;
                        }
        };


        Optimizacion();

        int Validation(vector<Mat> Datos, vector<float> Labels, int Porcentaje_validation, int id_clasificador, Parametros parame, float &Error, Mat &Confusion, vector<Analisis::Ratios_data> &Ratios);
        int Validation(vector<Mat> Datos, vector<float> Labels, int Porcentaje_validation, vector<int> id_clasif, Parametros parame, MultiClasificador::Multi_type multi, float &Error, Mat &Confusion, vector<Analisis::Ratios_data> &Ratios);
        int Cross_Validation(vector<Mat> Datos, vector<float> Labels, int Num_Folds, int Tam_Fold, int id_clasificador, Parametros inicio, Parametros fin, Parametros salto, Parametros &parametros, float &Error, cv::Mat &Confus);
        int Super_Cross_Validation(vector<Mat> Datos, vector<float> Labels, int Num_Folds, int Tam_Fold, vector<int> &id_clasificador, Parametros inicio, Parametros fin, Parametros salto, Parametros &parametros, float &Error, Mat &Confus);
        int Ratios_parametro(vector<Mat> Datos, vector<float> Labels, int porcentaje_validacion, string parametro, Parametros inicio, Parametros fin, Parametros salto, vector<vector<Analisis::Ratios_data> > &Ratios);

    #ifdef GUI
        int progreso;
        int max_progreso;
        int base_progreso;
        int total_progreso;

        Ui::MainWindow *window;
    #endif
    };
}

#endif // OPTIMIZACION_H
