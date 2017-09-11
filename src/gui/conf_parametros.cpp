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

#include "conf_parametros.h"
#include "ui_conf_parametros.h"

Conf_Parametros::Conf_Parametros(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_Parametros)
{
    punt=puntero;
    ui->setupUi(this);
    ui->max_depth_2->setValue(INT_MAX);
    ui->rp_dw_min->setValue(FLT_EPSILON);
}

Conf_Parametros::~Conf_Parametros()
{
    delete ui;
}

void Conf_Parametros::on_pushButton_clicked()
{
    Optimizacion::Parametros *param=(Optimizacion::Parametros*) punt;
    param->Hist_tam_celda=ui->Tam_Celda->value();

    param->KNN_k=ui->K->value();

    param->Boosting_priors=0;
    if(ui->boost_type->currentIndex()==0)
        param->Boosting_boost_type=ml::Boost::DISCRETE;
    else if(ui->boost_type->currentIndex()==1)
        param->Boosting_boost_type=ml::Boost::REAL;
    else if(ui->boost_type->currentIndex()==2)
        param->Boosting_boost_type=ml::Boost::LOGIT;
    else if(ui->boost_type->currentIndex()==3)
        param->Boosting_boost_type=ml::Boost::GENTLE;
    param->Boosting_max_depth=ui->max_depth_3->value();
    param->Boosting_use_surrogates=ui->use_surrogates_3->isChecked();
    param->Boosting_weak_count=ui->weak_count->value();
    param->Boosting_weight_trim_rate=ui->weight_trim_rate->value();

    param->DTrees_priors=0;
    param->DTrees_cv_folds=ui->cv_folds_2->value();
    param->DTrees_max_categories=ui->max_categories_2->value();
    param->DTrees_max_depth=ui->max_depth_2->value();
    param->DTrees_min_sample_count=ui->min_sample_count_2->value();
    param->DTrees_regression_accuracy=ui->regression_accuracy_2->value();
    param->DTrees_truncate_pruned_tree=ui->truncate_pruned_tree_2->isChecked();
    param->DTrees_use_1se_rule=ui->use_1se_rule_2->isChecked();
    param->DTrees_use_surrogates=ui->use_surrogates_2->isChecked();

    if(ui->Tipo_Cov->currentIndex()==0)
        param->EM_covMatType=ml::EM::COV_MAT_SPHERICAL;
    if(ui->Tipo_Cov->currentIndex()==1)
        param->EM_covMatType=ml::EM::COV_MAT_DIAGONAL;
    if(ui->Tipo_Cov->currentIndex()==2)
        param->EM_covMatType=ml::EM::COV_MAT_GENERIC;
    param->EM_nclusters=ui->Num_Clus->value();


//    param->ERTrees_priors=0;
//    param->ERTrees_calc_var_importance=ui->calc_var_importance->isChecked();
//    param->ERTrees_cv_folds=ui->cv_folds->value();
//    param->ERTrees_max_categories=ui->max_categories->value();
//    param->ERTrees_max_depth=ui->max_depth->value();
//    param->ERTrees_min_sample_count=ui->min_sample_count->value();
//    param->ERTrees_native_vars=ui->native_vars->value();
//    param->ERTrees_regression_accuracy=ui->regression_accuracy->value();
//    param->ERTrees_truncate_pruned_tree=ui->truncate_pruned_tree->isChecked();
//    param->ERTrees_use_1se_rule=ui->use_1se_rule->isChecked();
//    param->ERTrees_use_surrogates=ui->use_surrogates->isChecked();

//    if(ui->loss_function_type->currentIndex()==0)
//        param->GBT_loss_function_type=CvGBTrees::SQUARED_LOSS;
//    else if(ui->loss_function_type->currentIndex()==1)
//        param->GBT_loss_function_type=CvGBTrees::ABSOLUTE_LOSS;
//    else if(ui->loss_function_type->currentIndex()==2)
//        param->GBT_loss_function_type=CvGBTrees::HUBER_LOSS;
//    else if(ui->loss_function_type->currentIndex()==3)
//        param->GBT_loss_function_type=CvGBTrees::DEVIANCE_LOSS;
//    param->GBT_max_depth=ui->max_depth_4->value();
//    param->GBT_shrinkage=ui->shrinkage->value();
//    param->GBT_subsample_portion=ui->subsample_portion->value();
//    param->GBT_use_surrogates=ui->use_surrogates_4->isChecked();
//    param->GBT_weak_count=ui->weak_count_2->value();

//    param->Cascada_NumPos=ui->Positivos->value();
//    param->Cascada_NumNeg=ui->Negativos->value();
//    param->Cascada_Mode=ui->Modo->currentText().toStdString();
//    param->Cascada_NumStage=ui->Etapas->value();
//    param->Cascada_MinHitRate=ui->MinHitRate->value();
//    param->Cascada_MaxFalseAlarmRate=ui->MaxFalseAlarmRate->value();
//    param->Cascada_WeightTrimRate=ui->WeightTrimRate->value();
//    param->Cascada_MaxWeakCount=ui->MaxWeakCount->value();
//    param->Cascada_MaxDepth=ui->MaxDepth->value();
//    param->Cascada_Bt=ui->Bt->currentText().toStdString();
//    param->Cascada_PrecalcValBufSize=ui->PrecalcValBufSize->value();
//    param->Cascada_PrecalcidxBufSize=ui->PrecalcidxBufSize->value();


    std::string texto=ui->Num_neur->text().toStdString();
    std::vector<int> pos;
    pos.push_back(-1);
    for(uint i=0; i<texto.size(); i++){
        if(texto[i]==',')
            pos.push_back(i);
    }
    pos.push_back(texto.size());
    std::vector<int> num;
    for(uint i=1; i<pos.size(); i++){
        std::string numero;
        for(int j=pos[i-1]+1; j<pos[i]; j++){
            numero=numero+texto[j];
        }
        int n=atoi(numero.c_str());
        num.push_back(n);
    }
    param->Neuronal_layerSize=cv::Mat::zeros(num.size()+2,1,CV_32SC1);
    param->Neuronal_layerSize.row(param->Neuronal_layerSize.rows-1)=1;
    for(int i=1; i<param->Neuronal_layerSize.rows-1; i++)
        param->Neuronal_layerSize.row(i)=cv::Scalar(num[i-1]);
    if(ui->Metodo->currentIndex()==0)
        param->Neuronal_Method=ml::ANN_MLP::BACKPROP;
    else if(ui->Metodo->currentIndex()==1)
        param->Neuronal_Method=ml::ANN_MLP::RPROP;
    if(ui->Funcion->currentIndex()==0)
        param->Neuronal_Function=ml::ANN_MLP::IDENTITY;
    else if(ui->Funcion->currentIndex()==1)
        param->Neuronal_Function=ml::ANN_MLP::SIGMOID_SYM;
    else if(ui->Funcion->currentIndex()==2)
        param->Neuronal_Function=ml::ANN_MLP::GAUSSIAN;
    param->Neuronal_bp_dw_scale=ui->bp_dw_scale->value();
    param->Neuronal_fparam1=ui->fparam1->value();
    param->Neuronal_fparam2=ui->fparam2->value();
    param->Neuronal_rp_dw0=ui->rp_dw0->value();
    param->Neuronal_rp_dw_max=ui->rp_dw_max->value();
    param->Neuronal_rp_dw_min=ui->rp_dw_min->value();
    param->Neuronal_rp_dw_minus=ui->rp_dw_minus->value();
    param->Neuronal_rp_dw_plus=ui->rp_dw_plus->value();

    param->RTrees_priors=0;
    param->RTrees_calc_var_importance=ui->calc_var_importance->isChecked();
    param->RTrees_cv_folds=ui->cv_folds->value();
    param->RTrees_max_categories=ui->max_categories->value();
    param->RTrees_max_depth=ui->max_depth->value();
    param->RTrees_min_sample_count=ui->min_sample_count->value();
    param->RTrees_native_vars=ui->native_vars->value();
    param->RTrees_regression_accuracy=ui->regression_accuracy->value();
    param->RTrees_truncate_pruned_tree=ui->truncate_pruned_tree->isChecked();
    param->RTrees_use_1se_rule=ui->use_1se_rule->isChecked();
    param->RTrees_use_surrogates=ui->use_surrogates->isChecked();

    if(ui->Train->currentIndex()==0)
        param->SVM_train=TRAIN;
    else if(ui->Train->currentIndex()==1)
        param->SVM_train=TRAIN_AUTO;
    if(ui->Tipo_SVM->currentIndex()==0)
        param->SVM_Type=ml::SVM::C_SVC;
    else if(ui->Tipo_SVM->currentIndex()==1)
        param->SVM_Type=ml::SVM::NU_SVC;
    else if(ui->Tipo_SVM->currentIndex()==2)
        param->SVM_Type=ml::SVM::ONE_CLASS;
    else if(ui->Tipo_SVM->currentIndex()==3)
        param->SVM_Type=ml::SVM::EPS_SVR;
    else if(ui->Tipo_SVM->currentIndex()==4)
        param->SVM_Type=ml::SVM::NU_SVR;
    if(ui->Kernel->currentIndex()==0)
        param->SVM_kernel_type=ml::SVM::LINEAR;
    else if(ui->Kernel->currentIndex()==1)
        param->SVM_kernel_type=ml::SVM::POLY;
    else if(ui->Kernel->currentIndex()==2)
        param->SVM_kernel_type=ml::SVM::RBF;
    else if(ui->Kernel->currentIndex()==3)
        param->SVM_kernel_type=ml::SVM::SIGMOID;
    param->SVM_degree=(double)ui->Degree->value();
    param->SVM_gamma=(double)ui->Gamma->value();
    param->SVM_coef0=(double)ui->Coef0->value();
    param->SVM_C=(double)ui->C->value();
    param->SVM_nu=(double)ui->nu->value();
    param->SVM_p=(double)ui->p->value();

    delete this;
}
