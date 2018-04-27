#include "mainwindow.h"
#include "ui_mainwindow.h"

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

void MainWindow::on_v_tool_activated(int index)
{
    if(index==2)
        ui->v_config_tool->setEnabled(true);
    else
        ui->v_config_tool->setEnabled(false);
    if (index==1 || index==4){
        this->ui->v_datapath->setEnabled(true);
        this->ui->v_toolButton->setEnabled(true);
    }
    else{
        this->ui->v_datapath->setEnabled(false);
        this->ui->v_toolButton->setEnabled(false);
    }
}

void MainWindow::on_v_run_datamanaging_clicked()
{

}
