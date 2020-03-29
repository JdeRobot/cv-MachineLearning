#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    string g="../Data/Config_Data/Initial.xml";
    cv::FileStorage fs(g,FileStorage::READ);
    fs["Colors"]>>Col;

    fs["Tipo_Des"]>>descriptor;
    fs["Tipo_Ext"]>>extractor;
    fs["Parametro"]>>descriptor_parameter;

//    fs["Win_Size"]>>Win_Size;
//    fs["Block_Stride"]>>Block_Stride;
    this->win_size_x=64;
    this->win_size_y=128;
    this->block_x=8;
    this->block_y=8;
//    Win_Size=Size(64, 128);
//    Block_Stride=Size(8, 8);
    fs["Win_Sigma"]>>this->sigma;
    fs["Threshold_L2hys"]>>this->threhold_l2hys;
    fs["Gamma_Correction"]>>this->gamma;
    fs["Nlevels"]>>this->nlevels;

//    fs["ID"]>>ID;

    fs["Hist_tam_celda"]>>this->parameters.Hist_tam_celda;

    fs["KNN_k"]>>this->parameters.KNN_k;
    fs["KNN_regression"]>>this->parameters.KNN_regression;

//    fs["Cascada_Tam"]>>Cascada_Tam;
//    this->parameters.Cascada_Tam=Size(24,24);
    fs["Cascada_NumPos"]>>this->parameters.Cascada_NumPos;
    fs["Cascada_NumNeg"]>>this->parameters.Cascada_NumNeg;
    fs["Cascada_Mode"]>>this->parameters.Cascada_Mode;
    fs["Cascada_NumStage"]>>this->parameters.Cascada_NumStage;
    fs["Cascada_MinHitRate"]>>this->parameters.Cascada_MinHitRate;
    fs["Cascada_MaxFalseAlarmRate"]>>this->parameters.Cascada_MaxFalseAlarmRate;
    fs["Cascada_WeightTrimRate"]>>this->parameters.Cascada_WeightTrimRate;
    fs["Cascada_MaxWeakCount"]>>this->parameters.Cascada_MaxWeakCount;
    fs["Cascada_MaxDepth"]>>this->parameters.Cascada_MaxDepth;
    fs["Cascada_Bt"]>>this->parameters.Cascada_Bt;
    fs["Cascada_PrecalcValBufSize"]>>this->parameters.Cascada_PrecalcValBufSize;
    fs["Cascada_PrecalcidxBufSize"]>>this->parameters.Cascada_PrecalcidxBufSize;
    fs["si_entrenar"]>>si_entrenar;

    fs["Neuronal_Method"]>>this->parameters.Neuronal_Method;
    fs["Neuronal_Function"]>>this->parameters.Neuronal_Function;
    fs["Neuronal_bp_dw_scale"]>>this->parameters.Neuronal_bp_dw_scale;
    fs["Neuronal_bp_moment_scale"]>>this->parameters.Neuronal_bp_moment_scale;
    fs["Neuronal_rp_dw0"]>>this->parameters.Neuronal_rp_dw0;
    fs["Neuronal_rp_dw_max"]>>this->parameters.Neuronal_rp_dw_max;
    fs["Neuronal_rp_dw_min"]>>this->parameters.Neuronal_rp_dw_min;
    fs["Neuronal_rp_dw_minus"]>>this->parameters.Neuronal_rp_dw_minus;
    fs["Neuronal_rp_dw_plus"]>>this->parameters.Neuronal_rp_dw_plus;
    fs["Neuronal_fparam1"]>>this->parameters.Neuronal_fparam1;
    fs["Neuronal_fparam2"]>>this->parameters.Neuronal_fparam2;

    fs["SVM_train"]>>this->parameters.SVM_train;
    fs["SVM_Type"]>>this->parameters.SVM_Type;
    fs["SVM_kernel_type"]>>this->parameters.SVM_kernel_type;
    this->parameters.SVM_class_weights=0;
//    fs["SVM_class_weights"]>>SVM_class_weights;
    fs["SVM_degree"]>>this->parameters.SVM_degree;
    fs["SVM_coef0"]>>this->parameters.SVM_coef0;
    fs["SVM_nu"]>>this->parameters.SVM_nu;

    this->parameters.RTrees_priors=Mat();
    fs["RTrees_max_depth"]>>this->parameters.RTrees_max_depth;
    fs["RTrees_min_sample_count"]>>this->parameters.RTrees_min_sample_count;
    fs["RTrees_regression_accuracy"]>>this->parameters.RTrees_regression_accuracy;
    fs["RTrees_use_surrogates"]>>this->parameters.RTrees_use_surrogates;
    fs["RTrees_max_categories"]>>this->parameters.RTrees_max_categories;
    fs["RTrees_cv_folds"]>>this->parameters.RTrees_cv_folds;
    fs["RTrees_use_1se_rule"]>>this->parameters.RTrees_use_1se_rule;
    fs["RTrees_truncate_pruned_tree"]>>this->parameters.RTrees_truncate_pruned_tree;
    fs["RTrees_calc_var_importance"]>>this->parameters.RTrees_calc_var_importance;
    fs["RTrees_native_vars"]>>this->parameters.RTrees_native_vars;

    this->parameters.DTrees_priors=Mat();
    fs["DTrees_max_depth"]>>this->parameters.DTrees_max_depth;
    fs["DTrees_min_sample_count"]>>this->parameters.DTrees_min_sample_count;
    fs["DTrees_regression_accuracy"]>>this->parameters.DTrees_regression_accuracy;
    fs["DTrees_use_surrogates"]>>this->parameters.DTrees_use_surrogates;
    fs["DTrees_max_categories"]>>this->parameters.DTrees_max_categories;
    fs["DTrees_cv_folds"]>>this->parameters.DTrees_cv_folds;
    fs["DTrees_use_1se_rule"]>>this->parameters.DTrees_use_1se_rule;
    fs["DTrees_truncate_pruned_tree"]>>this->parameters.DTrees_truncate_pruned_tree;

    this->parameters.Boosting_priors=Mat();
    fs["Boosting_boost_type"]>>this->parameters.Boosting_boost_type;
    fs["Boosting_weak_count"]>>this->parameters.Boosting_weak_count;
    fs["Boosting_weight_trim_rate"]>>this->parameters.Boosting_weight_trim_rate;
    fs["Boosting_max_depth"]>>this->parameters.Boosting_max_depth;
    fs["Boosting_use_surrogates"]>>this->parameters.Boosting_use_surrogates;

    fs["EM_nclusters"]>>this->parameters.EM_nclusters;
    fs["EM_covMatType"]>>this->parameters.EM_covMatType;

//    fs["GBT_loss_function_type"]>>this->parameters.GBT_loss_function_type;
//    fs["GBT_weak_count"]>>this->parameters.GBT_weak_count;
//    fs["GBT_shrinkage"]>>this->parameters.GBT_shrinkage;
//    fs["GBT_subsample_portion"]>>this->parameters.GBT_subsample_portion;
//    fs["GBT_max_depth"]>>this->parameters.GBT_max_depth;
//    fs["GBT_use_surrogates"]>>this->parameters.GBT_use_surrogates;

//    ERTrees_priors=Mat();
//    fs["ERTrees_max_depth"]>>this->parameters.ERTrees_max_depth;
//    fs["ERTrees_min_sample_count"]>>this->parameters.ERTrees_min_sample_count;
//    fs["ERTrees_regression_accuracy"]>>this->parameters.ERTrees_regression_accuracy;
//    fs["ERTrees_use_surrogates"]>>this->parameters.ERTrees_use_surrogates;
//    fs["ERTrees_max_categories"]>>this->parameters.ERTrees_max_categories;
//    fs["ERTrees_cv_folds"]>>this->parameters.ERTrees_cv_folds;
//    fs["ERTrees_use_1se_rule"]>>this->parameters.ERTrees_use_1se_rule;
//    fs["ERTrees_truncate_pruned_tree"]>>this->parameters.ERTrees_truncate_pruned_tree;
//    fs["ERTrees_calc_var_importance"]>>this->parameters.ERTrees_calc_var_importance;
//    fs["ERTrees_native_vars"]>>this->parameters.ERTrees_native_vars;


    X=0;
    Y=0;


    fs["num_bar"]>>this->num_bar;
    fs["show_graphics"]>>this->show_graphics;
    fs["save_clasif"]>>this->save_clasif;
    fs["save_data"]>>this->save_data;
    fs["save_other"]>>this->save_other;
    fs["read"]>>this->read;
    fs["ifreduc"]>>this->ifreduc;

    fs.release();

    ui->setupUi(this);

    this->run.window=ui;

    this->run.num_bar=this->num_bar;
    this->run.show_graphics=this->show_graphics;
    this->run.save_clasif=this->save_clasif;
    this->run.save_data=this->save_data;
    this->run.save_other=this->save_other;
    this->run.read=this->read;
    this->run.ifreduc=this->ifreduc;
    this->run.colors=Col;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::error_control(QString error){
    QMessageBox msgBox;
    msgBox.setText(error);
    msgBox.exec();
    QApplication::restoreOverrideCursor();
    this->ui->g_progress_datamanaging->setValue(0);
    this->ui->g_progress_Analysis->setValue(0);
    this->ui->v_progress_datamanaging->setValue(0);
    this->ui->v_progress_Analysis->setValue(0);
    this->ui->m_progress_classifiers->setValue(0);
    this->ui->i_progress_datamanaging->setValue(0);
}


void MainWindow::on_g_tool_activated(int index)
{
    if (index==1 || index==3){
        this->ui->g_datapath->setEnabled(true);
        this->ui->g_toolButton->setEnabled(true);
    }
    else{
        this->ui->g_datapath->setEnabled(false);
        this->ui->g_toolButton->setEnabled(false);
    }
    if(index==1 || index==0){
        this->ui->g_dataname->setEnabled(false);
    }
    else
        this->ui->g_dataname->setEnabled(true);
}


void MainWindow::on_g_toolButton_clicked()
{
    QString filename= QFileDialog::getExistingDirectory(
                this,
                tr("CHOOSE FOLDER"),
                QDir::currentPath()+"/../Data/Imagenes/");
    this->ui->g_datapath->setText(filename);
}

void MainWindow::on_g_run_datamanaging_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if(this->ui->g_tool->currentIndex()==0){
        error_control("ERROR: Choose a tool");
        return;
    }
    QString path=this->ui->g_datapath->displayText();
    QString name=this->ui->g_dataname->displayText();
    string ref=name.toStdString();

    if(this->ui->g_tool->currentIndex()==2 || this->ui->g_tool->currentIndex()==3 || this->ui->g_tool->currentIndex()==4){
        for(uint i=0; i<ref.size(); i++){
            if(ref[i]==' '){
                error_control("ERROR: Name must not have spaces");
                return;
            }
        }
    }

    int er=0;
        this->ui->g_progress_datamanaging->setValue(1);
    if(this->ui->g_tool->currentIndex()==1){
        er=this->run.load_dataset(path.toStdString());
        if(er==1){
            error_control("ERROR: The folder has not the expected structure");
            return;
        }

        if(er==2){
            error_control("ERROR: Data could not be loaded");
            return;
        }
    }
    else if(this->ui->g_tool->currentIndex()==2){
        er=this->run.save(ref);
        if(er==1){
            error_control("ERROR: Data could not be created");
            return;
        }
        path="../Data/Imagenes/"+name;
        er=this->run.load_dataset(path.toStdString());
        if(er==1){
            error_control("ERROR: The folder has not the expected structure");
            return;
        }

        if(er==2){
            error_control("ERROR: Data could not be loaded");
            return;
        }
    }
    else if(this->ui->g_tool->currentIndex()==3){
        er=this->run.join_data(ref,path.toStdString());
        if(er==1){
            error_control("ERROR: Data could not be created");
            return;
        }
        path="../Data/Imagenes/"+name;
        er=this->run.load_dataset(path.toStdString());
        if(er==1){
            error_control("ERROR: The folder has not the expected structure");
            return;
        }

        if(er==2){
            error_control("ERROR: Data could not be loaded");
            return;
        }
    }


    this->ui->v_plotting_x->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());
    this->ui->v_plotting_y->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());
    this->ui->v_plotting_dimension->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());

    this->ui->v_dimensionality_dimensions->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());
    this->ui->v_plotting_dimension->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());

    QString reference=QString::fromStdString(this->run.org_ref);
    this->ui->g_progress_datamanaging->setValue(100);
    this->ui->g_progress_datamanaging->setValue(0);
    this->ui->dataset_lab->setText("Dataset: "+reference);
    this->ui->v_progress_datamanaging->setValue(100);
    this->ui->v_progress_datamanaging->setValue(0);
    this->ui->i_progress_datamanaging->setValue(100);
    this->ui->i_progress_datamanaging->setValue(0);

    QApplication::restoreOverrideCursor();
}

void MainWindow::on_g_analysis_analyse_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    this->ui->g_progress_Analysis->setValue(1);
    if(this->run.org_images.empty()){
        error_control("ERROR: There is not data loaded");
        return;
    }

    QStandardItemModel *model=new QStandardItemModel(0,0);
    int er=0;

    if(this->run.result_labels.empty()){
        error_control("ERROR: There is not labels loaded");
        return;
    }
    er=this->run.analyse_result(model);

    if(er==1){
        error_control("ERROR: Statistics could not be calculated");
        return;
    }

    this->ui->g_analysis_statistics->setModel(model);
    this->ui->g_progress_Analysis->setValue(100);
    this->ui->g_progress_Analysis->setValue(0);
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_v_tool_activated(int index)
{
    if(index==1){
        this->ui->v_config_tool->setEnabled(true);
        this->ui->v_dataname->setEnabled(true);
    }
    else{
        this->ui->v_config_tool->setEnabled(false);
        this->ui->v_dataname->setEnabled(false);
    }
}

void MainWindow::on_v_run_datamanaging_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if(this->ui->v_tool->currentIndex()==0){
        error_control("ERROR: Choose a tool");
        return;
    }
    QString name=this->ui->v_dataname->displayText();
    string ref=name.toStdString();

    if(this->ui->v_tool->currentIndex()==1){
        for(uint i=0; i<ref.size(); i++){
            if(ref[i]==' '){
                error_control("ERROR: Name must not have spaces");
                return;
            }
        }
    }

    int er=0;
    this->ui->v_progress_datamanaging->setValue(1);
    if(this->ui->v_tool->currentIndex()==1){
        int num_classes=this->ui->v_clases->value();
        int num_data_class=this->ui->v_dataperclass->value();
        int scale_x=this->ui->v_vectordimension->value();
        int scale_y=1;
        float width=this->ui->v_variance->value();
        float interclass=this->ui->v_interclassdistance->value();

        er=this->run.synthetic_data(ref,num_classes,num_data_class,scale_x,scale_y,width,interclass);
        if(er==1){
            error_control("ERROR: Data could not be created");
            return;
        }
    }
    this->ui->v_plotting_x->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());
    this->ui->v_plotting_y->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());
    this->ui->v_plotting_dimension->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());

    this->ui->v_dimensionality_dimensions->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());
    this->ui->v_plotting_dimension->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());

    QString reference=QString::fromStdString(this->run.org_ref);
    this->ui->dataset_lab->setText("Dataset: "+reference);
    this->ui->g_progress_datamanaging->setValue(100);
    this->ui->g_progress_datamanaging->setValue(0);
    this->ui->v_progress_datamanaging->setValue(100);
    this->ui->v_progress_datamanaging->setValue(0);
    this->ui->i_progress_datamanaging->setValue(100);
    this->ui->i_progress_datamanaging->setValue(0);

    QApplication::restoreOverrideCursor();
}

void MainWindow::on_v_analysis_analyse_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    this->ui->v_progress_Analysis->setValue(1);
    if(this->run.org_images.empty()){
        error_control("ERROR: There is not data loaded");
        return;
    }

    QStandardItemModel *model=new QStandardItemModel(0,0);
    int er=0;
    if(this->run.org_labels.empty()){
        error_control("ERROR: There is not labels loaded");
        return;
    }
    if(!this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels()>1024){
        QMessageBox msgBox;
        msgBox.setText("The number of dimensions is over 1024, so covariance is not calculated");
        msgBox.exec();
    }
    er=this->run.analyse_data(model);

    if(er==1){
        error_control("ERROR: Statistics could not be calculated");
        return;
    }

    this->ui->v_analysis_statistics->setModel(model);
    this->ui->v_progress_Analysis->setValue(100);
    this->ui->v_progress_Analysis->setValue(0);
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_v_plotting_represent_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if(this->run.org_images.empty()){
        error_control("ERROR: There is not data loaded");
        return;
    }
    int type_plot=-1;
    vector<int> dim;
    if(this->ui->v_plotting_data->isChecked()){
        type_plot=0;
        dim.push_back(this->ui->v_plotting_x->value());
        dim.push_back(this->ui->v_plotting_y->value());
    }
    else if(this->ui->v_plotting_ellipses->isChecked()){
        type_plot=1;
        dim.push_back(this->ui->v_plotting_x->value());
        dim.push_back(this->ui->v_plotting_y->value());
    }
    else if(this->ui->v_plotting_dataellipeses->isChecked()){
        type_plot=2;
        dim.push_back(this->ui->v_plotting_x->value());
        dim.push_back(this->ui->v_plotting_y->value());
    }
    else if(this->ui->v_plotting_histogram->isChecked()){
        type_plot=3;
        dim.push_back(this->ui->v_plotting_dimension->value());
    }
    if(this->ui->v_plotting_results->isChecked())
        type_plot=type_plot+4;

    int e=this->run.plot_data(type_plot,dim);
    if(e==1){
        error_control("ERROR: Data could not be plotted");
        return;
    }
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_v_clustering_method_activated(int index)
{
    if(index==1){
        this->ui->v_clustering_label_classes->setEnabled(true);
        this->ui->v_clustering_label_initialization->setEnabled(true);
        this->ui->v_clustering_label_repetitions->setEnabled(true);
        this->ui->v_clustering_classes->setEnabled(true);
        this->ui->v_clustering_initialization->setEnabled(true);
        this->ui->v_clustering_repetitions->setEnabled(true);
    }
    else{
        this->ui->v_clustering_label_classes->setEnabled(false);
        this->ui->v_clustering_label_initialization->setEnabled(false);
        this->ui->v_clustering_label_repetitions->setEnabled(false);
        this->ui->v_clustering_classes->setEnabled(false);
        this->ui->v_clustering_initialization->setEnabled(false);
        this->ui->v_clustering_repetitions->setEnabled(false);
    }
    if(index==2 || index==3){
        this->ui->v_clustering_label_maxdistance->setEnabled(true);
        this->ui->v_clustering_maxdistance->setEnabled(true);
    }
    else{
        this->ui->v_clustering_label_maxdistance->setEnabled(false);
        this->ui->v_clustering_maxdistance->setEnabled(false);
    }
    if(index==4){
        this->ui->v_clustering_label_cellsize->setEnabled(true);
        this->ui->v_clustering_cellsize->setEnabled(true);
    }
    else{
        this->ui->v_clustering_label_cellsize->setEnabled(false);
        this->ui->v_clustering_cellsize->setEnabled(false);
    }
    if(index==5){
        this->ui->v_clustering_label_classes->setEnabled(true);
        this->ui->v_clustering_classes->setEnabled(true);
        this->ui->v_clustering_covariance->setEnabled(true);
        this->ui->v_clustering_covariance->setEnabled(true);
    }
    else{
        this->ui->v_clustering_label_classes->setEnabled(false);
        this->ui->v_clustering_classes->setEnabled(false);
        this->ui->v_clustering_covariance->setEnabled(false);
        this->ui->v_clustering_covariance->setEnabled(false);
    }
}

void MainWindow::on_v_clustering_generate_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if(this->run.org_images.empty()){
        error_control("ERROR: There is not data loaded");
        return;
    }
    QString name=this->ui->v_clustering_dataname->displayText();
    string ref=name.toStdString();
    for(uint i=0; i<ref.size(); i++){
        if(ref[i]==' '){
            error_control("ERROR: Name must not have spaces");
            return;
        }
    }

    int k=this->ui->v_clustering_classes->value();
    int repetitions=this->ui->v_clustering_repetitions->value();
    float max_dist=this->ui->v_clustering_maxdistance->value();
    float cell_size=this->ui->v_clustering_cellsize->value();
    int type=0;
    if(this->ui->v_clustering_method->currentIndex()==1 && this->ui->v_clustering_initialization->currentIndex()==0)
        type=1;
    else if(this->ui->v_clustering_method->currentIndex()==1 && this->ui->v_clustering_initialization->currentIndex()==1)
        type=2;
    else if(this->ui->v_clustering_method->currentIndex()==2)
        type=3;
    else if(this->ui->v_clustering_method->currentIndex()==3)
        type=4;
    else if(this->ui->v_clustering_method->currentIndex()==4)
        type=5;
    else if(this->ui->v_clustering_method->currentIndex()==5 && this->ui->v_clustering_covariance->currentIndex()==0)
        type=6;
    else if(this->ui->v_clustering_method->currentIndex()==5 && this->ui->v_clustering_covariance->currentIndex()==1)
        type=7;
    else if(this->ui->v_clustering_method->currentIndex()==5 && this->ui->v_clustering_covariance->currentIndex()==2)
        type=8;

    int e=this->run.clustering(ref,type,k,repetitions,max_dist,cell_size);
    if(e==1){
        error_control("ERROR: Clustering could not be done");
        return;
    }

    QString reference=QString::fromStdString(ref);
    this->ui->results_lab->setText("Results: "+reference);

    QApplication::restoreOverrideCursor();

}

void MainWindow::on_v_dimensionality_generate_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if(this->run.org_images.empty()){
        error_control("ERROR: There is not data loaded");
        return;
    }
    QString name=this->ui->v_dimensionality_dataname->displayText();
    string ref=name.toStdString();
    for(uint i=0; i<ref.size(); i++){
        if(ref[i]==' '){
            error_control("ERROR: Name must not have spaces");
            return;
        }
    }

    int type=-1;
    if(this->ui->v_dimensionality_lda->isChecked())
        type=LDA_DIM;
    else if(this->ui->v_dimensionality_pca->isChecked())
        type=PCA_DIM;
    else if(this->ui->v_dimensionality_max_dist->isChecked())
        type=MAXDIST_DIM;
    else if(this->ui->v_dimensionality_dprime->isChecked())
        type=D_PRIME_DIM;

    int e=run.dimensionality(ref,this->ui->v_dimensionality_dimensions->value(),type);
    if(e==1){
        error_control("ERROR: Dimensionality reduction could not be done");
        return;
    }

    QString reference=QString::fromStdString(ref);
    this->ui->results_lab->setText("Results: "+reference);

    QApplication::restoreOverrideCursor();
}

void MainWindow::on_v_dimensionality_quality_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if(this->run.org_images.empty()){
        error_control("ERROR: There is not data loaded");
        return;
    }
    if(this->run.org_labels.empty()){
        error_control("ERROR: There is not labels loaded");
        return;
    }
    QString name=this->ui->v_dimensionality_dataname->displayText();
    string ref=name.toStdString();

    int type=-1;
    if(this->ui->v_dimensionality_lda->isChecked())
        type=LDA_DIM;
    else if(this->ui->v_dimensionality_pca->isChecked())
        type=PCA_DIM;
    else if(this->ui->v_dimensionality_max_dist->isChecked())
        type=MAXDIST_DIM;
    else if(this->ui->v_dimensionality_dprime->isChecked())
        type=D_PRIME_DIM;

    int measure=-1;
    if(this->ui->v_dimensionality_distance_parameter->isChecked())
        measure=0;
    else if(this->ui->v_dimensionality_dprime_parameter->isChecked())
        measure=1;

    string result;
    int e=run.dimension_cuality(ref,this->ui->v_dimensionality_dimensions->value(),type, measure, result);
    if(e==1){
        error_control("ERROR: Dimensionality reduction could not be done");
        return;
    }

    QMessageBox msgBox;
    msgBox.setText(QString::fromStdString(result));
    msgBox.exec();

    cv::waitKey(0);

    QApplication::restoreOverrideCursor();
    return;
}

void MainWindow::on_i_tool_activated(int index)
{
    if(index==0){
        this->ui->i_dataname->setEnabled(false);
        this->ui->i_datatype->setEnabled(false);
        this->ui->i_datapath->setEnabled(false);
        this->ui->i_toolButton->setEnabled(false);
        this->ui->i_config_tool->setEnabled(false);

        this->ui->i_label_clases->setEnabled(false);
        this->ui->i_label_dataperclass->setEnabled(false);
        this->ui->i_label_size->setEnabled(false);
        this->ui->i_label_variance->setEnabled(false);
        this->ui->i_label_interclassdistance->setEnabled(false);
        this->ui->i_label_images->setEnabled(false);
        this->ui->i_label_maxnoise->setEnabled(false);
        this->ui->i_label_maxblur->setEnabled(false);
        this->ui->i_label_maxrotx->setEnabled(false);
        this->ui->i_label_maxroty->setEnabled(false);
        this->ui->i_label_maxrotz->setEnabled(false);

        this->ui->i_clases->setEnabled(false);
        this->ui->i_dataperclass->setEnabled(false);
        this->ui->i_size_x->setEnabled(false);
        this->ui->i_size_y->setEnabled(false);
        this->ui->i_variance->setEnabled(false);
        this->ui->i_interclassdistance->setEnabled(false);
        this->ui->i_images->setEnabled(false);
        this->ui->i_maxnoise->setEnabled(false);
        this->ui->i_maxblur->setEnabled(false);
        this->ui->i_maxrotx->setEnabled(false);
        this->ui->i_maxroty->setEnabled(false);
        this->ui->i_maxrotz->setEnabled(false);

    }
    else if(index==1){
        this->ui->i_dataname->setEnabled(true);
        this->ui->i_datatype->setEnabled(true);
        this->ui->i_datapath->setEnabled(false);
        this->ui->i_toolButton->setEnabled(false);
        this->ui->i_config_tool->setEnabled(true);

        this->ui->i_label_clases->setEnabled(true);
        this->ui->i_label_dataperclass->setEnabled(true);
        this->ui->i_label_size->setEnabled(true);
        this->ui->i_label_variance->setEnabled(true);
        this->ui->i_label_interclassdistance->setEnabled(true);
        this->ui->i_label_images->setEnabled(false);
        this->ui->i_label_maxnoise->setEnabled(false);
        this->ui->i_label_maxblur->setEnabled(false);
        this->ui->i_label_maxrotx->setEnabled(false);
        this->ui->i_label_maxroty->setEnabled(false);
        this->ui->i_label_maxrotz->setEnabled(false);

        this->ui->i_clases->setEnabled(true);
        this->ui->i_dataperclass->setEnabled(true);
        this->ui->i_size_x->setEnabled(true);
        this->ui->i_size_y->setEnabled(true);
        this->ui->i_variance->setEnabled(true);
        this->ui->i_interclassdistance->setEnabled(true);
        this->ui->i_images->setEnabled(false);
        this->ui->i_maxnoise->setEnabled(false);
        this->ui->i_maxblur->setEnabled(false);
        this->ui->i_maxrotx->setEnabled(false);
        this->ui->i_maxroty->setEnabled(false);
        this->ui->i_maxrotz->setEnabled(false);
    }
    else if(index>=2 && index<=8){
        this->ui->i_dataname->setEnabled(true);
        this->ui->i_datatype->setEnabled(true);
        this->ui->i_datapath->setEnabled(true);
        this->ui->i_toolButton->setEnabled(true);
        this->ui->i_config_tool->setEnabled(true);

        this->ui->i_label_clases->setEnabled(false);
        this->ui->i_label_dataperclass->setEnabled(false);
        this->ui->i_label_size->setEnabled(true);
        this->ui->i_label_variance->setEnabled(false);
        this->ui->i_label_interclassdistance->setEnabled(false);
        this->ui->i_label_images->setEnabled(false);
        this->ui->i_label_maxnoise->setEnabled(false);
        this->ui->i_label_maxblur->setEnabled(false);
        this->ui->i_label_maxrotx->setEnabled(false);
        this->ui->i_label_maxroty->setEnabled(false);
        this->ui->i_label_maxrotz->setEnabled(false);

        this->ui->i_clases->setEnabled(false);
        this->ui->i_dataperclass->setEnabled(false);
        this->ui->i_size_x->setEnabled(true);
        this->ui->i_size_y->setEnabled(true);
        this->ui->i_variance->setEnabled(false);
        this->ui->i_interclassdistance->setEnabled(false);
        this->ui->i_images->setEnabled(false);
        this->ui->i_maxnoise->setEnabled(false);
        this->ui->i_maxblur->setEnabled(false);
        this->ui->i_maxrotx->setEnabled(false);
        this->ui->i_maxroty->setEnabled(false);
        this->ui->i_maxrotz->setEnabled(false);
    }
    else if(index==9){
        this->ui->i_dataname->setEnabled(true);
        this->ui->i_datatype->setEnabled(true);
        this->ui->i_datapath->setEnabled(false);
        this->ui->i_toolButton->setEnabled(true);
        this->ui->i_config_tool->setEnabled(false);

        this->ui->i_label_clases->setEnabled(false);
        this->ui->i_label_dataperclass->setEnabled(false);
        this->ui->i_label_size->setEnabled(false);
        this->ui->i_label_variance->setEnabled(false);
        this->ui->i_label_interclassdistance->setEnabled(false);
        this->ui->i_label_images->setEnabled(false);
        this->ui->i_label_maxnoise->setEnabled(false);
        this->ui->i_label_maxblur->setEnabled(false);
        this->ui->i_label_maxrotx->setEnabled(false);
        this->ui->i_label_maxroty->setEnabled(false);
        this->ui->i_label_maxrotz->setEnabled(false);

        this->ui->i_clases->setEnabled(false);
        this->ui->i_dataperclass->setEnabled(false);
        this->ui->i_size_x->setEnabled(false);
        this->ui->i_size_y->setEnabled(false);
        this->ui->i_variance->setEnabled(false);
        this->ui->i_interclassdistance->setEnabled(false);
        this->ui->i_images->setEnabled(false);
        this->ui->i_maxnoise->setEnabled(false);
        this->ui->i_maxblur->setEnabled(false);
        this->ui->i_maxrotx->setEnabled(false);
        this->ui->i_maxroty->setEnabled(false);
        this->ui->i_maxrotz->setEnabled(false);
    }
    if(index==10){
        this->ui->i_dataname->setEnabled(true);
        this->ui->i_datatype->setEnabled(false);
        this->ui->i_datapath->setEnabled(true);
        this->ui->i_toolButton->setEnabled(true);
        this->ui->i_config_tool->setEnabled(true);

        this->ui->i_label_clases->setEnabled(false);
        this->ui->i_label_dataperclass->setEnabled(false);
        this->ui->i_label_size->setEnabled(false);
        this->ui->i_label_variance->setEnabled(false);
        this->ui->i_label_interclassdistance->setEnabled(false);
        this->ui->i_label_images->setEnabled(true);
        this->ui->i_label_maxnoise->setEnabled(true);
        this->ui->i_label_maxblur->setEnabled(true);
        this->ui->i_label_maxrotx->setEnabled(true);
        this->ui->i_label_maxroty->setEnabled(true);
        this->ui->i_label_maxrotz->setEnabled(true);

        this->ui->i_clases->setEnabled(false);
        this->ui->i_dataperclass->setEnabled(false);
        this->ui->i_size_x->setEnabled(false);
        this->ui->i_size_y->setEnabled(false);
        this->ui->i_variance->setEnabled(false);
        this->ui->i_interclassdistance->setEnabled(false);
        this->ui->i_images->setEnabled(true);
        this->ui->i_maxnoise->setEnabled(true);
        this->ui->i_maxblur->setEnabled(true);
        this->ui->i_maxrotx->setEnabled(true);
        this->ui->i_maxroty->setEnabled(true);
        this->ui->i_maxrotz->setEnabled(true);
    }
    if(index==4 || index==5 || index==7 || index==8)
        this->ui->i_square->setEnabled(true);
    else
        this->ui->i_square->setEnabled(false);
    if(index==6 || index==8){
        this->ui->i_label_images->setEnabled(true);
        this->ui->i_images->setEnabled(true);
    }
    if(index==9){
        this->ui->i_datamanaging_descriptortype->setEnabled(true);
        this->ui->i_datamanaging_descriptortool->setEnabled(true);
    }
    else{
        this->ui->i_datamanaging_descriptortype->setEnabled(false);
        this->ui->i_datamanaging_descriptortool->setEnabled(false);
    }
    if(index==0)
        this->ui->i_datatype->setEnabled(false);
    else
        this->ui->i_datatype->setEnabled(true);
}

void MainWindow::on_i_toolButton_clicked()
{
    QString filename= QFileDialog::getExistingDirectory(
                this,
                tr("CHOOSE FOLDER"),
                QDir::currentPath()+"/../Data/Imagenes/");
    this->ui->i_datapath->setText(filename);
}

void MainWindow::on_i_datamanaging_descriptortool_clicked()
{
    if(this->ui->i_datamanaging_descriptortype->currentIndex()==0){
        QMessageBox msgBox;
        msgBox.setText("ERROR: Choose a descriptor");
        msgBox.exec();
        return;
    }
    if(this->ui->i_datamanaging_descriptortype->currentIndex()==11){
        Conf_HOG *conf_HOG= new Conf_HOG(this,this);
        conf_HOG->show();
    }
    else if(this->ui->i_datamanaging_descriptortype->currentIndex()==12){
        Config_PC *conf_PC= new Config_PC(this,this);
        conf_PC->show();
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: The selected descriptor hasn't got configuration");
        msgBox.exec();
        return;
    }
}

void MainWindow::on_i_run_datamanaging_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if(this->ui->i_tool->currentIndex()==0){
        error_control("ERROR: Choose a tool");
        return;
    }
    QString path=this->ui->i_datapath->displayText();
    QString name=this->ui->i_dataname->displayText();
    string ref=name.toStdString();

    if(this->ui->i_tool->currentIndex()==2 || this->ui->i_tool->currentIndex()==3 || this->ui->i_tool->currentIndex()==4){
        for(uint i=0; i<ref.size(); i++){
            if(ref[i]==' '){
                error_control("ERROR: Name must not have spaces");
                return;
            }
        }
    }

    int er=0;
        this->ui->i_progress_datamanaging->setValue(1);
    if(this->ui->i_tool->currentIndex()==1){
        int num_classes=this->ui->i_clases->value();
        int num_data_class=this->ui->i_dataperclass->value();
        int scale_x=this->ui->i_size_x->value();
        int scale_y=this->ui->i_size_y->value();
        float width=this->ui->i_variance->value();
        float interclass=this->ui->i_interclassdistance->value();

        er=this->run.synthetic_data(ref,num_classes,num_data_class,scale_x,scale_y,width,interclass);
        if(er==1){
            error_control("ERROR: Data could not be created");
            return;
        }
    }
    else if(this->ui->i_tool->currentIndex()>=2 && this->ui->i_tool->currentIndex()<=8){
        int type=-1;
        if(this->ui->i_tool->currentIndex()==2)
            type=0;
        else if(this->ui->i_tool->currentIndex()==3)
            type=1;
        else if(this->ui->i_tool->currentIndex()==4)
            type=2;
        else if(this->ui->i_tool->currentIndex()==5)
            type=3;
        else if(this->ui->i_tool->currentIndex()==6)
            type=4;
        else if(this->ui->i_tool->currentIndex()==7)
            type=5;
        else if(this->ui->i_tool->currentIndex()==8)
            type=6;

        int scale_x=this->ui->i_size_x->value();
        int scale_y=this->ui->i_size_y->value();
        er=this->run.generate_data(ref,path.toStdString(),type,scale_x,scale_y,this->ui->i_square->isChecked(),this->ui->i_images->value());
        if(er==1){
            error_control("ERROR: Data could not be created");
            return;
        }
    }
    else if(this->ui->i_tool->currentIndex()==9){
        int descriptor=this->ui->i_datamanaging_descriptortype->currentIndex()-1;
        er=this->run.descriptors(ref,descriptor,this->descriptor,this->extractor,this->win_size_x,this->win_size_y,this->block_x,this->block_y,this->sigma,this->threhold_l2hys,this->gamma,this->nlevels, this->descriptor_parameter);
        if(er==1){
            error_control("ERROR: Data could not be created");
            return;
        }
        else if(er==2){
            error_control("ERROR: Hog size bigger than images");
            return;
        }
        else if(er==3){
            error_control("ERROR: Choose a descriptor");
            return;
        }
        QString reference=QString::fromStdString(ref);
        this->ui->results_lab->setText("Results: "+reference);
    }
    else if(this->ui->i_tool->currentIndex()==10){
        int nframe=this->ui->i_images->value();
        float max_noise=this->ui->i_maxnoise->value();
        float max_blur=this->ui->i_maxblur->value();
        float max_x=this->ui->i_maxrotx->value();
        float max_y=this->ui->i_maxroty->value();
        float max_z=this->ui->i_maxrotz->value();
        er=this->run.expand_dataset(ref,nframe,max_noise,max_blur,max_x,max_y,max_z);
        if(er==1){
            error_control("ERROR: Data could not be created");
            return;
        }
        QString reference=QString::fromStdString(ref);
        this->ui->results_lab->setText("Results: "+reference);
    }

    this->ui->v_plotting_x->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());
    this->ui->v_plotting_y->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());
    this->ui->v_plotting_dimension->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());

    this->ui->v_dimensionality_dimensions->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());
    this->ui->v_plotting_dimension->setMaximum(this->run.org_images[0].cols*this->run.org_images[0].rows*this->run.org_images[0].channels());

    QString reference=QString::fromStdString(this->run.org_ref);
    this->ui->dataset_lab->setText("Dataset: "+reference);
    this->ui->g_progress_datamanaging->setValue(100);
    this->ui->g_progress_datamanaging->setValue(0);
    this->ui->v_progress_datamanaging->setValue(100);
    this->ui->v_progress_datamanaging->setValue(0);
    this->ui->i_progress_datamanaging->setValue(100);
    this->ui->i_progress_datamanaging->setValue(0);

    QApplication::restoreOverrideCursor();
}

void MainWindow::on_i_represent_clicked()
{
    int type=-1;
    if(this->ui->i_plotting_dataset->isChecked())
        type=0;
    else if(this->ui->i_plotting_result->isChecked())
        type=1;
    if((int)this->ui->i_plotting_label->value()==0)
        this->ui->i_plotting_label->setValue(-1);
    int e=this->run.represent_images(type,(int)this->ui->i_plotting_label->value());

    if(e==1)
        error_control("ERROR: An Error occured while showing the images");
    else if(e==2)
        error_control("ERROR: There is not data loaded");
    else if(e==3)
        error_control("ERROR: There is not result dataset");
}

void MainWindow::on_i_detection_postprocess_clicked(bool checked)
{
    if(checked && this->ui->i_detection_position->isChecked()){
        ui->i_detection_overlap->setEnabled(true);
        ui->i_detection_filteralone->setEnabled(true);
        ui->i_detection_distance->setEnabled(true);
        ui->i_detection_label_distance->setEnabled(true);
        ui->i_detection_angle->setEnabled(true);
        ui->i_detection_label_angle->setEnabled(true);
    }
    else{
        ui->i_detection_overlap->setEnabled(false);
        ui->i_detection_filteralone->setEnabled(false);
        ui->i_detection_distance->setEnabled(false);
        ui->i_detection_label_distance->setEnabled(false);
        ui->i_detection_angle->setEnabled(false);
        ui->i_detection_label_angle->setEnabled(false);
    }
}

void MainWindow::on_i_detection_position_clicked(bool checked)
{
    if(checked && this->ui->i_detection_postprocess->isChecked()){
        ui->i_detection_overlap->setEnabled(true);
        ui->i_detection_filteralone->setEnabled(true);
        ui->i_detection_distance->setEnabled(true);
        ui->i_detection_label_distance->setEnabled(true);
        ui->i_detection_angle->setEnabled(true);
        ui->i_detection_label_angle->setEnabled(true);
    }
    else{
        ui->i_detection_overlap->setEnabled(false);
        ui->i_detection_filteralone->setEnabled(false);
        ui->i_detection_distance->setEnabled(false);
        ui->i_detection_label_distance->setEnabled(false);
        ui->i_detection_angle->setEnabled(false);
        ui->i_detection_label_angle->setEnabled(false);
    }
}

void MainWindow::on_i_detection_segmentation_clicked(bool checked)
{
    if(checked){
        ui->i_detection_overlap->setEnabled(false);
        ui->i_detection_filteralone->setEnabled(false);
        ui->i_detection_distance->setEnabled(false);
        ui->i_detection_label_distance->setEnabled(false);
        ui->i_detection_angle->setEnabled(false);
        ui->i_detection_label_angle->setEnabled(false);
    }
}

void MainWindow::on_i_detection_random_clicked(bool checked)
{
    if(checked){
        this->ui->i_detection_label_nclasses->setEnabled(true);
        this->ui->i_detection_label_variance->setEnabled(true);
        this->ui->i_detection_label_interclass->setEnabled(true);
        this->ui->i_detection_nclasses->setEnabled(true);
        this->ui->i_detection_variance->setEnabled(true);
        this->ui->i_detection_interclass->setEnabled(true);
        this->ui->i_detection_file_tool->setEnabled(false);
        this->ui->i_detection_path->setEnabled(false);
    }
    else{
        this->ui->i_detection_label_nclasses->setEnabled(false);
        this->ui->i_detection_label_variance->setEnabled(false);
        this->ui->i_detection_label_interclass->setEnabled(false);
        this->ui->i_detection_nclasses->setEnabled(false);
        this->ui->i_detection_variance->setEnabled(false);
        this->ui->i_detection_interclass->setEnabled(false);
        this->ui->i_detection_file_tool->setEnabled(true);
        this->ui->i_detection_path->setEnabled(true);
    }
}

void MainWindow::on_i_detection_file_tool_clicked()
{
    QString filename = QFileDialog::getOpenFileName(
                             this,
                             tr("- CHOOSE AN IMAGE -"),
                             QDir::currentPath()+"..",
                             tr("Document files (*.bmp *.dib *.jpeg *.jpg *.jpe *.jp2 *.png *.pbm *.pgm *.ppm *.sr *.ras *.tiff *.tif);;All files (*.*)") );
    ui->i_detection_path->setText(filename);
}

void MainWindow::on_i_detection_datatype_tool_clicked()
{
    if(ui->i_detection_descriptor_type->currentIndex()==0){
        QMessageBox msgBox;
        msgBox.setText("ERROR: Select a descriptor");
        msgBox.exec();
        return;
    }
    if(ui->i_detection_descriptor_type->currentIndex()==10){
        Conf_HOG *conf_HOG= new Conf_HOG(this,this);
        conf_HOG->show();
    }
    else if(ui->i_detection_descriptor_type->currentIndex()==11){
        Config_PC *conf_PC= new Config_PC(this,this);
        conf_PC->show();
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: The chosen descriptor hasn't got configuration");
        msgBox.exec();
        return;
    }
}

void MainWindow::on_i_detection_multiclassifier_tool_clicked()
{
    Con_Multi *mul=new Con_Multi(this,this);
    mul->show();
}

void MainWindow::on_i_detection_run_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int type_running=0;
    if(this->ui->i_detection_position->isChecked() && this->ui->i_detection_loaded->isChecked())
        type_running=1;
    else if(this->ui->i_detection_segmentation->isChecked() && this->ui->i_detection_loaded->isChecked())
        type_running=2;
    else if(this->ui->i_detection_position->isChecked() && this->ui->i_detection_multiclassifier->isChecked())
        type_running=3;
    else if(this->ui->i_detection_segmentation->isChecked() && this->ui->i_detection_multiclassifier->isChecked())
        type_running=4;


    int input_type=0;
    string path;

    if(this->ui->i_detection_random->isChecked()){
        path="random";
        input_type=1;
    }
    else{
        path=this->ui->i_detection_path->text().toStdString();
        input_type=2;
    }

    int transform_type=-1;
    if(ui->i_detection_descriptor_type->currentIndex()==1)
        transform_type=RGB;
    else if(ui->i_detection_descriptor_type->currentIndex()==2)
        transform_type=GRAY;
    else if(ui->i_detection_descriptor_type->currentIndex()==3)
        transform_type=HSV;
    else if(ui->i_detection_descriptor_type->currentIndex()==4)
        transform_type=H_CHANNEL;
    else if(ui->i_detection_descriptor_type->currentIndex()==5)
        transform_type=S_CHANNEL;
    else if(ui->i_detection_descriptor_type->currentIndex()==6)
        transform_type=V_CHANNEL;
    else if(ui->i_detection_descriptor_type->currentIndex()==7)
        transform_type=THRESHOLD;
    else if(ui->i_detection_descriptor_type->currentIndex()==8)
        transform_type=CANNY;
    else if(ui->i_detection_descriptor_type->currentIndex()==9)
        transform_type=SOBEL;
    else if(ui->i_detection_descriptor_type->currentIndex()==10)
        transform_type=COLOR_PREDOMINANTE;
    else if(ui->i_detection_descriptor_type->currentIndex()==11)
        transform_type=HOG_DES;
    else if(ui->i_detection_descriptor_type->currentIndex()==12)
        transform_type=PUNTOS_CARACTERISTICOS;
    else{
        error_control("ERROR: Choose a descriptor");
        return;
    }


    cv::Mat image,labels;
    vector<cv::RotatedRect> detections;
    vector<float> labels_detections;
    int n_classes=this->ui->i_detection_nclasses->value();
    float variance=this->ui->i_detection_variance->value();
    float interclass=this->ui->i_detection_interclass->value();
    int window_x=this->ui->i_detection_windowx->value();
    int window_y=this->ui->i_detection_windowy->value();
    int jump=this->ui->i_detection_jump->value();
    int pyramid=this->ui->i_detection_pyramid->value();
    int rotation=this->ui->i_detection_rotation->value();
    bool postprocess=this->ui->i_detection_postprocess->isChecked();
    bool overlap=this->ui->i_detection_overlap->isChecked();
    bool isolated=this->ui->i_detection_filteralone->isChecked();
    float dist_boxes=this->ui->i_detection_distance->value();
    int dist_rotation=this->ui->i_detection_angle->value();

    int e=this->run.detect_image(type_running, input_type,path,transform_type, this->multi_type,
                           n_classes,variance,interclass,window_x,window_y,jump,pyramid,rotation,
                           postprocess,overlap,isolated,dist_boxes,dist_rotation,this->descriptor,this->extractor,
                           this->win_size_x,this->win_size_y,this->block_x,this->block_y,this->sigma,this->threhold_l2hys,
                           this->gamma,this->nlevels, this->descriptor_parameter,
                           image,labels,detections,labels_detections);
    if(e==1){
        error_control("ERROR: Image could not be loaded");
        return;
    }
    else if(e==2){
            error_control("ERROR: Window bigger thatn image");
            return;
    }
    else if(e==3){
        error_control("ERROR: How size bigger than images");
        return;
    }
    else if(e==4){
        error_control("ERROR: Detection didn't work");
        return;
    }
    QApplication::restoreOverrideCursor();

    cv::Mat show;
    imshow("Image",image);
    Representacion rep;
    if(type_running==1 || type_running==3){
        rep.Recuadros(image,detections,labels_detections,this->Col,show,this->show_graphics);
    }
    else if(type_running==2 || type_running==4){
        rep.Color(labels,this->Col,show,this->show_graphics);
    }
}

void MainWindow::on_m_classifier_typetool_clicked()
{
    if(ui->m_classifier_type->currentIndex()==1){
        QMessageBox msgBox;
        msgBox.setText("INFO: Distance classifier has not parameters");
        msgBox.exec();
        return;
    }
    else if(ui->m_classifier_type->currentIndex()==2){
        QMessageBox msgBox;
        msgBox.setText("INFO: Bayesian (Gauss) classifier has not parameters");
        msgBox.exec();
        return;
    }
    else if(ui->m_classifier_type->currentIndex()==3){
        Conf_Histograma *window=new Conf_Histograma(this,this);
        window->show();
    }
    else if(ui->m_classifier_type->currentIndex()==4){
        Conf_KNN *window=new Conf_KNN(this,this);
        window->show();
    }
    else if(ui->m_classifier_type->currentIndex()==5){
        Conf_neuronal *window=new Conf_neuronal(this,this);
        window->show();
    }
    else if(ui->m_classifier_type->currentIndex()==6){
        Conf_SVM *window=new Conf_SVM(this,this);
        window->show();
    }
    else if(ui->m_classifier_type->currentIndex()==7){
        Conf_DTrees *window=new Conf_DTrees(this,this);
        window->show();
    }
    else if(ui->m_classifier_type->currentIndex()==8){
        Conf_RTrees *window=new Conf_RTrees(this,this);
        window->show();
    }
    else if(ui->m_classifier_type->currentIndex()==9){
        Conf_Boosting *window=new Conf_Boosting(this,this);
        window->show();
    }
    else if(ui->m_classifier_type->currentIndex()==10){
        Conf_HAAR *window=new Conf_HAAR(this,this);
        window->show();
    }
    else if(ui->m_classifier_type->currentIndex()==11){
        Conf_HAAR *window=new Conf_HAAR(this,this);
        window->show();
    }
    else if(ui->m_classifier_type->currentIndex()==12){
        Conf_EM *window=new Conf_EM(this,this);
        window->show();
    }
//        else if(ui->Tipo_Clasif->currentIndex()==14){
//                Conf_Histograma *window=new Conf_Histograma(this,this);
//                window->show();
//        }
//        else if(ui->Tipo_Clasif->currentIndex()==15){
//            Conf_Histograma *window=new Conf_Histograma(this,this);
//            window->show();
//        }

}

void MainWindow::on_m_classifier_train_clicked()
{
    if(this->run.org_images.empty()){
        error_control("ERROR: There is not data loaded");
        return;
    }
    if(this->run.org_labels.empty()){
        error_control("ERROR: There is not labels loaded");
        return;
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    string ref=ui->m_classifier_name->displayText().toStdString();
    int classifier_type=ui->m_classifier_type->currentIndex();

    int er=0;
    er=this->run.train(ref,classifier_type,this->parameters);
    if(er==1){
        error_control("ERROR: The classifier could not be trained");
        return;
    }
    QString reference="Model: "+QString::fromStdString(ref);
    this->ui->classifier_lab->setText(reference);

    QApplication::restoreOverrideCursor();
}

void MainWindow::on_m_load_tool_clicked()
{
    QString filename= QFileDialog::getExistingDirectory(
                this,
                tr("CHOOSE FOLDER"),
                QDir::currentPath()+"/../Data/Configuracion/");
    this->ui->m_load_address->setText(filename);
}

void MainWindow::on_m_load_load_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QString path=this->ui->m_load_address->displayText();
    string ref;

    int er=0;
    er=this->run.load_model(path.toStdString(),ref);
    if(er==1){
        error_control("ERROR: The folder has not the expected structure");
        return;
    }
    QString reference="Model: "+QString::fromStdString(ref);
    this->ui->classifier_lab->setText(reference);

    QApplication::restoreOverrideCursor();
}

void MainWindow::on_m_classifier_config_multitool_clicked()
{
    Con_Multi *mul=new Con_Multi(this,this);
    mul->show();
}

void MainWindow::on_m_classifier_classify_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if(this->run.org_images.empty()){
        error_control("ERROR: There is not data loaded");
        return;
    }
    this->ui->m_progress_classifiers->setValue(1);
    string ref=ui->m_classifier_configname->displayText().toStdString();
    for(uint i=0; i<ref.size(); i++){
        if(ref[i]==' '){
            error_control("ERROR: Name must not have spaces");
            return;
        }
    }
    stringstream txt;
    int type=0;
    if(this->ui->m_classifier_config_loaded->isChecked())
        type=1;
    else if(this->ui->m_classifier_config_multi->isChecked())
        type=2;
    int er=this->run.classify(ref,type,txt,this->multi_type);
    if(er==1){
        error_control("ERROR: There is not classifier loaded");
        return;
    }
    else if(er==2){
        error_control("ERROR: Labels with 0. Wrong classification");
        return;
    }
    ui->m_results->setText(QString::fromStdString(txt.str()));
    this->ui->results_lab->setText("Results: "+QString::fromStdString(ref));
    this->ui->m_progress_classifiers->setValue(100);
    this->ui->m_progress_classifiers->setValue(0);
    QApplication::restoreOverrideCursor();
}

void MainWindow::on_m_optimize_validation_clicked()
{
    ui->m_optimize_parameters->setEnabled(true);
    ui->m_optimize_leap->setEnabled(false);
    ui->m_optimize_stop->setEnabled(false);
    ui->m_optimize_label_validation->setEnabled(true);
    ui->m_optimize_label_nfolds->setEnabled(false);
    ui->m_optimize_label_foldsize->setEnabled(false);
    ui->m_optimize_validation_percentage->setEnabled(true);
    ui->m_optimize_nfolds->setEnabled(false);
    ui->m_optimize_foldsize->setEnabled(false);
    ui->m_optimize_classifier->setEnabled(true);
}
void MainWindow::on_m_optimize_validation_multi_clicked()
{
    ui->m_optimize_parameters->setEnabled(true);
    ui->m_optimize_leap->setEnabled(false);
    ui->m_optimize_stop->setEnabled(false);
    ui->m_optimize_label_validation->setEnabled(true);
    ui->m_optimize_label_nfolds->setEnabled(false);
    ui->m_optimize_label_foldsize->setEnabled(false);
    ui->m_optimize_validation_percentage->setEnabled(true);
    ui->m_optimize_nfolds->setEnabled(false);
    ui->m_optimize_foldsize->setEnabled(false);
    ui->m_optimize_classifier->setEnabled(false);
}

void MainWindow::on_m_optimize_crossvalidation_clicked()
{
    ui->m_optimize_parameters->setEnabled(true);
    ui->m_optimize_leap->setEnabled(true);
    ui->m_optimize_stop->setEnabled(true);
    ui->m_optimize_label_validation->setEnabled(false);
    ui->m_optimize_label_nfolds->setEnabled(true);
    ui->m_optimize_label_foldsize->setEnabled(true);
    ui->m_optimize_validation_percentage->setEnabled(false);
    ui->m_optimize_nfolds->setEnabled(true);
    ui->m_optimize_foldsize->setEnabled(true);
    ui->m_optimize_classifier->setEnabled(true);
}

void MainWindow::on_m_optimize_supercross_clicked()
{
    ui->m_optimize_parameters->setEnabled(true);
    ui->m_optimize_leap->setEnabled(true);
    ui->m_optimize_stop->setEnabled(true);
    ui->m_optimize_label_validation->setEnabled(false);
    ui->m_optimize_label_nfolds->setEnabled(true);
    ui->m_optimize_label_foldsize->setEnabled(true);
    ui->m_optimize_validation_percentage->setEnabled(false);
    ui->m_optimize_nfolds->setEnabled(true);
    ui->m_optimize_foldsize->setEnabled(true);
    ui->m_optimize_classifier->setEnabled(false);
}

void MainWindow::on_m_optimize_validationtool_clicked()
{
    Con_Multi *mul=new Con_Multi(this,this);
    mul->show();
}

void MainWindow::on_m_optimize_svalidationtool_clicked()
{
    Conf_SC_Vali *SC=new Conf_SC_Vali(this,this);
    SC->show();
}

void MainWindow::on_m_optimize_parameters_clicked()
{
    Conf_Parametros *conf=new Conf_Parametros(&this->start,this);
    conf->show();
}

void MainWindow::on_m_optimize_leap_clicked()
{
    Conf_Parametros *conf=new Conf_Parametros(&this->leap,this);
    conf->show();
}

void MainWindow::on_m_optimize_stop_clicked()
{
    Conf_Parametros *conf=new Conf_Parametros(&this->stop,this);
    conf->show();
}

void MainWindow::on_m_optimize_run_clicked()
{
    if(this->run.org_images.empty()){
        error_control("ERROR: There is not data loaded");
        return;
    }
    if(this->run.org_labels.empty()){
        error_control("ERROR: There is not labels loaded");
        return;
    }
    QApplication::setOverrideCursor(Qt::WaitCursor);
    stringstream text;

    int id_classifier=0;
    int type=0;
    int percentage=this->ui->m_optimize_validation_percentage->value();
    int num_folds=this->ui->m_optimize_nfolds->value();
    int size_fold=this->ui->m_optimize_foldsize->value();

    if(this->ui->m_optimize_validation->isChecked())
        type=1;
    else if(this->ui->m_optimize_validation_multi->isChecked())
        type=2;
    else if(this->ui->m_optimize_crossvalidation->isChecked())
        type=3;
    else if(this->ui->m_optimize_supercross->isChecked())
        type=4;

    if(type==1 || type==3){
        if(ui->m_optimize_classifier->currentIndex()==1)
            id_classifier=DISTANCIAS;
        else if(ui->m_optimize_classifier->currentIndex()==2)
            id_classifier=GAUSSIANO;
        else if(ui->m_optimize_classifier->currentIndex()==3)
            id_classifier=HISTOGRAMA;
        else if(ui->m_optimize_classifier->currentIndex()==4)
            id_classifier=KNN;
        else if(ui->m_optimize_classifier->currentIndex()==5)
            id_classifier=NEURONAL;
        else if(ui->m_optimize_classifier->currentIndex()==6)
            id_classifier=C_SVM;
        else if(ui->m_optimize_classifier->currentIndex()==7)
            id_classifier=RTREES;
        else if(ui->m_optimize_classifier->currentIndex()==8)
            id_classifier=DTREES;
        else if(ui->m_optimize_classifier->currentIndex()==9)
            id_classifier=BOOSTING;
        else if(ui->m_optimize_classifier->currentIndex()==10)
            id_classifier=EXP_MAX;
    //        else if(ui->Tipo_Clasif->currentIndex()==11)
    //            id_clasificador=GBT;
    //        else if(ui->Tipo_Clasif->currentIndex()==12)
    //            id_clasificador=ERTREES;
        else{
            error_control("ERROR: Choose a classifier");
            return;
        }
    }


    int e=this->run.optimize(type,id_classifier,this->multi_type,text,this->start,this->leap,this->stop,percentage,num_folds,size_fold);

    this->ui->m_optimize_text->setText(QString::fromStdString((text.str())));
    if(e==1){
        error_control("ERROR: There is not data loaded");
        return;
    }
    else if(e==2){
        error_control("ERROR: There is not labels loaded");
        return;
    }
    else if(e==3){
        error_control("ERROR: Neural Network classififer wrong configuration");
        return;
    }
    else if(e==4){
        error_control("ERROR: Optimizacion didn't work");
        return;
    }
    else if(e==5){
        error_control("ERROR: This classifier hasn't got parameters");
        return;
    }
    else if(e==6){
        error_control("ERROR: Not implemented");
        return;
    }
    else if(e==7){
        error_control("ERROR: num_folds*size_fold > number of images");
        return;
    }
    else if(e==8){
        error_control("ERROR: Neural Network classififer wrong configuration");
        return;
    }
    QApplication::restoreOverrideCursor();
}
