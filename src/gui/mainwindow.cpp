#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    string g="../Data/Config_Data/Initial.xml";
    cv::FileStorage fs(g,CV_STORAGE_READ);
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
    this->ui->v_progress_datamanaging->setValue(0);
    this->ui->v_progress_Analysis->setValue(0);
    this->ui->m_progress_classifiers->setValue(0);
    this->ui->i_progress_datamanaging->setValue(0);
}

void MainWindow::on_v_tool_activated(int index)
{
    if(index==2)
        this->ui->v_config_tool->setEnabled(true);
    else
        this->ui->v_config_tool->setEnabled(false);
    if (index==1 || index==4){
        this->ui->v_datapath->setEnabled(true);
        this->ui->v_toolButton->setEnabled(true);
    }
    else{
        this->ui->v_datapath->setEnabled(false);
        this->ui->v_toolButton->setEnabled(false);
    }
    if(index==1 || index==0){
        this->ui->v_dataname->setEnabled(false);
    }
    else
        this->ui->v_dataname->setEnabled(true);
}

void MainWindow::on_v_toolButton_clicked()
{
    QString filename= QFileDialog::getExistingDirectory(
                this,
                tr("CHOOSE FOLDER"),
                QDir::currentPath()+"/../Data/Imagenes/");
    this->ui->v_datapath->setText(filename);
}

void MainWindow::on_v_run_datamanaging_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    if(this->ui->v_tool->currentIndex()==0){
        error_control("ERROR: Choose a tool");
        return;
    }
    QString path=this->ui->v_datapath->displayText();
    QString name=this->ui->v_dataname->displayText();
    string ref=name.toStdString();

    if(this->ui->v_tool->currentIndex()==2 || this->ui->v_tool->currentIndex()==3 || this->ui->v_tool->currentIndex()==4){
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
    else if(this->ui->v_tool->currentIndex()==2){
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
    else if(this->ui->v_tool->currentIndex()==3){
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
    else if(this->ui->v_tool->currentIndex()==4){
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
    this->ui->dataset_lab->setText("Dataset: "+reference);
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
    if(this->ui->v_analysis_dataset->isChecked()){
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
    }
    else if(this->ui->v_analysis_results->isChecked()){
        if(this->run.result_images.empty()){
            error_control("ERROR: There is not labels loaded");
            return;
        }
        er=this->run.analyse_result(model);
    }

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

    run.clustering(ref,type,k,repetitions,max_dist,cell_size);



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

    run.dimensionality(ref,this->ui->v_dimensionality_dimensions->value(),type);

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
    QString name=this->ui->v_dimensionality_dataname->displayText();
    string ref=name.toStdString();
//    for(uint i=0; i<ref.size(); i++){
//        if(ref[i]==' '){
//            error_control("ERROR: Name must not have spaces");
//            return;
//        }
//    }

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
    run.dimension_cuality(ref,this->ui->v_dimensionality_dimensions->value(),type, measure, result);
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
    else if(index==1 || index==12 || index==13){
        this->ui->i_dataname->setEnabled(false);
        this->ui->i_datatype->setEnabled(true);
        this->ui->i_datapath->setEnabled(true);
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
        this->ui->i_variance->setEnabled(true);
        this->ui->i_interclassdistance->setEnabled(false);
        this->ui->i_images->setEnabled(false);
        this->ui->i_maxnoise->setEnabled(false);
        this->ui->i_maxblur->setEnabled(false);
        this->ui->i_maxrotx->setEnabled(false);
        this->ui->i_maxroty->setEnabled(false);
        this->ui->i_maxrotz->setEnabled(false);
    }
    else if(index==2){
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
    else if(index==3){
        this->ui->i_dataname->setEnabled(false);
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
    else if(index>=4 && index<=9){
        this->ui->i_dataname->setEnabled(false);
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
    else if(index==10){
        this->ui->i_dataname->setEnabled(true);
        this->ui->i_datatype->setEnabled(true);
        this->ui->i_datapath->setEnabled(true);
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
    if(index==11){
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
    if(index==5 || index==6 || index==8 || index==9)
        this->ui->i_square->setEnabled(true);
    else
        this->ui->i_square->setEnabled(false);
    if(index==7 || index==9){
        this->ui->i_label_images->setEnabled(true);
        this->ui->i_images->setEnabled(true);
    }
    if(index==10){
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
    if(ui->i_datamanaging_descriptortype->currentIndex()==0){
        QMessageBox msgBox;
        msgBox.setText("ERROR: Choose a descriptor");
        msgBox.exec();
        return;
    }
    if(ui->i_datamanaging_descriptortype->currentIndex()==11){
        Conf_HOG *conf_HOG= new Conf_HOG(this,this);
        conf_HOG->show();
    }
    else if(ui->i_datamanaging_descriptortype->currentIndex()==12){
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
    else if(this->ui->i_tool->currentIndex()==2){
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
    else if(this->ui->i_tool->currentIndex()>=3 && this->ui->i_tool->currentIndex()<=9){
        int type=-1;
        if(this->ui->i_tool->currentIndex()==3)
            type=0;
        else if(this->ui->i_tool->currentIndex()==4)
            type=1;
        else if(this->ui->i_tool->currentIndex()==5)
            type=2;
        else if(this->ui->i_tool->currentIndex()==6)
            type=3;
        else if(this->ui->i_tool->currentIndex()==7)
            type=4;
        else if(this->ui->i_tool->currentIndex()==8)
            type=5;
        else if(this->ui->i_tool->currentIndex()==9)
            type=6;

        int scale_x=this->ui->i_size_x->value();
        int scale_y=this->ui->i_size_y->value();
        er=this->run.generate_data(ref,path.toStdString(),type,scale_x,scale_y,this->ui->i_square->isChecked(),this->ui->i_images->value());
        if(er==1){
            error_control("ERROR: Data could not be created");
            return;
        }
    }
    else if(this->ui->i_tool->currentIndex()==10){
        int descriptor=this->ui->i_datamanaging_descriptortype->currentIndex()-1;
        er=this->run.descriptors(ref,descriptor,this->descriptor,this->extractor,this->win_size_x,this->win_size_y,this->block_x,this->block_y,this->sigma,this->threhold_l2hys,this->gamma,this->nlevels);
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
    else if(this->ui->i_tool->currentIndex()==11){
//        er=this->run.join_data(ref,path.toStdString());
//        if(er==1){
//            error_control("ERROR: Data could not be created");
//            return;
//        }
//        path="../Data/Imagenes/"+name;
//        er=this->run.load_dataset(path.toStdString());
//        if(er==1){
//            error_control("ERROR: The folder has not the expected structure");
//            return;
//        }

//        if(er==2){
//            error_control("ERROR: Data could not be loaded");
//            return;
//        }
    }
    else if(this->ui->i_tool->currentIndex()==12){
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
    else if(this->ui->i_tool->currentIndex()==13){
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
    this->ui->dataset_lab->setText("Dataset: "+reference);
    this->ui->v_progress_datamanaging->setValue(100);
    this->ui->v_progress_datamanaging->setValue(0);
    this->ui->i_progress_datamanaging->setValue(100);
    this->ui->i_progress_datamanaging->setValue(0);

    QApplication::restoreOverrideCursor();
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
//    QApplication::setOverrideCursor(Qt::WaitCursor);
//    int current_type;
//    if(this->ui->i_detection_random->isChecked()){
//        string input="random_image";
//        Generacion gen;
//        gen.Random_Synthetic_Image(ui->Clases_aleatoria->value(),Size(500,500),ui->Varianza_2->value(),ui->Separacion_2->value(),imagen);
//        current_type=GRAY;
//    }
//    else{
//        imagen=cv::imread(t->text().toStdString());
//        if(imagen.empty()){
//            QMessageBox msgBox;
//            msgBox.setText("ERROR: No se ha podido abrir la imagen o la ruta es erronea");
//            msgBox.exec();
//            QApplication::restoreOverrideCursor();
//            return;
//        }
//        imagen.convertTo(imagen,CV_32F);
//        if(imagen.cols<ui->Vent_X->value() || imagen.rows<ui->Vent_Y->value()){
//            QMessageBox msgBox;
//            msgBox.setText("ERROR: El tamaño de la ventana es mayor que el de la imagen");
//            msgBox.exec();
//            QApplication::restoreOverrideCursor();
//            return;
//        }
//        current_type=RGB;
//    }
}

