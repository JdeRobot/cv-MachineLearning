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
//    this->ui->v_progress_clustering->setValue(0);
//    this->ui->v_progress_dimensionality->setValue(0);
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
        er=this->run.load_dataset(path, ref,this->LABELS,this->IMAGENES,this->info);
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
        int vector_size=this->ui->v_vectordimension->value();
        float width=this->ui->v_variance->value();
        float interclass=this->ui->v_interclassdistance->value();

        er=this->run.synthetic_data(name,num_classes,num_data_class,vector_size,width,interclass,this->IMAGENES,this->LABELS,this->info);
        if(er==1){
            error_control("ERROR: Data could not be created");
            return;
        }
    }
    else if(this->ui->v_tool->currentIndex()==3){
        er=this->run.save(ref,this->IMAGENES,this->resultado,this->info);
        if(er==1){
            error_control("ERROR: Data could not be created");
            return;
        }
        path="../Data/Imagenes/"+name;
        er=this->run.load_dataset(path, ref,this->LABELS,this->IMAGENES,this->info);
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
        er=this->run.join_data(ref,path);
        if(er==1){
            error_control("ERROR: Data could not be created");
            return;
        }
        path="../Data/Imagenes/"+name;
        er=this->run.load_dataset(path, ref,this->LABELS,this->IMAGENES,this->info);
        if(er==1){
            error_control("ERROR: The folder has not the expected structure");
            return;
        }

        if(er==2){
            error_control("ERROR: Data could not be loaded");
            return;
        }
    }


    this->ui->v_plotting_x->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    this->ui->v_plotting_y->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    this->ui->v_plotting_dimension->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
    this->ui->v_progress_datamanaging->setValue(100);
    this->ui->v_progress_datamanaging->setValue(0);

//            this->ui->Numero_Clases->setValue(aux.numero_etiquetas(LABELS,neg));
//            this->ui->Dim_X_4->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            this->ui->Dim_Y_4->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            this->ui->Dimension_4->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            ui->Num_dimensiones->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            ui->Num_dimensiones->setValue(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            ui->Dim_X_5->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            ui->Dim_Y_5->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            ui->Dimension_3->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            ui->Dim_X_6->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            ui->Dim_Y_6->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            ui->Dimension_5->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            ui->Dimension_graf->setMaximum(IMAGENES[0].cols*IMAGENES[0].rows*IMAGENES[0].channels());
//            ui->Tam_Folds->setValue(IMAGENES.size()/ui->Num_folds->value());
    this->Dat_Ref=ref;
    QString reference=QString::fromStdString(ref);
    this->ui->dataset_lab->setText("Dataset: "+reference);
    this->ui->v_progress_datamanaging->setValue(100);
    this->ui->v_progress_datamanaging->setValue(0);
    //                this->ui->progress_Clasificar->setValue(100);
    //                this->ui->progress_Clasificar->setValue(0);
    //                this->ui->progress_generar->setValue(100);
    //                this->ui->progress_generar->setValue(0);
    //                this->ui->progress_Cargar->setValue(100);
    //                this->ui->progress_Cargar->setValue(0);
    //                this->ui->progress_Clus->setValue(100);
    //                this->ui->progress_Clus->setValue(0);
    //                this->ui->progress_Dimensionalidad->setValue(100);
    //                this-> ui->progress_Dimensionalidad->setValue(0);

    QApplication::restoreOverrideCursor();
}

void MainWindow::on_v_analysis_analyse_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    this->ui->v_progress_Analysis->setValue(1);
    if(this->IMAGENES.empty()){
        error_control("ERROR: There is not data loaded");
        return;
    }

    QStandardItemModel *model=new QStandardItemModel(0,0);
    int er=0;
    if(this->ui->v_analysis_dataset->isChecked()){
        if(this->LABELS.empty()){
            error_control("ERROR: There is not labels loaded");
            return;
        }
        if(!this->IMAGENES[0].cols*this->IMAGENES[0].rows*this->IMAGENES[0].channels()>1024){
            QMessageBox msgBox;
            msgBox.setText("The number of dimensions is over 1024, so covariance is not calculated");
            msgBox.exec();
        }
        er=this->run.analyse_data(this->IMAGENES,this->LABELS,model);
    }
    else if(this->ui->v_analysis_results->isChecked()){
        if(this->resultado.empty()){
            error_control("ERROR: There is not labels loaded");
            return;
        }
        er=this->run.analyse_result(this->LABELS,this->resultado,model);
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
    std::vector<cv::Mat> images;
    std::vector<float> labels;
    if(this->ui->v_plotting_dataset->isChecked()){
        images=this->IMAGENES;
        labels=this->LABELS;
    }
    else if(this->ui->v_plotting_results->isChecked()){
        images=this->IMAGENES;
        labels=this->resultado;
    }

    if(images.empty()){
        error_control("ERROR: There is not data loaded");
        return;
    }
    if(labels.empty()){
        error_control("ERROR: There is not labels loaded");
        return;
    }
    int e=0;
    vector<int> dim;
    if(this->ui->v_plotting_dimension->value()>images[0].cols*images[0].rows*images[0].channels()){
        QMessageBox msgBox;
        msgBox.setText("ERROR: La dimension en Histograma esta fuera de rango");
        msgBox.exec();
        QApplication::restoreOverrideCursor();
        return;
    }
    if(this->ui->v_plotting_x->value()<=images[0].cols*images[0].rows*images[0].channels()){
        dim.push_back(this->ui->v_plotting_x->value());
        if(this->ui->v_plotting_y->value()>0 && ui->v_plotting_y->value()<=images[0].cols*images[0].rows*images[0].channels())
            dim.push_back(this->ui->v_plotting_y->value());
        else if(this->ui->v_plotting_y->value()==0){
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
    if(this->ui->v_plotting_data->isChecked()){
        e=rep.Data_represent("DATOS "+this->Dat_Ref,images,labels,dim,this->Col);
    }
    else if(this->ui->v_plotting_ellipses->isChecked()){
        e=rep.Ellipse_represent("ELIPSES "+this->Dat_Ref,images,labels,dim,this->Col);
    }
    else if(this->ui->v_plotting_dataellipeses->isChecked()){
        e=rep.Data_Ellipse_represent("DATOS CON ELIPSES "+this->Dat_Ref,images,labels,dim,this->Col);
    }
    else if(this->ui->v_plotting_histogram->isChecked()){
        Analisis an;
        vector<vector<Mat> > Histo;
        vector<vector<int> > pos_barras;
        an.Histograma(images,labels,this->num_bar,Histo,pos_barras);
        e=rep.Histogram_represent("HISTOGRAMA "+this->Dat_Ref,Histo,this->Col, this->ui->v_plotting_dimension->value());
    }
    if(e==1){
        QMessageBox msgBox;
        msgBox.setText("ERROR: No se han podido representar los datos");
        msgBox.exec();
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
    if(IMAGENES.empty()){
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

    run.clustering(this->IMAGENES,type,k,repetitions,max_dist,cell_size,this->resultado);

    QString reference=QString::fromStdString(ref);
    this->ui->results_lab->setText("Results: "+reference);

    QApplication::restoreOverrideCursor();

}
