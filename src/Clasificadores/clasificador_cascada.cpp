#include "clasificador_cascada.h"

MLT::Clasificador_Cascada::Clasificador_Cascada(string Nombre, string FeatureType, bool ejec_script, int NumPos, int NumNeg, string Mode, int NumStage,float MinHitRate, float MaxFalseAlarmRate, float WeightTrimRate, int MaxWeakCount, int MaxDepth, string Bt, int PrecalcValBufSize, int PrecalcidxBufSize){
    //    Command line arguments of opencv_traincascade application grouped by purposes:
    //    Common arguments:
    //    -data <cascade_dir_name>
    //    Where the trained classifier should be stored.
    //    -vec <vec_file_name>
    //    vec-file with positive samples (created by opencv_createsamples utility).
    //    -bg <background_file_name>
    //    Background description file.
    //    -numPos <number_of_positive_samples>
    //    -numNeg <number_of_negative_samples>
    //    Number of positive/negative samples used in training for every classifier stage.
    //    -numStages <number_of_stages>
    //    Number of cascade stages to be trained.
    //    -precalcValBufSize <precalculated_vals_buffer_size_in_Mb>
    //    Size of buffer for precalculated feature values (in Mb).
    //    -precalcIdxBufSize <precalculated_idxs_buffer_size_in_Mb>
    //    Size of buffer for precalculated feature indices (in Mb). The more memory you have the faster the training process.
    //    -baseFormatSave
    //    This argument is actual in case of Haar-like features. If it is specified, the cascade will be saved in the old format.
    //    Cascade parameters:
    //    -stageType <BOOST(default)>
    //    Type of stages. Only boosted classifier are supported as a stage type at the moment.
    //    -featureType<{HAAR(default), LBP}>
    //    Type of features: HAAR - Haar-like features, LBP - local binary patterns.
    //    -w <sampleWidth>
    //    -h <sampleHeight>
    //    Size of training samples (in pixels). Must have exactly the same values as used during training samples creation (opencv_createsamples utility).
    //    Boosted classifer parameters:
    //    -bt <{DAB, RAB, LB, GAB(default)}>
    //    Type of boosted classifiers: DAB - Discrete AdaBoost, RAB - Real AdaBoost, LB - LogitBoost, GAB - Gentle AdaBoost.
    //    -minHitRate <min_hit_rate>
    //    Minimal desired hit rate for each stage of the classifier. Overall hit rate may be estimated as (min_hit_rate^number_of_stages).
    //    -maxFalseAlarmRate <max_false_alarm_rate>
    //    Maximal desired false alarm rate for each stage of the classifier. Overall false alarm rate may be estimated as (max_false_alarm_rate^number_of_stages).
    //    -weightTrimRate <weight_trim_rate>
    //    Specifies whether trimming should be used and its weight. A decent choice is 0.95.
    //    -maxDepth <max_depth_of_weak_tree>
    //    Maximal depth of a weak tree. A decent choice is 1, that is case of stumps.
    //    -maxWeakCount <max_weak_tree_count>
    //    Maximal count of weak trees for every cascade stage. The boosted classifier (stage) will have so many weak trees (<=maxWeakCount), as needed to achieve the given -maxFalseAlarmRate.
    //    Haar-like feature parameters:
    //    -mode <BASIC (default) | CORE | ALL>
    //    Selects the type of Haar features set used in training. BASIC use only upright features, while ALL uses the full set of upright and 45 degree rotated feature set. See [Rainer2002] for more details.
    //    Local Binary Patterns parameters:
    //    Local Binary Patterns don’t have parameters.


    Parametrizar(FeatureType,ejec_script,NumPos,NumNeg,Mode,NumStage,MinHitRate,MaxFalseAlarmRate,WeightTrimRate,MaxWeakCount,MaxDepth,Bt,PrecalcValBufSize,PrecalcidxBufSize);
    nombre=Nombre;
    tipo_clasificador=CASCADA_CLAS;
}
MLT::Clasificador_Cascada::~Clasificador_Cascada(){}

int MLT::Clasificador_Cascada::Parametrizar(string FeatureType, bool ejec_script, int NumPos, int NumNeg, string Mode, int NumStage,float MinHitRate, float MaxFalseAlarmRate, float WeightTrimRate, int MaxWeakCount, int MaxDepth, string Bt, int PrecalcValBufSize, int PrecalcidxBufSize){
    ejecutar_script=ejec_script;
    featureType=FeatureType;
    numPos=NumPos;
    numNeg=NumNeg;
    mode=Mode;
    numStages=NumStage;
    minHitRate=MinHitRate;
    maxFalseAlarmRate=MaxFalseAlarmRate;
    weightTrimRate=WeightTrimRate;
    maxWeakCount=MaxWeakCount;
    maxDepth=MaxDepth;
    bt=Bt;
    precalcValBufSize=PrecalcValBufSize;
    precalcIdxBufSize=PrecalcidxBufSize;
    return 0;
}

int MLT::Clasificador_Cascada::Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save){
    int e=0;
    if(Data.size()==0){
        cout<<"ERROR en Autotrain: No hay datos"<<endl;
        return 1;
    }
    if(Labels.size()==0){
        cout<<"ERROR en Autotrain: No hay Etiquetas"<<endl;
        return 1;
    }
    if(Data.size()!=Labels.size()){
        cout<<"ERROR en Autotrain: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
    if(info.Tipo_Datos!=1){
        cout<<"ERROR en Autotrain: El tipo de dato no es correcto. El clasificador cascada sólo acepta datos de tipo GRAY"<<endl;
        return 1;
    }
    ventana_x=Data[0].cols;
    ventana_y=Data[0].rows;
    Auxiliares ax;
    bool negativa;
    numero_etiquetas=ax.numero_etiquetas(Labels,negativa);
    Mat lexic_labels(Labels.size(), 1, CV_32FC1, Labels.data());
    lexic_labels.convertTo(lexic_labels,CV_32FC1);
    Mat trainingDataMat;
    e=ax.Image2Lexic(Data,trainingDataMat);
    trainingDataMat.convertTo(trainingDataMat,CV_32FC1);
    if(e==1){
        cout<<"ERROR en Autorain: Error en Image2Lexic"<<endl;
        return 1;
    }
    Entrenamiento(trainingDataMat, lexic_labels);
    if(save){
        e=Save_Data();
        if(e==1){
            cout<<"ERROR en Autotrain: Error en Save_Data"<<endl;
            return 1;
        }
    }
    return 0;
}

int MLT::Clasificador_Cascada::Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read){
    this->running=true;
    int e=0;
    if(read){
        e=Read_Data();
        if(e==1){
            cout<<"ERROR en Autoclasificacion: Error en Read_Data"<<endl;
            this->running=false;
            return 1;
        }
    }
    for(uint i=0; i<Data.size(); i++){
        float response=Clasificacion(Data[i]);
        Labels.push_back(response);
#ifdef GUI
            progreso++;
//            window->progress_Clasificar->setValue(base_progreso+(max_progreso*progreso/total_progreso));
#endif
    }
    this->running=false;
    return 0;
}

void MLT::Clasificador_Cascada::Entrenamiento(Mat trainingDataMat, Mat labelsMat){
    string Sistema="CASCADE_SYSTEM_"+nombre+"/";
    string Positivos_directory=Sistema+"Positivos/";
    string Negativos_directory=Sistema+"Negativos/";
    string Parametros=Sistema+nombre+"_CASCADE/";

    DIR    *dir_p = opendir (Sistema.c_str());
    if(dir_p == NULL) {
        string command = "mkdir "+Sistema;
        int er=system(command.c_str());
        if(er==0)
            dir_p = opendir (Sistema.c_str());
    }

    DIR    *dir_p2 = opendir (Positivos_directory.c_str());
    if(dir_p2 == NULL) {
        string command = "mkdir "+Positivos_directory;
        int er=system(command.c_str());
        if(er==0)
            dir_p2 = opendir (Positivos_directory.c_str());
    }

    DIR    *dir_p3 = opendir (Negativos_directory.c_str());
    if(dir_p3 == NULL) {
        string command = "mkdir "+Negativos_directory;
        int er=system(command.c_str());
        if(er==0)
            dir_p3 = opendir (Negativos_directory.c_str());
    }

    DIR    *dir_p4 = opendir ("../Data/Configuracion");
    if(dir_p4 == NULL) {
        string command = "mkdir ../Data/Configuracion";
        int er=system(command.c_str());
        if(er!=0){
            cout<<"ERROR no se puede abrir ../Data/Configuracion"<<endl;
            return;
        }
    }

    DIR    *dir_p5 = opendir (Parametros.c_str());
    if(dir_p5 == NULL) {
        string command = "mkdir "+Parametros;
        int er=system(command.c_str());
        if(er!=0){
            cout<<"ERROR no se puede abrir ../Data/Configuracion"<<endl;
            return;
        }
    }

    string fichero_positivos=Sistema+"Positivos.txt";
    string fichero_negativos=Sistema+"Negativos.txt";
    ofstream f_positivos;
    ofstream f_negativos;
    f_positivos.open(fichero_positivos.c_str());
    f_negativos.open(fichero_negativos.c_str());
    Auxiliares ax;
    vector<Mat> imagenes;
    ax.Lexic2Image(trainingDataMat,Size(ventana_x,ventana_y),1,imagenes);
    vector<float> labels;
    for(int i=0; i<labelsMat.rows;i++)
        labels.push_back(labelsMat.at<float>(i,0));
    for(uint i=0; i<imagenes.size(); i++){
        if(labels[i]==1.0){
            ostringstream num;
            num<<i;
            string nombre=Positivos_directory+num.str()+".png";
            imwrite(nombre,imagenes[i]);
            f_positivos<<nombre.c_str();
            f_positivos<<" ";
            f_positivos<<1;
            f_positivos<<" ";
            f_positivos<<0;
            f_positivos<<" ";
            f_positivos<<0;
            f_positivos<<" ";
            f_positivos<<imagenes[i].cols-1;
            f_positivos<<" ";
            f_positivos<<imagenes[i].rows-1;
            f_positivos<<"\n";
        }
        else if(labels[i]==-1.0){
            ostringstream num;
            num<<i;
            string nombre=Negativos_directory+num.str()+".png";
            imwrite(nombre,imagenes[i]);
            f_negativos<<nombre;
            f_negativos<<"\n";
        }
    }
    f_positivos.close();
    f_negativos.close();

    string nombre_script="Entrenamiento.sh";
    string script=Sistema+nombre_script;
    ofstream f_script;
    f_script.open(script.c_str());

    f_script<<"#!/bin/bash \n";
    f_script<<"cd "+Sistema+"\n";
    f_script<<" opencv_createsamples -info Positivos.txt -vec Positivos.vec -w ";
    f_script<<imagenes[0].cols;
    f_script<<" -h ";
    f_script<<imagenes[0].rows;
    f_script<<" -num ";
    f_script<<999999999;
    f_script<<"\n";
    f_script<<"opencv_traincascade -vec Positivos.vec -bg Negativos.txt -data "+nombre+"_HAAR";
    f_script<<" -w ";
    f_script<<imagenes[0].cols;
    f_script<<" -h ";
    f_script<<imagenes[0].rows;
    f_script<<" -featureType ";
    f_script<<featureType;
    f_script<<" -numPos ";
    f_script<<numPos;
    f_script<<" -numNeg ";
    f_script<<numNeg;
    f_script<<" -mode ";
    f_script<<mode;
    f_script<<" -numStages ";
    f_script<<numStages;
    f_script<<" -minHitRate ";
    f_script<<minHitRate;
    f_script<<" -maxFalseAlarmRate ";
    f_script<<maxFalseAlarmRate;
    f_script<<" -weightTrimRate ";
    f_script<<weightTrimRate;
    f_script<<" -maxWeakCount ";
    f_script<<maxWeakCount;
    f_script<<" -maxDepth ";
    f_script<<maxDepth;
    f_script<<" -bt ";
    f_script<<bt;
    f_script<<" -precalcValBufSize ";
    f_script<<precalcValBufSize;
    f_script<<" -precalcIdxBufSize ";
    f_script<<precalcIdxBufSize;
    f_script<<"\n";
    f_script<<"mv "+Parametros+"cascade.xml ../Data/Configuracion/"+nombre+"/"+featureType+".xml";
    f_script.close();

    string command;
    command.clear();
    command="chmod +x "+script;
    system(command.c_str());
    command.clear();
    if(ejecutar_script){
        command="sh "+script;
        system(command.c_str());
    }
}

float MLT::Clasificador_Cascada::Clasificacion(Mat Data){
    Data.convertTo(Data,CV_8UC1);
    float response=0;
    if(Data.cols*Data.rows==(ventana_x*ventana_y)){
        vector<Rect> objects;
        Mat Data2;
        cv::resize(Data,Data2,Size(Data.rows*1.1,Data.cols*1.1));
        Cascade.detectMultiScale(Data2,objects,1.1,0);
        if(objects.size()>0)
            response=1.0;
        else
            response=-1.0;
    }
    return response;
}


int MLT::Clasificador_Cascada::Save_Data(){
    DIR    *dir_p = opendir ("../Data/Configuracion");
    if(dir_p == NULL) {
        string command = "mkdir ../Data/Configuracion";
        int er=system(command.c_str());
        if(er!=0){
            cout<<"ERROR en Read_Data: Error al crear carpeta"<<endl;
            return 1;
        }
    }
    String dir="../Data/Configuracion/"+nombre;
    DIR    *dir_p2 = opendir (dir.c_str());
    if(dir_p2 == NULL) {
        string command = "mkdir "+dir;
        int er=system(command.c_str());
        if(er!=0){
            cout<<"ERROR en Read_Data: Error al crear carpeta"<<endl;
            return 1;
        }
    }
    string g="../Data/Configuracion/"+nombre+"/HAAR2.xml";
    cv::FileStorage archivo_w(g,FileStorage::WRITE);
    if(archivo_w.isOpened()){
        archivo_w<<"ventana_x"<<ventana_x;
        archivo_w<<"ventana_y"<<ventana_y;
        archivo_w<<"numero_etiquetas"<<numero_etiquetas;
        archivo_w<<"tam_reduc"<<ventana_x*ventana_y;
        archivo_w<<"lda"<<false;
        archivo_w<<"LDA"<<Mat();
        archivo_w<<"pca"<<false;
        archivo_w<<"Pca"<<Mat();
        archivo_w<<"dist"<<false;
        archivo_w<<"DS"<<Mat();
    }
    else
        return 1;
    archivo_w.release();
    g="../Data/Configuracion/"+nombre+"/Clasificador.xml";
    cv::FileStorage clas(g,FileStorage::WRITE);
    if(clas.isOpened()){
        int id=CASCADA_CLAS;
        clas<<"Tipo"<<id;
    }
    else
        return 1;
    clas.release();
    return 0;
}

int MLT::Clasificador_Cascada::Read_Data(){
    string g="../Data/Configuracion/"+nombre+"/HAAR2.xml";
    cv::FileStorage archivo_r(g,FileStorage::READ);
    if(archivo_r.isOpened()){
        archivo_r["ventana_x"]>>ventana_x;
        archivo_r["ventana_y"]>>ventana_y;
        archivo_r["numero_etiquetas"]>>numero_etiquetas;
    }
    else
        return 1;
    archivo_r.release();
    g="../Data/Configuracion/"+nombre+"/HAAR.xml";
    Cascade.load(g);
    return 0;
}
