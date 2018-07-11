#include "clasificador_histograma.h"

MLT::Clasificador_Histograma::Clasificador_Histograma(string Nombre, float tam_celda){
    Parametrizar(tam_celda);
    nombre=Nombre;
    tipo_clasificador=HISTOGRAMA;
}

MLT::Clasificador_Histograma::~Clasificador_Histograma(){}

int MLT::Clasificador_Histograma::Parametrizar(float tam_celda){
    HIST.Tamano_Celda=tam_celda;
    return 0;
}

int MLT::Clasificador_Histograma::Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save){
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
    if((reduc.si_dist&&(reduc.si_pca||reduc.si_lda||reduc.si_d_prime))||
            (reduc.si_d_prime&&(reduc.si_pca||reduc.si_lda||reduc.si_dist))||
            (reduc.si_pca&&(reduc.si_dist||reduc.si_lda||reduc.si_d_prime))||
            (reduc.si_lda&&(reduc.si_pca||reduc.si_dist||reduc.si_d_prime))){
        cout<<"ERROR en Autotrain: Solo puede haber un metodo de reduccion de dimensionalidad activado"<<endl;
        return 1;
    }
    if((reduc.si_lda || reduc.si_pca || reduc.si_dist || reduc.si_d_prime) && reduc.tam_reduc<=0){
        cout<<"ERROR en Autotrain: si_lda=true o si_pca=true o si_dist=true o si_d_prime=true pero t_reduc es igual o menor a 0"<<endl;
        return 1;
    }
    ventana_o_x=info.Tam_Orig_X;
    ventana_o_y=info.Tam_Orig_Y;
    ventana_x=info.Tam_X;
    ventana_y=info.Tam_Y;
    tipo_dato=info.Tipo_Datos;
    if((reduc.si_dist==true || reduc.si_d_prime==true || reduc.si_lda==true || reduc.si_pca==true)&&(info.si_dist==true || info.si_d_prime==true || info.si_lda==true || info.si_pca==true)){
        cout<<"ERROR en Autotrain: Ya se le ha hecho una reduccion anteriormente a los datos"<<endl;
        return 1;
    }
    reduccion=reduc;
    Auxiliares ax;
    numero_etiquetas=ax.numero_etiquetas(Labels,negativa);
    Mat lexic_data;
    int e=ax.Image2Lexic(Data,lexic_data);
    if(e==1){
        cout<<"ERROR en Autorain: Error en Image2Lexic"<<endl;
        return 1;
    }
    Mat lexic_labels(Labels.size(), 1, CV_32FC1, Labels.data());
    lexic_data.convertTo(lexic_data,CV_32FC1);
    lexic_labels.convertTo(lexic_labels,CV_32FC1);
    Mat trainingDataMat;
    if(reduccion.si_lda){
        Dimensionalidad dim(nombre);
        e=dim.LDA_matriz(lexic_data,Labels,reduccion.tam_reduc,reduccion.LDA,save);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en LDA_matriz"<<endl;
            return 1;
        }
        Mat Proyectada;
        e=dim.Proyeccion(lexic_data,Proyectada,LDA_DIM,reduccion.LDA);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en Proyeccion"<<endl;
            return 1;
        }
        Proyectada.copyTo(trainingDataMat);
    }
    else if(reduccion.si_pca){
        Dimensionalidad dim(nombre);
        e=dim.PCA_matriz(lexic_data,reduccion.tam_reduc,reduccion.PCA,save);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en PCA_matriz"<<endl;
            return 1;
        }
        Mat Proyectada;
        e=dim.Proyeccion(lexic_data,Proyectada,PCA_DIM,reduccion.PCA);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en Proyeccion"<<endl;
            return 1;
        }
        Proyectada.copyTo(trainingDataMat);
    }
    else if(reduccion.si_dist){
        Dimensionalidad dim(nombre);
        e=dim.MaxDist_matriz(lexic_data,Labels,reduccion.tam_reduc,reduccion.DS,save);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en MaxDist_matriz"<<endl;
            return 1;
        }
        Mat Proyectada;
        e=dim.Proyeccion(lexic_data,Proyectada,MAXDIST_DIM,reduccion.DS);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en Proyeccion"<<endl;
            return 1;
        }
        Proyectada.copyTo(trainingDataMat);
    }
    else if(reduccion.si_d_prime){
        Dimensionalidad dim(nombre);
        e=dim.D_Prime_matriz(lexic_data,Labels,reduccion.tam_reduc,reduccion.D_PRIME,save);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en D_PRIME_matriz"<<endl;
            return 1;
        }
        Mat Proyectada;
        e=dim.Proyeccion(lexic_data,Proyectada,D_PRIME_DIM,reduccion.D_PRIME);
        if(e==1){
            cout<<"ERROR en Autotrain: Error en Proyeccion"<<endl;
            return 1;
        }
        Proyectada.copyTo(trainingDataMat);
    }
    else
        lexic_data.copyTo(trainingDataMat);
    Entrenamiento(trainingDataMat, lexic_labels);
    if(reduc.si_dist==false && reduc.si_d_prime==false && reduc.si_lda==false && reduc.si_pca==false){
        reduccion.si_dist=info.si_dist;
        reduccion.si_d_prime=info.si_d_prime;
        reduccion.si_lda=info.si_lda;
        reduccion.si_pca=info.si_pca;
        reduccion.DS=info.DS;
        reduccion.D_PRIME=info.D_PRIME;
        reduccion.LDA=info.LDA;
        reduccion.PCA=info.PCA;
        reduccion.tam_reduc=info.Tam_X*info.Tam_Y;
    }
    if(save){
        e=Save_Data();
        if(e==1){
            cout<<"ERROR en Autotrain: Error en Save_Data"<<endl;
            return 1;
        }
    }
    return 0;
}

int MLT::Clasificador_Histograma::Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read){
    int e=0;
    if(read){
        e=Read_Data();
        if(e==1){
            cout<<"ERROR en Autoclasificacion: Error en Read_Data"<<endl;
            return 1;
        }
    }
    Auxiliares ax;
    Mat lexic_data;
    e=ax.Image2Lexic(Data,lexic_data);
    if(e==1){
        cout<<"ERROR en Autoclasificacion: Error en Image2Lexic"<<endl;
        return 1;
    }
    Mat trainingDataMat;
    if(reducir){
        if(reduccion.si_lda){
            Dimensionalidad dim(nombre);
            Mat Proyectada;
            e=dim.Proyeccion(lexic_data,Proyectada,LDA_DIM,reduccion.LDA);
            if(e==1){
                cout<<"ERROR en Autoclasificacion: Error en Proyeccion"<<endl;
                return 1;
            }
            Proyectada.copyTo(trainingDataMat);
        }
        else if(reduccion.si_pca){
            Dimensionalidad dim(nombre);
            Mat Proyectada;
            e=dim.Proyeccion(lexic_data,Proyectada,PCA_DIM,reduccion.PCA);
            if(e==1){
                cout<<"ERROR en Autoclasificacion: Error en Proyeccion"<<endl;
                return 1;
            }
            Proyectada.copyTo(trainingDataMat);
        }
        else if(reduccion.si_dist){
            Dimensionalidad dim(nombre);
            Mat Proyectada;
            e=dim.Proyeccion(lexic_data,Proyectada,MAXDIST_DIM,reduccion.DS);
            if(e==1){
                cout<<"ERROR en Autoclasificacion: Error en Proyeccion"<<endl;
                return 1;
            }
            Proyectada.copyTo(trainingDataMat);
        }
        else if(reduccion.si_d_prime){
            Dimensionalidad dim(nombre);
            Mat Proyectada;
            e=dim.Proyeccion(lexic_data,Proyectada,D_PRIME_DIM,reduccion.D_PRIME);
            if(e==1){
                cout<<"ERROR en Autoclasificacion: Error en Proyeccion"<<endl;
                return 1;
            }
            Proyectada.copyTo(trainingDataMat);
        }
        else
            lexic_data.copyTo(trainingDataMat);
    }
    else
        lexic_data.copyTo(trainingDataMat);
    for(int i=0; i<trainingDataMat.rows; i++){
        float response=Clasificacion(trainingDataMat.row(i));
        Labels.push_back(response);
#ifdef GUI
            progreso++;
//            window->progress_Clasificar->setValue(base_progreso+(max_progreso*progreso/total_progreso));
#endif
    }
    return 0;
}

void MLT::Clasificador_Histograma::Entrenamiento(Mat trainingDataMat, Mat labelsMat){
    trainingDataMat.convertTo(trainingDataMat,CV_32FC1);
    labelsMat.convertTo(labelsMat,CV_32FC1);
    trainingDataMat.copyTo(HIST.Datos);
    labelsMat.copyTo(HIST.Labels);
}

float MLT::Clasificador_Histograma::Clasificacion(Mat Data){
    Data.convertTo(Data,CV_32FC1);
    float response=0;
    bool etiqueta_encontrada=false;
    if(Data.cols==(ventana_x*ventana_y) || Data.cols==reduccion.tam_reduc){
        Mat celda_dato(Data.rows,Data.cols,CV_32FC1);
        celda_dato.row(0)=Data.row(0)/HIST.Tamano_Celda;
        Mat celdas(HIST.Datos.rows,HIST.Datos.cols,CV_32FC1);
        celdas=HIST.Datos/HIST.Tamano_Celda;
        vector<float> votacion(numero_etiquetas);
        for(uint i=0; i<votacion.size(); i++)
            votacion[i]=0.0;
        for(int i=0; i< HIST.Datos.rows; i++){
            bool igual=true;
            for(int j=0; j<celdas.cols; j++){
                if(floor(celda_dato.at<float>(0,j))!=floor(celdas.at<float>(i,j))){
                    igual=false;
                }
            }
            if(igual){
                etiqueta_encontrada=true;
                float lab=HIST.Labels.at<float>(i,0);
                if(negativa && lab==-1)
                    votacion[lab+1]=votacion[lab+1]+1;
                else if(negativa)
                    votacion[lab]=votacion[lab]+1;
                else
                    votacion[lab-1]=votacion[lab-1]+1;
            }
        }
        if(etiqueta_encontrada){
            int max_vot=0,label=0;
            for(uint i=0; i<votacion.size(); i++){
                if(votacion[i]>max_vot){
                    max_vot=votacion[i];
                    label=i;
                }
            }
            if(negativa && label==0)
                response=label-1;
            else if(negativa)
                response=label;
            else
                response=label+1;
        }
        else{
            float min_distan=99999999;
            Mat celda_cercana;
            for(int i=0; i< HIST.Datos.rows; i++){
                float distan=0;
                for(int j=0; j<celdas.cols; j++){
                    distan=distan+abs(floor(celda_dato.at<float>(0,j))-floor(celdas.at<float>(i,j)));
                }
                if(distan<min_distan){
                    min_distan=distan;
                    celdas.row(i).copyTo(celda_cercana);
                }
            }
            for(int i=0; i< HIST.Datos.rows; i++){
                bool igual=true;
                for(int j=0; j<celdas.cols; j++){
                    if(floor(celda_cercana.at<float>(0,j))!=floor(celdas.at<float>(i,j))){
                        igual=false;
                    }
                }
                if(igual){
                    float lab=HIST.Labels.at<float>(i,0);
                    if(negativa && lab==-1)
                        votacion[lab+1]=votacion[lab+1]+1;
                    else if(negativa)
                        votacion[lab]=votacion[lab]+1;
                    else
                        votacion[lab-1]=votacion[lab-1]+1;
                }
            }
            int max_vot=0,label=0;
            for(uint i=0; i<votacion.size(); i++){
                if(votacion[i]>max_vot){
                    max_vot=votacion[i];
                    label=i;
                }
            }
            if(negativa && label==0)
                response=label-1;
            else if(negativa)
                response=label;
            else
                response=label+1;
        }
    }
    return response;
}

int MLT::Clasificador_Histograma::Save_Data(){
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
    string g="../Data/Configuracion/"+nombre+"/HISTOGRAMA2.xml";
    cv::FileStorage archivo_w(g,CV_STORAGE_WRITE);
    if(archivo_w.isOpened()){
        archivo_w<<"ventana_x"<<ventana_x;
        archivo_w<<"ventana_y"<<ventana_y;
        archivo_w<<"ventana_o_x"<<ventana_o_x;
        archivo_w<<"ventana_o_y"<<ventana_o_y;
        archivo_w<<"numero_etiquetas"<<numero_etiquetas;
        archivo_w<<"tipo_dato"<<tipo_dato;
        archivo_w<<"tam_reduc"<<reduccion.tam_reduc;
        archivo_w<<"lda"<<reduccion.si_lda;
        archivo_w<<"LDA"<<reduccion.LDA;
        archivo_w<<"pca"<<reduccion.si_pca;
        archivo_w<<"Pca"<<reduccion.PCA;
        archivo_w<<"dist"<<reduccion.si_dist;
        archivo_w<<"DS"<<reduccion.DS;
        archivo_w<<"tamano_celda"<<HIST.Tamano_Celda;
    }
    else
        return 1;
    archivo_w.release();
    g="../Data/Configuracion/"+nombre+"/HISTOGRAMA.xml";
    cv::FileStorage archivo_w2(g,CV_STORAGE_WRITE);
    if(archivo_w2.isOpened()){
        archivo_w2<<"Datos"<<HIST.Datos;
        archivo_w2<<"Etiquetas"<<HIST.Labels;
    }
    else
        return 1;
    archivo_w2.release();
    g="../Data/Configuracion/"+nombre+"/Clasificador.xml";
    cv::FileStorage clas(g,CV_STORAGE_WRITE);
    if(clas.isOpened()){
        int id=HISTOGRAMA;
        clas<<"Tipo"<<id;
    }
    else
        return 1;
    clas.release();
    return 0;
}

int MLT::Clasificador_Histograma::Read_Data(){
    string g="../Data/Configuracion/"+nombre+"/HISTOGRAMA2.xml";
    cv::FileStorage archivo_r(g,CV_STORAGE_READ);
    cv::Mat trainingDataMat,labelsMat;
    if(archivo_r.isOpened()){
        archivo_r["ventana_x"]>>ventana_x;
        archivo_r["ventana_y"]>>ventana_y;
        archivo_r["ventana_o_x"]>>ventana_o_x;
        archivo_r["ventana_o_y"]>>ventana_o_y;
        archivo_r["numero_etiquetas"]>>numero_etiquetas;
        archivo_r["tipo_dato"]>>tipo_dato;
        archivo_r["tam_reduc"]>>reduccion.tam_reduc;
        archivo_r["lda"]>>reduccion.si_lda;
        archivo_r["LDA"]>>reduccion.LDA;
        archivo_r["pca"]>>reduccion.si_pca;
        archivo_r["Pca"]>>reduccion.PCA;
        archivo_r["dist"]>>reduccion.si_dist;
        archivo_r["DS"]>>reduccion.DS;
        archivo_r["tamano_celda"]>>HIST.Tamano_Celda;
    }
    else
        return 1;
    archivo_r.release();
    g="../Data/Configuracion/"+nombre+"/HISTOGRAMA.xml";
    cv::FileStorage archivo_r2(g,CV_STORAGE_READ);
    if(archivo_r2.isOpened()){
        archivo_r2["Datos"]>>HIST.Datos;
        archivo_r2["Etiquetas"]>>HIST.Labels;
    }
    else
        return 1;
    archivo_r2.release();
    return 0;
}
