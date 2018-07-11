/**
  @file miclasificador.cpp
  @brief Clase para un clasificador personalizado
  Contiene las funciones necesarias para parametrizar, y clasificar con un clasificador personalizado que puede ser una combinación
  de los clasificadores que se encuentran en la librería.
  Sólo funciona con el tipo de dato IMAGEN sin ningún tipo de transformación, si se quiere utilizar otro tipo de dato
 se debe transformar dentro del código de clasificación. Por tanto no admite datos de descriptores ni reducidos.
  Se debe generar un archivo MiClasificador.xml que debe encontrarse dentro de una carpeta con el nombre del clasificador,
 y que a su vez debe encontrarse dentro de la donde se encuentran los datos de los clasificadoes entenados (Data/Configuracion)
en el que se debe poner el tamaño de las muestras (ventana_o_x ventana_o_y, y el numero de etiquetas
  que tienen los datos.
  MiClasificador no tiene funciones de entrenamiento por lo que no se puede optimizar
*/

#include "miclasificador.h"
#include "../Extraccion_Caracteristicas/hog.h"


MLT::MiClasificador::MiClasificador(string Nombre){
    nombre=Nombre;
    //Tipo de dato con el que trabaja el clasificador
    tipo_dato=RGB;
}

MLT::MiClasificador::~MiClasificador(){
}

/**
  @brief Clasifica los datos
  @param Data: Datos de entrada
  @param Labels: Etiquetas de los datos
  @return Control de errores (0=OK)
*/
int MLT::MiClasificador::Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read){
    int e=0;
    Auxiliares ax;
    Mat lexic_data;
    e=ax.Image2Lexic(Data,lexic_data);
    if(e==1){
        cout<<"ERROR en Autoclasificacion: Error en Image2Lexic"<<endl;
        return 1;
    }
    tam_imagen=Data[0].size();
    Mat trainingDataMat;
    lexic_data.copyTo(trainingDataMat);
    for(int i=0; i<trainingDataMat.rows; i++){
#ifdef GUI
        progreso++;

/*******************************************************************************************/
        //Tu codigo para la barra de cargado de la GUI
        //Ejemplo:
        //clasificador.progreso=progreso;
        //clasificador.max_progreso=max_progreso;
        //clasificador.base_progreso=base_progreso;
        //clasificador.total_progreso=total_progreso;
        //clasificador.window=window;
/*****************************************************************************************/
#endif
        float response=Clasificacion(trainingDataMat.row(i));
        Labels.push_back(response);
    }
    return 0;
}

float MLT::MiClasificador::Clasificacion(Mat Data){
    Data.convertTo(Data,CV_32FC1);
    Auxiliares aux;
    vector<Mat> Datos;
    int n_channels=-1;
    if(tipo_dato==RGB)
        n_channels=3;
    else if(tipo_dato==GRAY || tipo_dato==H_CHANNEL || tipo_dato==S_CHANNEL || tipo_dato==V_CHANNEL)
        n_channels=1;
    aux.Lexic2Image(Data,tam_imagen,n_channels,Datos);
    float response=0;

/*******************************************************************************************/
    //Tu codigo de clasificacion con las reglas necesarias
    //Ejemplo:
    //vector<float> label;
    //clasificador.Autoclasificación(Datos,labels,true,false);
    //response=label[0];
/*****************************************************************************************/

    return response;
}

int MLT::MiClasificador::Read_Data(){
    string g="../Data/Configuracion/"+nombre+"/MiClasificador.xml";
    cv::FileStorage archivo_r(g,CV_STORAGE_READ);
    if(archivo_r.isOpened()){
        archivo_r["ventana_o_x"]>>ventana_o_x;
        archivo_r["ventana_o_y"]>>ventana_o_y;
        archivo_r["numero_etiquetas"]>>numero_etiquetas;
    }
    else
        return 1;

/*******************************************************************************************/
    //Tu codigo para cargar los clasificadores
    // Ejemplo:
    // clasificador.nombre=NombreClasificador;
    // clasificador.Read_Data();
/*****************************************************************************************/

    archivo_r.release();
    return 0;
}

int MLT::MiClasificador::Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save){}
int MLT::MiClasificador::Save_Data(){}
void MLT::MiClasificador::Entrenamiento(Mat trainingDataMat, Mat labelsMat){}
