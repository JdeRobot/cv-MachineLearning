#ifndef CLASIFICADOR_H
#define CLASIFICADOR_H

#include <opencv2/opencv.hpp>
#include "../Herramientas/auxiliares.h"
#include "../Herramientas/dimensionalidad.h"
#ifdef GUI
#include <ui_mainwindow.h>
#endif

namespace MLT {
enum{
    DISTANCIAS=0,
    GAUSSIANO=1,
    CASCADA_CLAS=2,
    HISTOGRAMA=3,
    KNN=4,
    NEURONAL=5,
    C_SVM=6,
    RTREES=7,
    DTREES=8,
    BOOSTING=9,
    EXP_MAX=11,
    MICLASIFICADOR=33,
    MULTICLASIFICADOR=100
};

    class Clasificador
    {
    public:
        Clasificador(){}

        virtual int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true){}
        virtual int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read){}
        virtual int Save_Data(){}
        virtual int Read_Data(){}


        int tipo_clasificador,numero_etiquetas,ventana_x,ventana_y,ventana_o_x,ventana_o_y,tipo_dato;
        string nombre;

#ifdef GUI
    int progreso;
    int total_progreso;
    int error;
    bool running;
#endif

    private:
        virtual void Entrenamiento(Mat trainingDataMat, Mat labelsMat){}
        virtual float Clasificacion(Mat Data){}
    };
}

#endif // CLASIFICADOR_H
