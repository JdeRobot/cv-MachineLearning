#ifndef CLASIFICADOR_HISTOGRAMA_H
#define CLASIFICADOR_HISTOGRAMA_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Clasificador_Histograma: public Clasificador{
    public:
        Clasificador_Histograma(string Nombre="",float tam_celda=1);
        ~Clasificador_Histograma();
        int Parametrizar(float tam_celda);
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        Dimensionalidad::Reducciones reduccion;

        struct Histograma {
           Mat Datos;
           Mat Labels;
           double Tamano_Celda;
        } HIST;

        int pos_x, pos_y, tam_x, tam_y, p_x,p_y, tam_ag_x,tam_ag_y;
        cv::Mat frame, frame2;
        bool flag;
        bool negativa;
    };
}

#endif // CLASIFICADOR_HISTOGRAMA_H
