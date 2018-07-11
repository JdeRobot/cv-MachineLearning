#ifndef CLASIFICADOR_DISTANCIAS_H
#define CLASIFICADOR_DISTANCIAS_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Clasificador_Distancias: public Clasificador{
    public:
        Clasificador_Distancias(string Nombre="");
        ~Clasificador_Distancias();
        int Parametrizar();
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

    private:

        struct Distancias {
           vector<Mat> Medias;
           vector<float> Etiquetas;
        } DIST;

        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        Dimensionalidad::Reducciones reduccion;
    };
}

#endif // CLASIFICADOR_DISTANCIAS_H
