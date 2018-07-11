#ifndef CLASIFICADOR_GAUSSIANO_H
#define CLASIFICADOR_GAUSSIANO_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Clasificador_Gaussiano: public Clasificador{
    public:
        Clasificador_Gaussiano(string Nombre="");
        ~Clasificador_Gaussiano();
        int Parametrizar();
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        Ptr<ml::NormalBayesClassifier>  GAUSS;
        Dimensionalidad::Reducciones reduccion;
    };
}

#endif // CLASIFICADOR_GAUSSIANO_H
