#ifndef CLASIFICADOR_KNN_H
#define CLASIFICADOR_KNN_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Clasificador_KNN: public Clasificador{
    public:
        Clasificador_KNN(string Nombre="",int k=1, bool regression=false);
        ~Clasificador_KNN();
        int Parametrizar(int k, bool regression);
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        Ptr<ml::KNearest>  KN;
        Dimensionalidad::Reducciones reduccion;
        struct Knn {
           Mat Datos;
           Mat Labels;
        } K;
        int k_neig;
        bool reg;
    };
}

#endif // CLASIFICADOR_KNN_H
