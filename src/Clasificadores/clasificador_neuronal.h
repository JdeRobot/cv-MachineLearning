#ifndef CLASIFICADOR_NEURONAL_H
#define CLASIFICADOR_NEURONAL_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Clasificador_Neuronal: public Clasificador{
    public:
        Clasificador_Neuronal(string Nombre="", Mat layerSize=Mat(), int Method=ml::ANN_MLP::RPROP, int Function=ml::ANN_MLP::SIGMOID_SYM, double bp_dw_scale=0.1, double bp_moment_scale=0.1, double rp_dw0=0.1, double rp_dw_max=50.0, double rp_dw_min=FLT_EPSILON, double rp_dw_minus=0.5, double rp_dw_plus=1.2, double fparam1=0, double fparam2=0);
        ~Clasificador_Neuronal();
        int Parametrizar(Mat layerSize, int Method, int Function, double bp_dw_scale, double bp_moment_scale, double rp_dw0, double rp_dw_max, double rp_dw_min, double rp_dw_minus, double rp_dw_plus, double fparam1, double fparam2);
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        Ptr<ml::ANN_MLP>  MLP;
        Dimensionalidad::Reducciones reduccion;
        Mat Layers;
        bool negativa;
    };
}

#endif // CLASIFICADOR_NEURONAL_H
