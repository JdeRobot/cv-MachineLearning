#ifndef CLASIFICADOR_SVM_H
#define CLASIFICADOR_SVM_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    enum{
        TRAIN=0,
        TRAIN_AUTO=1
    };

    class Clasificador_SVM: public Clasificador{
    public:
        Clasificador_SVM(string Nombre="",int train=TRAIN_AUTO, int Type=ml::SVM::NU_SVC, int kernel_type=ml::SVM::RBF, Mat class_weights=Mat(), double degree=0, double gamma=1, double coef0=0, double C=1, double nu=0.00001, double p=0);
        ~Clasificador_SVM();
        int Parametrizar(int train, int Type, int kernel_type, Mat class_weights, double degree, double gamma, double coef0, double C, double nu, double p);
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        Ptr<ml::SVM>  SVM;
        int tipo_entrena;
        Dimensionalidad::Reducciones reduccion;
    };
}

#endif // CLASIFICADOR_SVM_H
