#ifndef CLASIFICADOR_EM_H
#define CLASIFICADOR_EM_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Clasificador_EM: public Clasificador{
    public:
        Clasificador_EM(string Nombre="", int nclusters=ml::EM::DEFAULT_NCLUSTERS, int covMatType=ml::EM::COV_MAT_DIAGONAL);
        ~Clasificador_EM();
        int Parametrizar(int nclusters, int covMatType);
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

        int numero_etiquetas,ventana_x,ventana_y,ventana_o_x,ventana_o_y,tipo_dato;
        string nombre;

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        bool negativa;

        Ptr<ml::EM>  EXP_M;
        Dimensionalidad::Reducciones reduccion;
    };
}

#endif // CLASIFICADOR_EM_H
