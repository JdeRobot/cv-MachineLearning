#ifndef CLASIFICADOR_BOOSTING_H
#define CLASIFICADOR_BOOSTING_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Clasificador_Boosting: public Clasificador{
    public:
        Clasificador_Boosting(string Nombre="",int boost_type=ml::Boost::REAL, int weak_count=100, double weight_trim_rate=0.95, int max_depth=1, bool use_surrogates=false, Mat priors=Mat());
        ~Clasificador_Boosting();
        int Parametrizar(int boost_type, int weak_count, double weight_trim_rate, int max_depth, bool use_surrogates, Mat priors);
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        Ptr<ml::Boost>  BOOST;
        Dimensionalidad::Reducciones reduccion;
    };
}

#endif // CLASIFICADOR_BOOSTING_H
