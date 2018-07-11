#ifndef CLASIFICADOR_DTREES_H
#define CLASIFICADOR_DTREES_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Clasificador_DTrees: public Clasificador{
    public:
        Clasificador_DTrees(string Nombre="",int max_depth=INT_MAX, int min_sample_count=10, float regression_accuracy=0.01, bool use_surrogates=true, int max_categories=10, int cv_folds=1, bool use_1se_rule=true, bool truncate_pruned_tree=true, Mat priors=Mat());
        ~Clasificador_DTrees();
        int Parametrizar(int max_depth, int min_sample_count, float regression_accuracy, bool use_surrogates, int max_categories, int cv_folds, bool use_1se_rule, bool truncate_pruned_tree, Mat priors);
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        Ptr<ml::DTrees>  TREES;
        Dimensionalidad::Reducciones reduccion;
    };
}

#endif // CLASIFICADOR_DTREES_H
