#ifndef CLASIFICADOR_RTREES_H
#define CLASIFICADOR_RTREES_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Clasificador_RTrees: public Clasificador{
    public:
        Clasificador_RTrees(string Nombre="",int max_depth=5, int min_sample_count=10, float regression_accuracy=0, bool use_surrogates=false, int max_categories=10, int cv_folds=0, bool use_1se_rule=false, bool truncate_pruned_tree=false, Mat priors=Mat(),bool calc_var_importance=false, int native_vars=0);
        ~Clasificador_RTrees();
        int Parametrizar(int max_depth, int min_sample_count, float regression_accuracy, bool use_surrogates, int max_categories, int cv_folds, bool use_1se_rule, bool truncate_pruned_tree, Mat priors,bool calc_var_importance, int native_vars);
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        Ptr<ml::RTrees>  TREES;
        Dimensionalidad::Reducciones reduccion;
    };
}

#endif // CLASIFICADOR_RTREES_H
