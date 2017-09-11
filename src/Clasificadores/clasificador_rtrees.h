/*
*
* Copyright 2014-2016 Ignacio San Roman Lana
*
* This file is part of OpenCV_ML_Tool
*
* OpenCV_ML_Tool is free software: you can redistribute it and/or
* modify it under the terms of the GNU General Public License as
* published by the Free Software Foundation, either version 3 of the
* License, or (at your option) any later version.
*
* OpenCV_ML_Tool is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with OpenCV_ML_Tool. If not, see http://www.gnu.org/licenses/.
*
* For those usages not covered by this license please contact with
* isanromanlana@gmail.com
*/

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
