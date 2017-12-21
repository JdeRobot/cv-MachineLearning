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

        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true) override;
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read) override;
        int SaveData() override;
        int ReadData() override;

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat) override;
        float Clasificacion(Mat Data) override;

        Ptr<ml::Boost> BOOST;
        Dimensionalidad::Reducciones reduccion;
    };
}

#endif // CLASIFICADOR_BOOSTING_H
