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
