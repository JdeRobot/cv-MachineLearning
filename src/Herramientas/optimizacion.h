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

#ifndef OPTIMIZACION_H
#define OPTIMIZACION_H

#include <opencv2/opencv.hpp>
#include "analisis.h"
#include "auxiliares.h"
#include "../Clasificadores/multiclasificador.h"
#ifdef GUI
#include <ui_mainwindow.h>
#endif

using namespace std;
using namespace cv;

namespace MLT {
    class Optimizacion
    {
    public:

        Optimizacion();

        int Validation(vector<Mat> Datos, vector<float> Labels, int Porcentaje_validation, int id_clasificador, Clasificadores::Parametros parame, float &Error, Mat &Confusion, vector<Analisis::Ratios_data> &Ratios);
        int Validation(vector<Mat> Datos, vector<float> Labels, int Porcentaje_validation, vector<int> id_clasif, Clasificadores::Parametros parame, MultiClasificador::Multi_type multi, float &Error, Mat &Confusion, vector<Analisis::Ratios_data> &Ratios);
        int Cross_Validation(vector<Mat> Datos, vector<float> Labels, int Num_Folds, int Tam_Fold, int id_clasificador, Clasificadores::Parametros inicio, Clasificadores::Parametros fin, Clasificadores::Parametros salto, Clasificadores::Parametros &parametros, float &Error, cv::Mat &Confus);
        int Super_Cross_Validation(vector<Mat> Datos, vector<float> Labels, int Num_Folds, int Tam_Fold, vector<int> &id_clasificador, Clasificadores::Parametros inicio, Clasificadores::Parametros fin, Clasificadores::Parametros salto, Clasificadores::Parametros &parametros, float &Error, Mat &Confus);
        int Ratios_parametro(vector<Mat> Datos, vector<float> Labels, int porcentaje_validacion, string parametro, Clasificadores::Parametros inicio, Clasificadores::Parametros fin, Clasificadores::Parametros salto, vector<vector<Analisis::Ratios_data> > &Ratios);

#ifdef GUI
    int progreso;
    int total_progreso;
    int error;
    bool running;
#endif
    };
}

#endif // OPTIMIZACION_H
