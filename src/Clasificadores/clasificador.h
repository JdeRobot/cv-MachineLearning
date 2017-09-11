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

#ifndef CLASIFICADOR_H
#define CLASIFICADOR_H

#include <opencv2/opencv.hpp>
#include "Herramientas/auxiliares.h"
#include "Herramientas/dimensionalidad.h"
#ifdef GUI
#include <ui_mainwindow.h>
#endif

namespace MLT {
enum{
    DISTANCIAS=0,
    GAUSSIANO=1,
    CASCADA_CLAS=2,
    HISTOGRAMA=3,
    KNN=4,
    NEURONAL=5,
    C_SVM=6,
    RTREES=7,
    DTREES=8,
    BOOSTING=9,
    EXP_MAX=11,
    MICLASIFICADOR=33,
    MULTICLASIFICADOR=100
};

    class Clasificador
    {
    public:
        Clasificador(){}

        int virtual Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true){}
        int virtual Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read){}
        int virtual Save_Data(){}
        int virtual Read_Data(){}


        int tipo_clasificador,numero_etiquetas,ventana_x,ventana_y,ventana_o_x,ventana_o_y,tipo_dato;
        string nombre;

    #ifdef GUI
        int progreso;
        int max_progreso;
        int base_progreso;
        int total_progreso;

        Ui::MainWindow *window;
    #endif

    private:
        void virtual Entrenamiento(Mat trainingDataMat, Mat labelsMat){}
        float virtual Clasificacion(Mat Data){}
    };
}

#endif // CLASIFICADOR_H
