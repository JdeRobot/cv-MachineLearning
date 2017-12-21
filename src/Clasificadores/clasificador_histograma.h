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

#ifndef CLASIFICADOR_HISTOGRAMA_H
#define CLASIFICADOR_HISTOGRAMA_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Clasificador_Histograma: public Clasificador{
    public:
        Clasificador_Histograma(string Nombre="",float tam_celda=1);
        ~Clasificador_Histograma();

        int Parametrizar(float tam_celda);

        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true) override;
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read) override;
        int SaveData() override;
        int ReadData() override;

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat) override;
        float Clasificacion(Mat Data) override;

        Dimensionalidad::Reducciones reduccion;

        struct Histograma {
           Mat Datos;
           Mat Labels;
           double Tamano_Celda;
        } HIST;

        int pos_x, pos_y, tam_x, tam_y, p_x,p_y, tam_ag_x,tam_ag_y;
        cv::Mat frame, frame2;
        bool flag;
        bool negativa;
    };
}

#endif // CLASIFICADOR_HISTOGRAMA_H
