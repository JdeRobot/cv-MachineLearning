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

    #ifdef GUI
        int progreso;
        int max_progreso;
        int base_progreso;
        int total_progreso;

        Ui::MainWindow *window;
    #endif

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        bool negativa;

        Ptr<ml::EM>  EXP_M;
        Dimensionalidad::Reducciones reduccion;
    };
}

#endif // CLASIFICADOR_EM_H
