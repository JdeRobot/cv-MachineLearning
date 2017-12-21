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

#ifndef MULTICLASIFICADOR_H
#define MULTICLASIFICADOR_H

#include <dirent.h>
#include "../Clasificadores/Clasificadores.h"


using namespace std;
using namespace cv;

namespace MLT {
    enum
    {
        CASCADA     = 0,
        VOTACION    = 1
    };

    enum
    {
        IGUAL       = 0,
        DISTINTO    = 1,
        MENOR       = 2,
        MAYOR       = 3
    };

    class MultiClasificador{
    public:
        struct Multi_type {
           int tipo;
           vector<int> tipo_regla;
           vector<float> label_ref;
           vector<float> w_clasif;
        };

        MultiClasificador(vector<Clasificador*> Clasificadores);
        int Cascada(vector<Mat> Data, vector<int> tipo_regla, vector<float> labels_ref, vector<float> &Labels);
        int Votacion(vector<Mat> Data, vector<float> w_clasif, vector<float> &Labels);

        int numero_etiquetas,ventana_x,ventana_y,tipo_dato;

    #ifdef GUI
        int progreso;
        int max_progreso;
        int base_progreso;
        int total_progreso;

        Ui::MainWindow *window;
    #endif

    private:
        vector<Clasificador*> clasificadores;
        vector<int> ventanas_x,ventanas_y,n_etiquetas,tipos_dato;

        bool Error;
    };
}

#endif // MULTICLASIFICADOR_H
