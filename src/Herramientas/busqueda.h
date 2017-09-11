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

#ifndef BUSQUEDA_H
#define BUSQUEDA_H

#include <opencv2/opencv.hpp>
#include <Herramientas/auxiliares.h>
#include <Herramientas/dimensionalidad.h>
#include <Clasificadores/Clasificadores.h>
#include <Clasificadores/multiclasificador.h>
#include "Extraccion_Caracteristicas/Caracteristicas.h"
#include "Clasificadores/miclasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Busqueda
    {
    public:
        Busqueda(Clasificador *clasificador,int Tipo_Descriptor, Descriptor *descriptor, MultiClasificador::Multi_type *multitipo=0);
        Busqueda(MultiClasificador *clasificador,int Tipo_Descriptor, Descriptor *descriptor, MultiClasificador::Multi_type *multitipo=0);
        int Textura(Mat src, Size tam_base, int escalas, int salto, int rotate, bool relleno, Mat &OUT);
        int Posicion(Mat src, Size tam_base, int escalas, int salto, int rotate, bool juntar_recuadros, bool solapamiento, bool aislamiento, float distancia_recuadros, int rotacion_recuadros, vector<RotatedRect> &recuadros, vector<float> &Labels);
        Descriptor *descrip;

    private:
        Clasificador *clasificador;
        MultiClasificador *Multi;

        int tipo,tipo_dato;
        int numero_etiquetas,ventana_x,ventana_y;

        MultiClasificador::Multi_type Tipo_Multi;
    };
}

#endif // BUSQUEDA_H
