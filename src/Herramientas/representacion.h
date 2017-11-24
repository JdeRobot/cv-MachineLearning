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

#ifndef REPRESENTACION_H
#define REPRESENTACION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "auxiliares.h"
#include "analisis.h"
#ifdef GUI
#include <QApplication>
#endif

using namespace std;
using namespace cv;

namespace MLT {
class Representacion
    {
    public:
        Representacion();

        int Color(Mat Result_Etiq, vector<cv::Scalar> Colores, Mat &Colored, bool Show);
        int Recuadros(Mat imagen, vector<RotatedRect> recuadros, vector<float> labels_recuadros, vector<Scalar> Colores, Mat &salida, bool show);
        int Data_represent(string nombre,vector<Mat> Data, vector<float> labels, vector<int> dimensions, vector<cv::Scalar> Colores);
        int Ellipse_represent(string nombre,vector<Mat> Data, vector<float> labels, vector<int> dimensions, vector<cv::Scalar> Colores);
        int Data_Ellipse_represent(string nombre,vector<Mat> Data, vector<float> labels, vector<int> dimensions, vector<cv::Scalar> Colores);
        int Continuous_data_represent(string nombre,Mat Data, vector<float> labels, vector<cv::Scalar> Colores);
        int Histogram_represent(string nombre, vector<vector<Mat> > Histograma, vector<cv::Scalar> Colores, int dimension);
        int Imagen(vector<Mat> Imagenes, int numero);
    };
}

#endif // REPRESENTACION_H
