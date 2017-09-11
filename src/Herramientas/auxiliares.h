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

#ifndef AUXILIARES_H
#define AUXILIARES_H

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <iostream>

using namespace std;
using namespace cv;

namespace MLT {
class Auxiliares
    {
    public:
        Auxiliares();
        int numero_imagenes(String input_directory);
        int numero_etiquetas(vector<float> Labels, bool &negativa);
        int Image2Lexic(std::vector<cv::Mat> Imagen, Mat &Datos);
        int Lexic2Image(cv::Mat Datos, Size tam_imagen, int num_channels, std::vector<cv::Mat> &Imagen);
    };
}

#endif // AUXILIARES_H
