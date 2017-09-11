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

#ifndef HOG_H
#define HOG_H

#include "Herramientas/auxiliares.h"
#include "descriptor.h"

using namespace std;
using namespace cv;

namespace MLT {
    class HOG: public Descriptor{
    public:
        HOG(Size win_size=Size(64, 128),Size block_stride=Size(8, 8),
            double win_sigma=-1,double threshold_L2hys=0.2, bool gamma_correction=true,
            int nlevels=HOGDescriptor::DEFAULT_NLEVELS);
        ~HOG();
        int Extract(vector<Mat> Imagenes, vector<Mat> &Descriptores);
        int Mostrar(vector<Mat> Images, int scaleimage, int scalelines);
    private:
        cv::HOGDescriptor Hog;
        vector<Mat> Valores;
    };
}

#endif // HOG_H
