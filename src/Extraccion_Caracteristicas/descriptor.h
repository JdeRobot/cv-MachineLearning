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

#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include <iostream>
#include <opencv2/opencv.hpp>

namespace MLT
{
    enum
    {
        RGB                     = 0,
        GRAY                    = 1,
        HOG_DES                 = 2,
        PUNTOS_CARACTERISTICOS  = 3,
        THRESHOLD               = 4,
        CANNY                   = 5,
        SOBEL                   = 6,
        HSV                     = 7,
        H_CHANNEL               = 8,
        S_CHANNEL               = 9,
        V_CHANNEL               = 10,
        COLOR_PREDOMINANTE      = 11
    };

    class Descriptor
    {
    public:
        Descriptor() { }
        virtual ~Descriptor() { }

        int virtual Extract(std::vector<cv::Mat> /*images*/, std::vector<cv::Mat>& /*descriptors*/) = 0;
    };
}

#endif // DESCRIPTOR_H
