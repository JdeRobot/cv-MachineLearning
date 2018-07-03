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

#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <opencv2/opencv.hpp>
#include "auxiliares.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Clustering
    {
    public:
        Clustering();
        ~Clustering();

        int K_mean(vector<Mat> Data, int K, vector<float> &Labels, Mat &Centers, int attempts=5, int inicializacion=0);
        int Min_Max(vector<Mat> Data, float max_dist, vector<float> &Labels, Mat &Centers);
        int Distancias_Encadenadas(vector<Mat> Data, float max_dist, vector<float> &Labels, Mat &Centers);
        int Histograma(vector<Mat> Data, float tam_celda, vector<float> &Labels, Mat &Centers);
        int EXP_MAX(vector<Mat> Data, vector<float> &Labelsint, Mat &Centers, int nclusters=ml::EM::DEFAULT_NCLUSTERS, int covMatType=ml::EM::COV_MAT_DIAGONAL);

        int error;
    };
}

#endif // CLUSTERING_H
