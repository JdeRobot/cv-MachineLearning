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

#include "puntos_caracteristicos.h"

bool sortFunction(cv::KeyPoint A, cv::KeyPoint B)
{
    if (round(A.pt.y) != round(B.pt.y))
        return round(A.pt.y) < round(B.pt.y);
    else
        return round(A.pt.x) < round(B.pt.x);

    return false;
}

MLT::Puntos_Caracteristicos::Puntos_Caracteristicos(cv::String detectorType, cv::String extractorType, float paramDetector)
{
    detectorType = detectorType;
    extractorType = extractorType;
    paramDetector = paramDetector;
}

int MLT::Puntos_Caracteristicos::Extract(vector<cv::Mat> images, vector<cv::Mat>& descriptores)
{
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(paramDetector);
    Ptr<MSER> mser = MSER::create(paramDetector);
    Ptr<ORB> orb = ORB::create(paramDetector);
    Ptr<BRISK> brisk = BRISK::create(paramDetector);
    Ptr<KAZE> kaze = KAZE::create(/*paramDetector*/);
    Ptr<AKAZE> akaze = AKAZE::create(/*paramDetector*/);
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create((int)paramDetector);
    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create((int)paramDetector);

    for(uint k = 0; k < images.size(); k++)
    {
        std::vector<cv::KeyPoint> keypoints;
        Mat image, descriptors;

        if(images[k].channels() == 3)
            cvtColor(images[k], image, COLOR_BGR2GRAY);

        image.convertTo(image,CV_8UC1);

        if(detectorType == "FAST")
            fast->detect(image, keypoints);
        else if(detectorType == "MSER")
            mser->detect(image, keypoints);
        else if(detectorType == "ORB")
            orb->detect(image, keypoints);
        else if(detectorType == "BRISK")
            brisk->detect(image,keypoints);
        else if(detectorType == "KAZE")
            kaze->detect(image,keypoints);
        else if(detectorType == "AZAKE")
            akaze->detect(image,keypoints);
        else if(detectorType == "SIFT")
            sift->detect(image,keypoints);
        else if(detectorType == "SURF")
            surf->detect(image,keypoints);

        sort(keypoints.begin(), keypoints.end(), sortFunction);

        if(extractorType == "ORB")
            orb->compute( image, keypoints, descriptors );
        else if(extractorType == "BRISK")
            brisk->compute( image, keypoints, descriptors );
        else if(extractorType == "KAZE")
            kaze->compute( image, keypoints, descriptors );
        else if(extractorType == "AKAZE")
            akaze->compute( image, keypoints, descriptors );
        else if(extractorType == "SIFT")
            sift->compute( image, keypoints, descriptors );
        else if(extractorType == "SURF")
            surf->compute( image, keypoints, descriptors );

        if(descriptors.empty())
        {
#ifdef WARNINGS
            cout << "WARNING en Extract: descriptores vacio" << endl;
#endif
        }
        else
        {
            keypointsI.push_back(keypoints);
            Mat descriptores_imagen = Mat::zeros(image.cols * image.rows, descriptors.cols, CV_32FC1);

            for(uint i = 0; i < keypoints.size(); i++)
            {
                int fila = (image.cols*round(keypoints[i].pt.y)) + round(keypoints[i].pt.x);
                descriptors.row(i).copyTo(descriptores_imagen.row(fila));
            }
            descriptores.push_back(descriptores_imagen);
        }
    }

    if(descriptores.empty())
    {
        cout << "ERROR en Extract: descriptores vacio" << endl;
        this->error=1;
        return this->error;
    }
    this->error=0;
    return this->error;
}

void MLT::Puntos_Caracteristicos::Mostrar(vector<Mat> images)
{
    for(uint i = 0; i < images.size(); i++)
    {
        cv::Mat imgKeypointsI;
        cv::drawKeypoints(images[i], keypointsI[i], imgKeypointsI, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

        if(!imgKeypointsI.empty())
            cv::imshow("Keypoints Imagen", imgKeypointsI);

        cv::waitKey(0);
    }
}
