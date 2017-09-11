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

bool myfunction (cv::KeyPoint A,cv::KeyPoint B){
    if(round(A.pt.y)!=round(B.pt.y)) return round(A.pt.y)<round(B.pt.y);
      else return round(A.pt.x)<round(B.pt.x);
    return false;
}

MLT::Puntos_Caracteristicos::Puntos_Caracteristicos(cv::String DetectorType, cv::String ExtractorType, float ParamDetector){
    detectorType=DetectorType;
    extractorType=ExtractorType;
    paramDetector=ParamDetector;
}

int MLT::Puntos_Caracteristicos::Extract(vector<cv::Mat> Images, vector<cv::Mat> &Descriptores){
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(paramDetector);
    Ptr<MSER> mser = MSER::create(paramDetector);
    Ptr<ORB> orb = ORB::create(paramDetector);
    Ptr<BRISK> brisk = BRISK::create(paramDetector);
    Ptr<KAZE> kaze = KAZE::create(paramDetector);
    Ptr<AKAZE> akaze = AKAZE::create(paramDetector);
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create((int)paramDetector);
    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create((int)paramDetector);
    for(uint k=0; k<Images.size(); k++){
        std::vector<cv::KeyPoint> keypoints;
        Mat Image, descriptors;
        if(Images[k].channels()==3)
            cvtColor(Images[k],Image,CV_BGR2GRAY);
        Image.convertTo(Image,CV_8UC1);
        if(detectorType=="FAST"){
            fast->detect(Image,keypoints);
        }
        else if(detectorType=="MSER"){
            mser->detect(Image,keypoints);
        }
        else if(detectorType=="ORB"){
            orb->detect(Image,keypoints);
        }
        else if(detectorType=="BRISK"){
            brisk->detect(Image,keypoints);
        }
        else if(detectorType=="KAZE"){
            kaze->detect(Image,keypoints);
        }
        else if(detectorType=="AZAKE"){
            akaze->detect(Image,keypoints);
        }
        else if(detectorType=="SIFT"){
            sift->detect(Image,keypoints);
        }
        else if(detectorType=="SURF"){
            surf->detect(Image,keypoints);
        }
        sort(keypoints.begin(), keypoints.end(), myfunction);
        if(extractorType=="ORB"){
            orb->compute( Image, keypoints, descriptors );
        }
        else if(extractorType=="BRISK"){
            brisk->compute( Image, keypoints, descriptors );
        }
        else if(extractorType=="KAZE"){
            kaze->compute( Image, keypoints, descriptors );
        }
        else if(extractorType=="AKAZE"){
            akaze->compute( Image, keypoints, descriptors );
        }
        else if(extractorType=="SIFT"){
            sift->compute( Image, keypoints, descriptors );
        }
        else if(extractorType=="SURF"){
            surf->compute( Image, keypoints, descriptors );
        }
        if(descriptors.empty()){
#ifdef WARNINGS
            cout<<"WARNING en Extract: Descriptores vacio"<<endl;
#endif
        }
        else{
            keypoints_I.push_back(keypoints);
            Mat descriptores_imagen=Mat::zeros(Image.cols*Image.rows, descriptors.cols, CV_32FC1);
            for(uint i=0; i<keypoints.size(); i++){
                int fila=(Image.cols*round(keypoints[i].pt.y))+round(keypoints[i].pt.x);
                descriptors.row(i).copyTo(descriptores_imagen.row(fila));
            }
            Descriptores.push_back(descriptores_imagen);
        }
    }
    if(Descriptores.empty()){
        cout<<"ERROR en Extract: Descriptores vacio"<<endl;
        return 1;
    }
    return 0;
}

void MLT::Puntos_Caracteristicos::Mostrar(vector<Mat> Images){
    for(uint i=0; i<Images.size(); i++){
        cv::Mat img_keypoints_I;
        cv::drawKeypoints(Images[i], keypoints_I[i], img_keypoints_I, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
        if(!img_keypoints_I.empty())
            cv::imshow("Keypoints Imagen", img_keypoints_I );
        cv::waitKey(0);
    }
}
