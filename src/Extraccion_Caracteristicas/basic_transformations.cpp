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

#include "basic_transformations.h"

MLT::Basic_Transformations::Basic_Transformations(int input_type, int output_type){
    input=input_type;
    output=output_type;
}
MLT::Basic_Transformations::~Basic_Transformations(){}

int MLT::Basic_Transformations::Extract(vector<Mat> Imagenes, vector<Mat> &Descriptores){
    for(uint i=0; i<Imagenes.size(); i++){
        Mat img_out;
        if(input==RGB && output==RGB)
            Imagenes[i].copyTo(img_out);
        else if(input==RGB && output==GRAY)
            cvtColor(Imagenes[i],img_out,CV_BGR2GRAY);
        else if(input==RGB && output==THRESHOLD){
            Mat gray,img8u;
            Imagenes[i].convertTo(img8u,CV_8U);
            cvtColor(img8u,gray,CV_BGR2GRAY);
            cv::threshold(gray, img_out, 0, 1, CV_THRESH_BINARY | CV_THRESH_OTSU);
        }
        else if(input==RGB && output==CANNY){
            Mat gray;
            cvtColor(Imagenes[i],gray,CV_BGR2GRAY);
            Mat aux,img8u;
            gray.convertTo(img8u,CV_8U);
            int level=cv::threshold(img8u, aux, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            cv::Canny(img8u,img_out,0.5*level,level);
        }
        else if(input==RGB && output==SOBEL){
            Mat gray;
            cvtColor(Imagenes[i],gray,CV_BGR2GRAY);
            Mat img_out_a,img_out_b;
            cv::Sobel(gray,img_out_a,CV_32F,1,0);
            cv::Sobel(gray,img_out_b,CV_32F,0,1);
            convertScaleAbs(img_out_a+img_out_b,img_out);
        }
        else if(input==RGB && output==HSV)
            cvtColor(Imagenes[i],img_out,CV_BGR2HSV);
        else if(input==RGB && output==H_CHANNEL){
            Mat img_out_hsv;
            cvtColor(Imagenes[i],img_out_hsv,CV_BGR2HSV);
            vector<Mat> channels;
            split(img_out_hsv,channels);
            channels[0].copyTo(img_out);
        }
        else if(input==RGB && output==S_CHANNEL){
            Mat img_out_hsv;
            cvtColor(Imagenes[i],img_out_hsv,CV_BGR2HSV);
            vector<Mat> channels;
            split(img_out_hsv,channels);
            channels[1].copyTo(img_out);
        }
        else if(input==RGB && output==V_CHANNEL){
            Mat img_out_hsv;
            cvtColor(Imagenes[i],img_out_hsv,CV_BGR2HSV);
            vector<Mat> channels;
            split(img_out_hsv,channels);
            channels[2].copyTo(img_out);
        }
        else if(input==RGB && output==COLOR_PREDOMINANTE){
            Mat img_out_hsv;
            cvtColor(Imagenes[i],img_out_hsv,CV_BGR2HSV);
            img_out=Mat::zeros(img_out_hsv.rows,img_out_hsv.cols,CV_32FC1);
            for(int y=0;y<img_out_hsv.rows;y++){
                for(int x=0; x<img_out_hsv.cols;x++){
                    Vec3b pixel = img_out_hsv.at<Vec3b>(y, x);
                    if(pixel[0]>=0. && pixel[0]<30.)
                        img_out.at<float>(y,x)=1;
                    else if(pixel[0]>=30. && pixel[0]<60.)
                        img_out.at<float>(y,x)=2;
                    else if(pixel[0]>=60. && pixel[0]<90.)
                        img_out.at<float>(y,x)=3;
                    else if(pixel[0]>=90. && pixel[0]<120.)
                        img_out.at<float>(y,x)=4;
                    else if(pixel[0]>=120. && pixel[0]<150.)
                        img_out.at<float>(y,x)=5;
                    else if(pixel[0]>=150. && pixel[0]<180.)
                        img_out.at<float>(y,x)=6;
                }
            }
        }
        else if(input==HSV && output==RGB)
            cvtColor(Imagenes[i],img_out,CV_HSV2BGR);
        else if(input==HSV && output==GRAY){
            vector<Mat> channels;
            split(Imagenes[i],channels);
            channels[2].copyTo(img_out);
        }
        else if(input==HSV && output==HSV)
            Imagenes[i].copyTo(img_out);
        else if(input==HSV && output==THRESHOLD){
            Mat gray;
            vector<Mat> channels;
            split(Imagenes[i],channels);
            channels[2].copyTo(gray);
            Mat img8u;
            gray.convertTo(img8u,CV_8U);
            cv::threshold(img8u, img_out, 0, 1, CV_THRESH_BINARY | CV_THRESH_OTSU);
        }
        else if(input==HSV && output==CANNY){
            Mat gray;
            vector<Mat> channels;
            split(Imagenes[i],channels);
            channels[2].copyTo(gray);
            Mat aux,img8u;
            Imagenes[i].convertTo(img8u,CV_8U);
            int level=cv::threshold(img8u, aux, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            cv::Canny(img8u,img_out,0.5*level,level);
        }
        else if(input==HSV && output==SOBEL){
            Mat gray;
            vector<Mat> channels;
            split(Imagenes[i],channels);
            channels[2].copyTo(gray);
            Mat img_out_a,img_out_b;
            cv::Sobel(gray,img_out_a,CV_32F,1,0);
            cv::Sobel(gray,img_out_b,CV_32F,0,1);
            convertScaleAbs(img_out_a+img_out_b,img_out);
        }
        else if(input==HSV && output==H_CHANNEL){
            vector<Mat> channels;
            split(Imagenes[i],channels);
            channels[0].copyTo(img_out);
        }
        else if(input==HSV && output==S_CHANNEL){
            vector<Mat> channels;
            split(Imagenes[i],channels);
            channels[1].copyTo(img_out);
        }
        else if(input==HSV && output==V_CHANNEL){
            vector<Mat> channels;
            split(Imagenes[i],channels);
            channels[2].copyTo(img_out);
        }
        else if(input==HSV && output==COLOR_PREDOMINANTE){
            img_out=Mat::zeros(Imagenes[i].rows,Imagenes[i].cols,CV_32FC1);
            for(int y=0;y<Imagenes[i].rows;y++){
                for(int x=0; x<Imagenes[i].cols;x++){
                    Vec3b pixel = Imagenes[i].at<Vec3b>(y, x);
                    if(pixel[0]>=0. && pixel[0]<30.)
                        img_out.at<float>(y,x)=1;
                    else if(pixel[0]>=30. && pixel[0]<60.)
                        img_out.at<float>(y,x)=2;
                    else if(pixel[0]>=60. && pixel[0]<90.)
                        img_out.at<float>(y,x)=3;
                    else if(pixel[0]>=90. && pixel[0]<120.)
                        img_out.at<float>(y,x)=4;
                    else if(pixel[0]>=120. && pixel[0]<150.)
                        img_out.at<float>(y,x)=5;
                    else if(pixel[0]>=150. && pixel[0]<180.)
                        img_out.at<float>(y,x)=6;
                }
            }
        }
        else if(input==GRAY && output==GRAY)
            Imagenes[i].copyTo(img_out);
        else if(input==GRAY && output==THRESHOLD){
            Mat img8u;
            Imagenes[i].convertTo(img8u,CV_8U);
            cv::threshold(img8u, img_out, 0, 1, CV_THRESH_BINARY | CV_THRESH_OTSU);
        }
        else if(input==GRAY && output==CANNY){
            Mat aux,img8u;
            Imagenes[i].convertTo(img8u,CV_8U);
            int level=cv::threshold(img8u, aux, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            cv::Canny(img8u,img_out,0.5*level,level);
        }
        else if(input==GRAY && output==SOBEL){
            Mat img_out_a,img_out_b;
            cv::Sobel(Imagenes[i],img_out_a,CV_32F,1,0);
            cv::Sobel(Imagenes[i],img_out_b,CV_32F,0,1);
            convertScaleAbs(img_out_a+img_out_b,img_out);
        }
        else if(input==GRAY && output==V_CHANNEL)
            Imagenes[i].copyTo(img_out);
        else if(input==H_CHANNEL && output==COLOR_PREDOMINANTE){
            img_out=Mat::zeros(Imagenes[i].rows,Imagenes[i].cols,CV_32FC1);
            for(int y=0;y<Imagenes[i].rows;y++){
                for(int x=0; x<Imagenes[i].cols;x++){
                    Vec3b pixel = Imagenes[i].at<Vec3b>(y, x);
                    if(pixel[0]>=0. && pixel[0]<30.)
                        img_out.at<float>(y,x)=1;
                    else if(pixel[0]>=30. && pixel[0]<60.)
                        img_out.at<float>(y,x)=2;
                    else if(pixel[0]>=60. && pixel[0]<90.)
                        img_out.at<float>(y,x)=3;
                    else if(pixel[0]>=90. && pixel[0]<120.)
                        img_out.at<float>(y,x)=4;
                    else if(pixel[0]>=120. && pixel[0]<150.)
                        img_out.at<float>(y,x)=5;
                    else if(pixel[0]>=150. && pixel[0]<180.)
                        img_out.at<float>(y,x)=6;
                }
            }
        }
        else{
            cout<<"ERROR en Extract: El tipo de transformacion indicado no esta contemplado"<<endl;
            return 1;
        }
        img_out.convertTo(img_out,CV_32F);
        Descriptores.push_back(img_out);
    }
    return 0;
}

