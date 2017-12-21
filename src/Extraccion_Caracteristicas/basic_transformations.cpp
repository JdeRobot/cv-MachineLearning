/*
*
* Copyright 2014-2016 Ignacio San Roman Lana
*
* This file is part of OpenCV_ML_Tool
*
* OpenCV_ML_Tool is free software: you can redistribute it and/or
* modify it under the terms of the GNU General Public License as
* published by the Free Software Foundation, either version 3 of the
* License, or  (at your option) any later version.
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

MLT::Basic_Transformations::Basic_Transformations (int inputType, int outputType)
{
    input = inputType;
    output = outputType;
}

MLT::Basic_Transformations::~Basic_Transformations ()
{

}

int MLT::Basic_Transformations::Extract (vector<Mat> imagenes, vector<Mat> &descriptores)
{
    for (uint i = 0; i < imagenes.size (); i++){
        Mat img_out;

        Mat gray;
        Mat aux;
        Mat img8u;
        Mat imgOutHsv;
        Mat imgOutA;
        Mat imgOutB;

        int level = 0;

        vector<Mat> channels;

        switch (input)
        {
            case RGB:
            {
                switch (output)
                {
                case RGB:
                    imagenes[i].copyTo(img_out);
                    break;
                case GRAY:
                    cvtColor(imagenes[i], img_out, CV_BGR2GRAY);
                    break;
                case THRESHOLD:
                    imagenes[i].convertTo(img8u,CV_8U);
                    cvtColor(img8u, gray, CV_BGR2GRAY);
                    cv::threshold(gray, img_out, 0, 1, CV_THRESH_BINARY | CV_THRESH_OTSU);
                    break;
                case CANNY:
                    cvtColor(imagenes[i], gray, CV_BGR2GRAY);
                    gray.convertTo(img8u,CV_8U);
                    level = cv::threshold (img8u, aux, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
                    cv::Canny(img8u, img_out, 0.5 * level, level);
                    break;
                case SOBEL:
                    cvtColor(imagenes[i], gray, CV_BGR2GRAY);
                    cv::Sobel(gray, imgOutA, CV_32F, 1, 0);
                    cv::Sobel(gray, imgOutB, CV_32F, 0, 1);
                    convertScaleAbs(imgOutA + imgOutB, img_out);
                    break;
                case HSV:
                    cvtColor(imagenes[i], img_out, CV_BGR2HSV);
                    break;
                case H_CHANNEL:
                    cvtColor(imagenes[i], imgOutHsv, CV_BGR2HSV);
                    split(imgOutHsv, channels);
                    channels[0].copyTo(img_out);
                    break;
                case S_CHANNEL:
                    cvtColor(imagenes[i], imgOutHsv, CV_BGR2HSV);
                    split(imgOutHsv, channels);
                    channels[1].copyTo(img_out);
                    break;
                case V_CHANNEL:
                    cvtColor(imagenes[i], imgOutHsv, CV_BGR2HSV);
                    split(imgOutHsv, channels);
                    channels[2].copyTo(img_out);
                    break;
                case COLOR_PREDOMINANTE:
                    cvtColor (imagenes[i], imgOutHsv, CV_BGR2HSV);
                    img_out = Mat::zeros(imgOutHsv.rows, imgOutHsv.cols, CV_32FC1);

                    for (int y = 0;y < imgOutHsv.rows; y++)
                    {
                        for (int x = 0; x < imgOutHsv.cols; x++)
                        {
                            Vec3b pixel = imgOutHsv.at<Vec3b>(y, x);

                            if (pixel[0] >= 0. && pixel[0]<30.)
                                img_out.at<float>(y,x) = 1;
                            else if (pixel[0] >= 30. && pixel[0] < 60.)
                                img_out.at<float>(y,x) = 2;
                            else if (pixel[0] >= 60. && pixel[0] < 90.)
                                img_out.at<float>(y,x) = 3;
                            else if (pixel[0] >= 90. && pixel[0] < 120.)
                                img_out.at<float>(y,x) = 4;
                            else if (pixel[0] >= 120. && pixel[0] < 150.)
                                img_out.at<float>(y,x) = 5;
                            else if (pixel[0] >= 150. && pixel[0] < 180.)
                                img_out.at<float>(y,x) = 6;
                        }
                    }
                    break;
                default:
                    cout << "ERROR en Extract: El tipo de transformacion indicado no esta contemplado" << endl;
                    return 1;
                }
                break;
            }
            case HSV:
            {
                switch (output)
                {
                case RGB:
                    cvtColor(imagenes[i], img_out, CV_HSV2BGR);
                    break;
                case GRAY:
                    split(imagenes[i], channels);
                    channels[2].copyTo(img_out);
                    break;
                case THRESHOLD:
                    split(imagenes[i], channels);
                    channels[2].copyTo(gray);
                    gray.convertTo(img8u,CV_8U);
                    cv::threshold(img8u, img_out, 0, 1, CV_THRESH_BINARY | CV_THRESH_OTSU);
                    break;
                case CANNY:
                    split(imagenes[i], channels);
                    channels[2].copyTo(gray);
                    imagenes[i].convertTo(img8u, CV_8U);
                    level = cv::threshold(img8u, aux, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
                    cv::Canny(img8u, img_out, 0.5 * level, level);
                    break;
                case SOBEL:
                    split(imagenes[i], channels);
                    channels[2].copyTo(gray);
                    cv::Sobel(gray, imgOutA, CV_32F,1,0);
                    cv::Sobel(gray, imgOutB, CV_32F,0,1);
                    convertScaleAbs(imgOutA + imgOutB, img_out);
                    break;
                case HSV:
                    imagenes[i].copyTo(img_out);
                    break;
                case H_CHANNEL:
                    split(imagenes[i],channels);
                    channels[0].copyTo(img_out);
                    break;
                case S_CHANNEL:
                    split(imagenes[i],channels);
                    channels[1].copyTo(img_out);
                    break;
                case V_CHANNEL:
                    split(imagenes[i],channels);
                    channels[2].copyTo(img_out);
                    break;
                case COLOR_PREDOMINANTE:
                    img_out = Mat::zeros(imagenes[i].rows, imagenes[i].cols, CV_32FC1);
                    for (int y = 0; y < imagenes[i].rows; y++)
                    {
                        for (int x = 0; x < imagenes[i].cols; x++)
                        {
                            Vec3b pixel = imagenes[i].at<Vec3b>(y, x);

                            if (pixel[0] >= 0. && pixel[0] < 30.)
                                img_out.at<float>(y,x) = 1;
                            else if (pixel[0] >= 30. && pixel[0] < 60.)
                                img_out.at<float>(y,x) = 2;
                            else if (pixel[0] >= 60. && pixel[0] < 90.)
                                img_out.at<float>(y,x) = 3;
                            else if (pixel[0] >= 90. && pixel[0] < 120.)
                                img_out.at<float>(y,x) = 4;
                            else if (pixel[0] >= 120. && pixel[0] < 150.)
                                img_out.at<float>(y,x) = 5;
                            else if (pixel[0] >= 150. && pixel[0] < 180.)
                                img_out.at<float>(y,x) = 6;
                        }
                    }
                    break;
                default:
                    cout << "ERROR en Extract: El tipo de transformacion indicado no esta contemplado" << endl;
                    return 1;
                }
                break;
            }
            case GRAY:
            {
                switch (output)
                {
                case RGB:
                    break;
                case GRAY:
                    imagenes[i].copyTo(img_out);
                    break;
                case THRESHOLD:
                    imagenes[i].convertTo(img8u,CV_8U);
                    cv::threshold(img8u, img_out, 0, 1, CV_THRESH_BINARY | CV_THRESH_OTSU);
                    break;
                case CANNY:
                    imagenes[i].convertTo(img8u,CV_8U);
                    level = cv::threshold(img8u, aux, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
                    cv::Canny(img8u, img_out, 0.5 * level, level);
                    break;
                case SOBEL:
                    cv::Sobel(imagenes[i], imgOutA, CV_32F,1,0);
                    cv::Sobel(imagenes[i], imgOutB, CV_32F,0,1);
                    convertScaleAbs(imgOutA + imgOutB, img_out);
                    break;
                //case HSV:
                //    break;
                //case H_CHANNEL:
                //    break;
                //case S_CHANNEL:
                //    break;
                case V_CHANNEL:
                    imagenes[i].copyTo(img_out);
                    break;
                //case COLOR_PREDOMINANTE:
                //    break;
                default:
                    cout << "ERROR en Extract: El tipo de transformacion indicado no esta contemplado" << endl;
                    return 1;
                }
                break;
            }
            case H_CHANNEL:
            {
                switch (output)
                {
                    case COLOR_PREDOMINANTE:
                        img_out = Mat::zeros(imagenes[i].rows,imagenes[i].cols,CV_32FC1);
                        for (int y = 0;y<imagenes[i].rows;y++)
                        {
                            for (int x = 0; x < imagenes[i].cols;x++)
                            {
                                Vec3b pixel = imagenes[i].at<Vec3b>(y, x);
                                if (pixel[0] >= 0. && pixel[0]<30.)
                                    img_out.at<float>(y,x) = 1;
                                else if (pixel[0] >= 30. && pixel[0] < 60.)
                                    img_out.at<float>(y,x) = 2;
                                else if (pixel[0] >= 60. && pixel[0] < 90.)
                                    img_out.at<float>(y,x) = 3;
                                else if (pixel[0] >= 90. && pixel[0] < 120.)
                                    img_out.at<float>(y,x) = 4;
                                else if (pixel[0] >= 120. && pixel[0] < 150.)
                                    img_out.at<float>(y,x) = 5;
                                else if (pixel[0] >= 150. && pixel[0] < 180.)
                                    img_out.at<float>(y,x) = 6;
                            }
                        }
                        break;
                    default:
                        cout << "ERROR en Extract: El tipo de transformacion indicado no esta contemplado" << endl;
                        return 1;
                }
                break;
            }
            default:
                cout << "ERROR en Extract: El tipo de transformacion indicado no esta contemplado" << endl;
                return 1;
        }

        img_out.convertTo (img_out,CV_32F);
        descriptores.push_back (img_out);
    }
    return 0;
}
