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

#include "auxiliares.h"

MLT::Auxiliares::Auxiliares(){}

int MLT::Auxiliares::numero_imagenes(String input_directory){
    string strPrefix;
    DIR    *dir_p = opendir (input_directory.c_str());
    struct dirent *dir_entry_p;
    int num=0;
    while((dir_entry_p = readdir(dir_p)) != NULL){
        if(strcmp(dir_entry_p->d_name, ""))
            strPrefix=input_directory+dir_entry_p->d_name;
        if(strcmp(dir_entry_p->d_name, ".")!=0 && strcmp(dir_entry_p->d_name, "..")!=0){
            num++;
        }
    }
    return num;
}

int MLT::Auxiliares::numero_etiquetas(vector<float> Labels, bool &negativa){
    int num_etiq=0;
    negativa=false;
    for(uint i=0; i<Labels.size(); i++){
        if(Labels[i]<0)
            negativa=true;
        if(num_etiq<Labels[i])
            num_etiq=Labels[i];
    }
    if(negativa)
        num_etiq=num_etiq+1;
    return num_etiq;
}

//Cada dato es una ROW
int MLT::Auxiliares::Image2Lexic(std::vector<cv::Mat> Imagen, cv::Mat &Datos){
    if(Imagen.size()==0){
        cout<<"ERROR en Image2Lexic: Imagen esta vacio"<<endl;
        return 1;
    }
    int n_rows=Imagen[0].rows;
    int n_cols=Imagen[0].cols;
    int n_channels=Imagen[0].channels();
    for(uint i=1; i<Imagen.size(); i++){
        if(Imagen[i].rows!=n_rows || Imagen[i].cols>n_cols || Imagen[i].channels()>n_channels){
            cout<<"ERROR en Image2Lexic: Los datos no son del mismo tipo"<<endl;
            return 1;
        }
    }
    Mat D=Mat::zeros(Imagen.size(),n_channels*n_cols*n_rows,CV_32FC1);
    for(uint k=0; k<Imagen.size(); k++){
        int z=Imagen[k].channels();
        vector<Mat> channels;
        if(z>1){
            split(Imagen[k],channels);
        }
        else
            channels.push_back(Imagen[k]);
        for(int c=0; c<z; c++){
            int x=channels[c].rows;
            int y=channels[c].cols;
            for(int i=0; i<x; i++){
                for(int j=0; j<y; j++){
                    D.at<float>(k,(c*x*y)+(i*y)+j)=channels[c].at<float>(i,j);
                }
            }
        }
    }
    D.copyTo(Datos);
    return 0;
}

int MLT::Auxiliares::Lexic2Image(cv::Mat Datos, Size tam_imagen, int num_channels, std::vector<cv::Mat> &Imagen){
    if(Datos.empty()){
        cout<<"ERROR en Lexic2Image: Datos de entrada vacíos"<<endl;
        return 1;
    }
    if(tam_imagen.height*tam_imagen.width*num_channels!= Datos.cols){
        cout<<"ERROR en Lexic2Image: El tamaño introducido como entrada no encaja con el de los datos"<<endl;
        return 1;
    }
    for(int k=0; k<Datos.rows; k++){
        vector<Mat> channels;
        for(int z=0; z<num_channels;z++){
            Mat im=Mat::zeros(tam_imagen.height,tam_imagen.width,CV_32FC1);
            for(int i=0; i<tam_imagen.height; i++){
                for(int j=0; j<tam_imagen.width; j++){
                    im.at<float>(i,j)=Datos.at<float>(k,(z*tam_imagen.width*tam_imagen.height)+(tam_imagen.width*i)+j);
                }
            }
            channels.push_back(im);
        }
        Mat imagen;
        merge(channels,imagen);
        Imagen.push_back(imagen);
    }
    return 0;
}
