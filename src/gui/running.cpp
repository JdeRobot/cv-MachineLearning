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

#include "running.h"

MLT::Running::Running(){}

int MLT::Running::load_dataset(QString path, string &ref, bool &negative,std::vector<float> &labels, std::vector<cv::Mat> &images){
    Generacion gen;
    Auxiliares aux;

    std::string dir=path.toStdString();
    int pos=0;
    for(uint i=0; i<dir.size(); i++){
        if(dir[i]=='/')
            pos=i;
    }
    for(uint i=pos+1; i<dir.size(); i++)
        ref=ref+dir[i];
    string archivo_i=dir+"/Info.xml";
    cv::FileStorage Archivo_i(archivo_i,CV_STORAGE_READ);

    if(!Archivo_i.isOpened())
        return 1;

    int num;
    Archivo_i["Num_Datos"]>>num;
    Archivo_i.release();

    string input_directory=dir+"/Recortes.txt";
    Generacion::Info_Datos info;
    int e=gen.Cargar_Fichero(input_directory,images,labels,info);

    if(e==1)
        return 2;

    aux.numero_etiquetas(labels,negative);

    return 0;
}
