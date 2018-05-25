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

void MLT::Running::update(){
    this->window->v_progress_datamanaging->setValue(this->gen.base_progreso+(this->gen.max_progreso*this->gen.progreso/this->gen.total_progreso));
    this->window->i_progress_datamanaging->setValue(this->gen.base_progreso+(this->gen.max_progreso*this->gen.progreso/this->gen.total_progreso));
    cout<<"Estoy"<<endl;
}

int MLT::Running::load_dataset(QString path, string &ref, bool &negative,std::vector<float> &labels, std::vector<cv::Mat> &images){

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
//    int e=gen.Cargar_Fichero(input_directory,images,labels,info);

    gen.total_progreso=num;
    gen.progreso=0;
    gen.base_progreso=0;
    gen.max_progreso=100;
    gen.running=true;
    std::thread thrd(&MLT::Generacion::Cargar_Fichero,&gen,input_directory,std::ref(images),std::ref(labels),std::ref(info));
    while(gen.running==true){
        update();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    thrd.join();

    if(gen.running==false)
        cout<<"Estoy a false"<<endl;
    if(gen.error==1)
        return 2;

    aux.numero_etiquetas(labels,negative);

    return 0;
}
