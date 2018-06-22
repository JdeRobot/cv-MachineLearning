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

#ifndef GENERACION_H
#define GENERACION_H

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <fstream>
#include <iostream>
#include "auxiliares.h"

#ifdef GUI
#include "ui_mainwindow.h"
#endif

using namespace std;
using namespace cv;


void mouseEvent(int evt, int x, int y, int flags, void* param);
void mouseEvent2(int evt, int x, int y, int flags, void* param);

namespace MLT {
    class Generacion
    {
    public:
    struct Info_Datos {
           int Tipo_Datos;
           int Num_Datos;
           int Tam_X;
           int Tam_Y;
           int Tam_Orig_X;
           int Tam_Orig_Y;
           bool si_lda;
           bool si_pca;
           bool si_dist;
           bool si_d_prime;
           Mat LDA;
           Mat PCA;
           Mat DS;
           Mat D_PRIME;
           Info_Datos(){
               Tipo_Datos=0;
               Tam_X=0;
               Tam_Y=0;
               Tam_Orig_X=0;
               Tam_Orig_Y=0;
               si_lda=false;
               si_pca=false;
               si_dist=false;
               si_d_prime=false;
               LDA=Mat();
               PCA=Mat();
               DS=Mat();
               D_PRIME=Mat();
           }
        };

        Generacion();

        int Cargar_Imagenes(string input_directory, std::vector<cv::Mat> &Images);
        int Guardar_Datos(string nombre, vector<Mat> Imagenes, vector<float> Labels, Info_Datos info);
        int Cargar_Fichero(string Archivo, vector<Mat> &Imagenes, vector<float> &Labels, Info_Datos &info);
        int Juntar_Recortes(string nombre, string Path);
        int Datos_Imagenes(string nombre, string input_directory, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save);
        int Etiquetar(string nombre, string input_directory, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save);
        int Recortar_Etiquetar(string nombre, string input_directory, bool cuadrado, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save);
        int Recortar_Etiquetar(string nombre, VideoCapture cap, bool cuadrado, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save);
        int Random_Synthetic_Data(string nombre, int num_clases, int num_data_clase, Size tam_img, float ancho, float separacion_clases, vector<Mat> &Data, vector<float> &Labels, Info_Datos &info, bool save);
        int Random_Synthetic_Image(int num_clases, Size tam_img, float ancho, float separacion_clases,  Mat &Imagen);
        int Synthethic_Data(string nombre, vector<Mat> input, vector<float> inputLabels, vector<Mat> &output, vector<float> &outputLabels, int num_by_frame, float max_noise, float max_blur, float max_rot_x, float max_rot_y, float max_rot_z, Info_Datos &info, bool save);
        int Autopositivos(string nombre, VideoCapture cap, bool cuadrado, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save);
        int Autonegativos(string nombre, string Archivo, Size reescalado, int num_recortes_imagen, vector<Mat> &Negativos, vector<float> &Labels, Info_Datos &info,bool save);
        int Autogeneracion(string nombre, VideoCapture cap, int num_negativos_imagen, bool cuadrado, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save);

        int pos_x, pos_y, pos_x2, pos_y2, tam_x, tam_y, p_x,p_y, tam_ag_x,tam_ag_y;
        cv::Mat frame, frame2;
        bool flag,flag2;
        bool Cuadrado;

    #ifdef GUI
        int progreso;
        int total_progreso;
        bool error;
        bool running;
    #endif

    };
}

#endif // GENERACION_H
