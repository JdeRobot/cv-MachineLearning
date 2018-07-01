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

#ifndef ANALISIS_H
#define ANALISIS_H

#include <opencv2/opencv.hpp>
#include "auxiliares.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Analisis
    {
    public:
        struct Ellipse_data{
            float angle;
            float small;
            float big;
            float media_x;
            float media_y;
        };

        struct Ratios_data{
            float valor_x;
            float VP;
            float VN;
            float FN;
            float FP;
            float FAR;
            float FRR;
            float TAR;
            float TRR;
            float PPV;
            float NPV;
            float F1;
            float FDR;
            float INFORMEDNESS;
            float MARKEDNESS;
            float EXP_ERROR;
            float LR_POS;
            float LR_NEG;
            float DOR;
            float ACC;
            float PREVALENCE;
            Ratios_data(){valor_x=0; VP=0; VN=0; FN=0; FP=0; FAR=0; FRR=0; TAR=0; TRR=0;
                          F1=0;PPV=0;NPV=0;INFORMEDNESS=0;MARKEDNESS=0;FDR=0;
                          EXP_ERROR=0;LR_POS=0;LR_NEG=0;DOR=0;ACC=0;PREVALENCE=0;
                         }
        };


        Analisis();

        int Confusion(std::vector<float> Etiquetas, std::vector<float> Resultados, Mat &Confusion, float &error);
        int Ratios(std::vector<float> Etiquetas, std::vector<float> Resultados, vector<Ratios_data> &Rat);
        int Ratios_Histograma(std::vector<Mat> Datos, std::vector<float> Etiquetas, std::vector<float> Resultados, int num_barras, vector<vector<Ratios_data> > &Hist_Rat);
        int Estadisticos(vector<Mat> Datos, vector<float> Etiquetas, vector<Mat> &Medias, vector<Mat> &Des_Tipics, vector<vector<Mat> > &D_prime);
        int Estadisticos(Mat Datos, vector<float> Etiquetas, vector<Mat> &Medias, vector<Mat> &Des_Tipics, vector<vector<Mat> > &D_prime);
        int Covarianza(vector<Mat> Datos, vector<float> Etiquetas, vector<Mat> &Covarianzas);
        int Covarianza(Mat Datos, vector<float> Etiquetas, vector<Mat> &Covarianzas);
        int Estadisticos_Covarianzas(vector<Mat> Datos, vector<float> Etiquetas, vector<Mat> &Medias, vector<Mat> &Des_Tipics, vector<vector<Mat> > &D_Prime, vector<Mat> &Covarianzas, bool &negativa, vector<int> &numero);
        int Histograma(vector<Mat> Datos, vector<float> Etiquetas, int Num_Barras, vector<vector<Mat> > &His, vector<vector<int> > &pos_barra);
        int Ellipse_Error(vector<Mat> Datos, vector<float> Etiquetas, vector<int> dimensiones, vector<Ellipse_data> &Elipses);   

#ifdef GUI
    int progreso;
    int total_progreso;
    bool error;
    bool running;
#endif
    };
}

#endif // ANALISIS_H
