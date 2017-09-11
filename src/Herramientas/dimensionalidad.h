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

#ifndef DIMENSIONALIDAD_H
#define DIMENSIONALIDAD_H

#include <opencv2/opencv.hpp>
#include <Herramientas/auxiliares.h>
#include <Herramientas/generacion.h>

using namespace std;
using namespace cv;

namespace MLT {
    enum
    {
        LDA_DIM=0,
        PCA_DIM=1,
        MAXDIST_DIM=2,
        D_PRIME_DIM=3
    };

    class Dimensionalidad
    {
    public:
        struct Reducciones {
           bool si_lda;
           bool si_pca;
           bool si_dist;
           bool si_d_prime;
           int tam_reduc;
           Mat LDA;
           Mat PCA;
           Mat DS;
           Mat D_PRIME;
           Reducciones(){
               si_lda=false;
               si_pca=false;
               si_dist=false;
               si_d_prime=false;
               tam_reduc=0;
               LDA=Mat();
               PCA=Mat();
               DS=Mat();
               D_PRIME=Mat();
           }
        };

        Dimensionalidad(string Nombre);

        int Reducir(vector<Mat> Imagenes, vector<Mat> &Reducidas, vector<float> Labels, Reducciones reduccion, Generacion::Info_Datos &info, bool save);
        int LDA_matriz(Mat img, std::vector<float> Etiquetas, int tam_final, Mat &lda, bool guardar);
        int PCA_matriz(Mat img, int tam_final, Mat &pca, bool guardar);
        int MaxDist_matriz(Mat img, std::vector<float> Etiquetas, int tam_final, Mat &mat_reduc, bool guardar);
        int D_Prime_matriz(Mat img, std::vector<float> Etiquetas, int tam_final, Mat &mat_reduc, bool guardar);
        int Proyeccion(Mat img, Mat &Proyectada, int tipo, Mat reduc=Mat());
        int Retro_Proyeccion(Mat img, Mat &Proyectada, int tipo);
        int Calidad_dimensiones_distancia(vector<Mat> img, vector<float> Etiquetas, int tipo_reduccion, int dim_max, Mat &Separabilidad, Mat &Separabilidad_acumulada, int &dim_optim);
        int Calidad_dimensiones_d_prime(vector<Mat> img, vector<float> Etiquetas, int tipo_reduccion, int dim_max, Mat &Separabilidad, Mat &Separabilidad_acumulada, int &dim_optim);

        string nombre;
    };
}

#endif // DIMENSIONALIDAD_H
