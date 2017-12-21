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

#ifndef CLASIFICADOR_SVM_H
#define CLASIFICADOR_SVM_H

#include "clasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    enum{
        TRAIN=0,
        TRAIN_AUTO=1
    };

    class Clasificador_SVM: public Clasificador{
    public:
        Clasificador_SVM(string Nombre="",int train=TRAIN_AUTO, int Type=ml::SVM::NU_SVC, int kernel_type=ml::SVM::RBF, Mat class_weights=Mat(), double degree=0, double gamma=1, double coef0=0, double C=1, double nu=0.00001, double p=0);
        ~Clasificador_SVM();
        int Parametrizar(int train, int Type, int kernel_type, Mat class_weights, double degree, double gamma, double coef0, double C, double nu, double p);

        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true) override;
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read) override;
        int SaveData() override;
        int ReadData() override;

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat) override;
        float Clasificacion(Mat Data) override;

        Ptr<ml::SVM>  SVM;
        int tipo_entrena;
        Dimensionalidad::Reducciones reduccion;
    };
}

#endif // CLASIFICADOR_SVM_H
