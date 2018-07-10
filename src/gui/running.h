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

#ifndef RUNNING_H
#define RUNNING_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <QStandardItem>
#include "../ClasificationSystem.h"
#include "../Clasificadores/miclasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Running
    {
    public:
        Running();
        int load_dataset(QString path, string &ref, std::vector<float> &labels, std::vector<Mat> &images, Generacion::Info_Datos &info);
        int synthetic_data(QString nombre, int num_classes, int num_data_class, int vector_size, float ancho, float separacion_clases, std::vector<Mat> &data, std::vector<float> &labels, Generacion::Info_Datos &info);
        int join_data(string ref, QString path);
        int save(string ref, vector<Mat> images, vector<float> labels, Generacion::Info_Datos info);
        int analyse_data(vector<Mat> images, vector<float> labels, QStandardItemModel *model);
        int analyse_result(vector<float> labels, vector<float> results, QStandardItemModel *model);
        int clustering(vector<Mat> images, int type, int k, int repetitions, float max_dist, float cell_size, vector<float> &labels);

        Ui::MainWindow *window;

        int num_bar;
        bool show_graphics;
        bool save_clasif;
        bool save_data;
        bool save_other;
        bool read;
        bool ifreduc;

    private:
        void update_gen();
        void update_analysis();

        Generacion gen;
        Analisis ana;

        int max_progreso;
        int base_progreso;
    };
}

#endif // RUNNING_H
