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
        int load_dataset(string path);
        int synthetic_data(string ref, int num_classes, int num_data_class, int vector_size, float ancho, float separacion_clases);
        int save(string ref);
        int join_data(string ref, string path);
        int plot_data(int type_plot, vector<int> dim);
        int plot_results(int type_plot);
        int analyse_data(QStandardItemModel *model);
        int analyse_result(QStandardItemModel *model);
        int clustering(string ref, int type, int k, int repetitions, float max_dist, float cell_size);
        int dimensionality(string ref, int size_reduc, int type);
        int dimension_cuality(string ref, int size_reduc, int type_reduc, int type_measure, string &result);


        Ui::MainWindow *window;

        int num_bar;
        bool show_graphics;
        bool save_clasif;
        bool save_data;
        bool save_other;
        bool read;
        bool ifreduc;

        Generacion::Info_Datos org_info;
        string org_ref;
        std::vector<cv::Mat> org_images;
        std::vector<float> org_labels;

        string result_ref;
        Generacion::Info_Datos result_info;
        std::vector<cv::Mat> result_images;
        std::vector<float> result_labels;
//        std::vector<float> resultado;

        vector<cv::Scalar> colors;


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
