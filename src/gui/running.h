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
//#include "../Clasificadores/miclasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Running
    {
    public:
        Running();
        int load_dataset(string path);
        int synthetic_data(string ref, int classes, int number, int size_x, int size_y, float variance, float interclass);
        int save(string ref);
        int join_data(string ref, string path);
        int plot_data(int type_plot, vector<int> dim);
        int plot_results(int type_plot);
        int analyse_data(QStandardItemModel *model);
        int analyse_result(QStandardItemModel *model);
        int clustering(string ref, int type, int k, int repetitions, float max_dist, float cell_size);
        int dimensionality(string ref, int size_reduc, int type);
        int dimension_cuality(string ref, int size_reduc, int type_reduc, int type_measure, string &result);
        int generate_data(string ref, string input_directory, int type, int scale_x, int scale_y, bool square, int number);
        int descriptors(string &ref, int descriptor, string pc_descriptor, string extractor,
                        int size_x, int size_y, int block_x, int block_y, double sigma, double threshold, bool gamma, int n_levels, bool descriptor_parameter);
        int expand_dataset(string ref, int nframe, float max_noise, float max_blur, float max_x, float max_y, float max_z);
        int represent_images(int type, int label);
        int detect_image(int type_running, int input_type, string input_path, int descriptor_type, MultiClasificador::Multi_type multi_params,
                         int n_classes, float variance, float interclass,
                         int window_x, int window_y, int jump, int pyramid, int rotation,
                         bool postprocess, bool overlap, bool isolated, float dist_boxes,
                         int dist_rotation, string pc_descriptor, string extractor,
                         int size_x, int size_y, int block_x, int block_y, double sigma,
                         double threhold_l2hys, bool gamma, int n_levels, bool descriptor_parameter, Mat &image, Mat &output, vector<RotatedRect> &detections, vector<float> &labels_detections);
        int train(string ref, int classifier_type, Clasificadores::Parametros params);
        int load_model(string path, string &name);
        int classify(string ref, int type_classification, stringstream &txt, MultiClasificador::Multi_type multi_params);
        int optimize(int type, int id_classifier, MultiClasificador::Multi_type multi_type, stringstream &text,
                     Clasificadores::Parametros start, Clasificadores::Parametros leap, Clasificadores::Parametros stop,
                     int percentage, int num_folds, int size_fold);



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

        vector<cv::Scalar> colors;



    private:
        void update_gen();
        void update_analysis();
        void update_classifier(int progress, int total_progress);

        Generacion gen;
        Analisis ana;
        Clasificador *classifier;

        int max_progreso;
        int base_progreso;

    };
}

#endif // RUNNING_H
