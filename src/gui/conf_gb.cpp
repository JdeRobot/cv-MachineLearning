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

#include "conf_gb.h"
#include "ui_conf_gb.h"

Conf_GB::Conf_GB(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_GB)
{
    punt=puntero;
    ui->setupUi(this);
}

Conf_GB::~Conf_GB()
{
    delete ui;
}

void Conf_GB::on_pushButton_clicked()
{
    MainWindow *window=(MainWindow*) punt;
//    if(ui->loss_function_type->currentIndex()==0)
//        window->GBT_loss_function_type=CvGBTrees::SQUARED_LOSS;
//    else if(ui->loss_function_type->currentIndex()==1)
//        window->GBT_loss_function_type=CvGBTrees::ABSOLUTE_LOSS;
//    else if(ui->loss_function_type->currentIndex()==2)
//        window->GBT_loss_function_type=CvGBTrees::HUBER_LOSS;
//    else if(ui->loss_function_type->currentIndex()==3)
//        window->GBT_loss_function_type=CvGBTrees::DEVIANCE_LOSS;
//    window->GBT_max_depth=ui->max_depth->value();
//    window->GBT_shrinkage=ui->shrinkage->value();
//    window->GBT_subsample_portion=ui->subsample_portion->value();
//    window->GBT_use_surrogates=ui->use_surrogates->isChecked();
//    window->GBT_weak_count=ui->weak_count->value();
    delete this;
}
