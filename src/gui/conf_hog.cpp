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

#include "conf_hog.h"
#include "ui_conf_hog.h"

Conf_HOG::Conf_HOG(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_HOG)
{
    punt=puntero;
    ui->setupUi(this);
}

Conf_HOG::~Conf_HOG()
{
    delete ui;
}

void Conf_HOG::on_Aceptar_clicked()
{
    MainWindow *window=(MainWindow*) punt;
    window->win_size_x=ui->Win_Size_X->value();
    window->win_size_y=ui->Win_Size_Y->value();
    window->block_x=ui->Block_Stride_X->value();
    window->block_y=ui->Block_Stride_Y->value();
    window->sigma=ui->Sigma->value();
    window->threhold_l2hys=ui->Threshold->value();
    window->gamma=ui->Gamma->isChecked();
    window->nlevels=ui->NLevels->value();
    delete this;
}
