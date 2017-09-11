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

#include "conf_dtrees.h"
#include "ui_conf_dtrees.h"

Conf_DTrees::Conf_DTrees(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_DTrees)
{
    punt=puntero;
    ui->setupUi(this);
    ui->max_depth->setValue(INT_MAX);
}

Conf_DTrees::~Conf_DTrees()
{
    delete ui;
}

void Conf_DTrees::on_Aceptar_clicked()
{
    MainWindow *window=(MainWindow*) punt;
    window->DTrees_priors=0;
    window->DTrees_cv_folds=ui->cv_folds->value();
    window->DTrees_max_categories=ui->max_categories->value();
    window->DTrees_max_depth=ui->max_depth->value();
    window->DTrees_min_sample_count=ui->min_sample_count->value();
    window->DTrees_regression_accuracy=ui->regression_accuracy->value();
    window->DTrees_truncate_pruned_tree=ui->truncate_pruned_tree->isChecked();
    window->DTrees_use_1se_rule=ui->use_1se_rule->isChecked();
    window->DTrees_use_surrogates=ui->use_surrogates->isChecked();
    delete this;
}
