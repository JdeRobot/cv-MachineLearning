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

#include "conf_rtrees.h"
#include "ui_conf_rtrees.h"

Conf_RTrees::Conf_RTrees(void *puntero,QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_RTrees)
{
    punt=puntero;
    ui->setupUi(this);
}

Conf_RTrees::~Conf_RTrees()
{
    delete ui;
}

void Conf_RTrees::on_Aceptar_clicked()
{
     MainWindow *window=(MainWindow*) punt;
     window->RTrees_priors=0;
     window->RTrees_calc_var_importance=ui->calc_var_importance->isChecked();
     window->RTrees_cv_folds=ui->cv_folds->value();
     window->RTrees_max_categories=ui->max_categories->value();
     window->RTrees_max_depth=ui->max_depth->value();
     window->RTrees_min_sample_count=ui->min_sample_count->value();
     window->RTrees_native_vars=ui->native_vars->value();
     window->RTrees_regression_accuracy=ui->regression_accuracy->value();
     window->RTrees_truncate_pruned_tree=ui->truncate_pruned_tree->isChecked();
     window->RTrees_use_1se_rule=ui->use_1se_rule->isChecked();
     window->RTrees_use_surrogates=ui->use_surrogates->isChecked();
     delete this;
}
