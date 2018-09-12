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

#include "conf_boosting.h"
#include "ui_conf_boosting.h"

Conf_Boosting::Conf_Boosting(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_Boosting)
{
    punt=puntero;
    ui->setupUi(this);
}

Conf_Boosting::~Conf_Boosting()
{
    delete ui;
}

void Conf_Boosting::on_pushButton_clicked()
{
    MainWindow *window=(MainWindow*) punt;
    window->parameters.Boosting_priors=0;
    if(ui->boost_type->currentIndex()==0)
        window->parameters.Boosting_boost_type=ml::Boost::DISCRETE;
    else if(ui->boost_type->currentIndex()==1)
        window->parameters.Boosting_boost_type=ml::Boost::REAL;
    else if(ui->boost_type->currentIndex()==2)
        window->parameters.Boosting_boost_type=ml::Boost::LOGIT;
    else if(ui->boost_type->currentIndex()==3)
        window->parameters.Boosting_boost_type=ml::Boost::GENTLE;
    window->parameters.Boosting_max_depth=ui->max_depth->value();
    window->parameters.Boosting_use_surrogates=ui->use_surrogates->isChecked();
    window->parameters.Boosting_weak_count=ui->weak_count->value();
    window->parameters.Boosting_weight_trim_rate=ui->weight_trim_rate->value();
    delete this;
}
