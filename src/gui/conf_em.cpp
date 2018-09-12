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

#include "conf_em.h"
#include "ui_conf_em.h"

Conf_EM::Conf_EM(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_EM)
{
    punt=puntero;
    ui->setupUi(this);
}

Conf_EM::~Conf_EM()
{
    delete ui;
}

void Conf_EM::on_Aceptar_clicked()
{
    MainWindow *window=(MainWindow*) punt;
    if(ui->Tipo_Cov->currentIndex()==0)
        window->parameters.EM_covMatType=ml::EM::COV_MAT_SPHERICAL;
    if(ui->Tipo_Cov->currentIndex()==1)
        window->parameters.EM_covMatType=ml::EM::COV_MAT_DIAGONAL;
    if(ui->Tipo_Cov->currentIndex()==2)
        window->parameters.EM_covMatType=ml::EM::COV_MAT_GENERIC;
    window->parameters.EM_nclusters=ui->Num_Clus->value();
    delete this;
}
