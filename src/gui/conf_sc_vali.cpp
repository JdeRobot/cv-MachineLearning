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

#include "conf_sc_vali.h"
#include "ui_conf_sc_vali.h"

Conf_SC_Vali::Conf_SC_Vali(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_SC_Vali)
{
    punt=puntero;
    id_clasificadores.clear();
    ui->setupUi(this);
}

Conf_SC_Vali::~Conf_SC_Vali()
{
    delete ui;
}

void Conf_SC_Vali::on_Iniciar_2_clicked()
{
    MainWindow *window=(MainWindow*) punt;
    if(ui->Distancias->isChecked())
        id_clasificadores.push_back(DISTANCIAS);
    if(ui->GAUSS->isChecked())
        id_clasificadores.push_back(GAUSSIANO);
    if(ui->Histograma->isChecked())
        id_clasificadores.push_back(HISTOGRAMA);
    if(ui->KNN->isChecked())
        id_clasificadores.push_back(KNN);
    if(ui->Neuronal->isChecked())
        id_clasificadores.push_back(NEURONAL);
    if(ui->SVM->isChecked())
        id_clasificadores.push_back(C_SVM);
    if(ui->RTrees->isChecked())
        id_clasificadores.push_back(RTREES);
    if(ui->DTrees->isChecked())
        id_clasificadores.push_back(DTREES);
    if(ui->Boosting->isChecked())
        id_clasificadores.push_back(BOOSTING);
//    if(ui->GBT->isChecked())
//        id_clasificadores.push_back(GBT);
    if(ui->EM->isChecked())
        id_clasificadores.push_back(EXP_MAX);
//    if(ui->ERT->isChecked())
//        id_clasificadores.push_back(ERTREES);
    window->id_clasificadores=id_clasificadores;
    delete this;
}
