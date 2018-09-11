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
    this->punt=puntero;
    this->id_clasificadores.clear();
    this->ui->setupUi(this);
}

Conf_SC_Vali::~Conf_SC_Vali()
{
    delete ui;
}

void Conf_SC_Vali::on_Iniciar_2_clicked()
{
    MainWindow *window=(MainWindow*) punt;
    if(this->ui->Distancias->isChecked())
        this->id_clasificadores.push_back(DISTANCIAS);
    if(this->ui->GAUSS->isChecked())
        this->id_clasificadores.push_back(GAUSSIANO);
    if(this->ui->Histograma->isChecked())
        this->id_clasificadores.push_back(HISTOGRAMA);
    if(this->ui->KNN->isChecked())
        this->id_clasificadores.push_back(KNN);
    if(this->ui->Neuronal->isChecked())
        this->id_clasificadores.push_back(NEURONAL);
    if(this->ui->SVM->isChecked())
        this->id_clasificadores.push_back(C_SVM);
    if(this->ui->RTrees->isChecked())
        this->id_clasificadores.push_back(RTREES);
    if(this->ui->DTrees->isChecked())
        this->id_clasificadores.push_back(DTREES);
    if(this->ui->Boosting->isChecked())
        this->id_clasificadores.push_back(BOOSTING);
//    if(ui->GBT->isChecked())
//        id_clasificadores.push_back(GBT);
    if(this->ui->EM->isChecked())
        this->id_clasificadores.push_back(EXP_MAX);
//    if(ui->ERT->isChecked())
//        id_clasificadores.push_back(ERTREES);
    window->multi_type.identificadores=this->id_clasificadores;
    delete this;
}
