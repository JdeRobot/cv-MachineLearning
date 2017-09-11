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

#include "config_pc.h"
#include "ui_config_pc.h"

Config_PC::Config_PC(void *puntero,QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Config_PC)
{
    punt=puntero;
    ui->setupUi(this);
}

Config_PC::~Config_PC()
{
    delete ui;
}

void Config_PC::on_Aceptar_clicked()
{
    MainWindow *window=(MainWindow*) punt;
    QString descriptor,detector;
    descriptor=ui->Descriptor->currentText();
    detector=ui->Detector->currentText();
    window->Tipo_Des=descriptor.toStdString();
    window->Tipo_Ext=detector.toStdString();
    window->Parametro=(float)ui->Parametro->value();
    delete this;
}
