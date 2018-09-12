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

#include "conf_haar.h"
#include "ui_conf_haar.h"

Conf_HAAR::Conf_HAAR(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_HAAR)
{
    punt=puntero;
    ui->setupUi(this);
}

Conf_HAAR::~Conf_HAAR()
{
    delete ui;
}

void Conf_HAAR::on_pushButton_clicked()
{
    MainWindow *window=(MainWindow*) punt;
    window->parameters.Cascada_NumPos=ui->Positivos->value();
    window->parameters.Cascada_NumNeg=ui->Negativos->value();
    window->parameters.Cascada_Mode=ui->Modo->currentText().toStdString();
    window->parameters.Cascada_NumStage=ui->Etapas->value();
    window->parameters.Cascada_MinHitRate=ui->MinHitRate->value();
    window->parameters.Cascada_MaxFalseAlarmRate=ui->MaxFalseAlarmRate->value();
    window->parameters.Cascada_WeightTrimRate=ui->WeightTrimRate->value();
    window->parameters.Cascada_MaxWeakCount=ui->MaxWeakCount->value();
    window->parameters.Cascada_MaxDepth=ui->MaxDepth->value();
    window->parameters.Cascada_Bt=ui->Bt->currentText().toStdString();
    window->parameters.Cascada_PrecalcValBufSize=ui->PrecalcValBufSize->value();
    window->parameters.Cascada_PrecalcidxBufSize=ui->PrecalcidxBufSize->value();
    window->si_entrenar=!ui->Entrenar->isChecked();
    delete this;
}
