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

#include "conf_svm.h"
#include "ui_conf_svm.h"

Conf_SVM::Conf_SVM(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_SVM)
{
    punt=puntero;
    ui->setupUi(this);
}

Conf_SVM::~Conf_SVM()
{
    delete ui;
}

void Conf_SVM::on_pushButton_clicked()
{
    MainWindow *window=(MainWindow*) punt;
    if(ui->Train->currentIndex()==0)
        window->SVM_train=TRAIN;
    else if(ui->Train->currentIndex()==1)
        window->SVM_train=TRAIN_AUTO;
    if(ui->Tipo_SVM->currentIndex()==0)
        window->SVM_Type=ml::SVM::C_SVC;
    else if(ui->Tipo_SVM->currentIndex()==1)
        window->SVM_Type=ml::SVM::NU_SVC;
    else if(ui->Tipo_SVM->currentIndex()==2)
        window->SVM_Type=ml::SVM::ONE_CLASS;
    else if(ui->Tipo_SVM->currentIndex()==3)
        window->SVM_Type=ml::SVM::EPS_SVR;
    else if(ui->Tipo_SVM->currentIndex()==4)
        window->SVM_Type=ml::SVM::NU_SVR;
    if(ui->Kernel->currentIndex()==0)
        window->SVM_kernel_type=ml::SVM::LINEAR;
    else if(ui->Kernel->currentIndex()==1)
        window->SVM_kernel_type=ml::SVM::POLY;
    else if(ui->Kernel->currentIndex()==2)
        window->SVM_kernel_type=ml::SVM::RBF;
    else if(ui->Kernel->currentIndex()==3)
        window->SVM_kernel_type=ml::SVM::SIGMOID;
    window->SVM_degree=(double)ui->Degree->value();
    window->SVM_gamma=(double)ui->Gamma->value();
    window->SVM_coef0=(double)ui->Coef0->value();
    window->SVM_C=(double)ui->C->value();
    window->SVM_nu=(double)ui->nu->value();
    window->SVM_p=(double)ui->p->value();
    delete this;
}
