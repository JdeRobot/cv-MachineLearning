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

#include "conf_neuronal.h"
#include "ui_conf_neuronal.h"

Conf_neuronal::Conf_neuronal(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_neuronal)
{
    punt=puntero;
    ui->setupUi(this);
    ui->rp_dw_min->setValue(FLT_EPSILON);
}

Conf_neuronal::~Conf_neuronal()
{
    delete ui;
}

void Conf_neuronal::on_Aceptar_clicked()
{
    MainWindow *window=(MainWindow*) punt;
    std::string texto=ui->Num_neur->text().toStdString();
    std::vector<int> pos;
    pos.push_back(-1);
    for(uint i=0; i<texto.size(); i++){
        if(texto[i]==',')
            pos.push_back(i);
    }
    pos.push_back(texto.size());
    std::vector<int> num;
    for(uint i=1; i<pos.size(); i++){
        std::string numero;
        for(int j=pos[i-1]+1; j<pos[i]; j++){
            numero=numero+texto[j];
        }
        int n=atoi(numero.c_str());
        num.push_back(n);
    }
//    window->Neuronal_layerSize=cv::Mat::zeros(num.size()+2,1,CV_32SC1);
//    window->Neuronal_layerSize.row(window->Neuronal_layerSize.rows-1)=1;
//    for(int i=1; i<window->Neuronal_layerSize.rows-1; i++)
//        window->Neuronal_layerSize.row(i)=cv::Scalar(num[i-1]);
//    if(ui->Metodo->currentIndex()==0)
//        window->Neuronal_Method=ml::ANN_MLP::BACKPROP;
//    else if(ui->Metodo->currentIndex()==1)
//        window->Neuronal_Method=ml::ANN_MLP::RPROP;
//    if(ui->Funcion->currentIndex()==0)
//        window->Neuronal_Function=ml::ANN_MLP::IDENTITY;
//    else if(ui->Funcion->currentIndex()==1)
//        window->Neuronal_Function=ml::ANN_MLP::SIGMOID_SYM;
//    else if(ui->Funcion->currentIndex()==2)
//        window->Neuronal_Function=ml::ANN_MLP::GAUSSIAN;
//    window->Neuronal_bp_dw_scale=ui->bp_dw_scale->value();
//    window->Neuronal_fparam1=ui->fparam1->value();
//    window->Neuronal_fparam2=ui->fparam2->value();
//    window->Neuronal_rp_dw0=ui->rp_dw0->value();
//    window->Neuronal_rp_dw_max=ui->rp_dw_max->value();
//    window->Neuronal_rp_dw_min=ui->rp_dw_min->value();
//    window->Neuronal_rp_dw_minus=ui->rp_dw_minus->value();
//    window->Neuronal_rp_dw_plus=ui->rp_dw_plus->value();
//    delete this;
}
