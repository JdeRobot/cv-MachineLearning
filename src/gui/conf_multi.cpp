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

#include "conf_multi.h"
#include "ui_conf_multi.h"

Conf_Multi::Conf_Multi(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Conf_Multi)
{
    punt=puntero;
    texto.str("");
    id_clasificadores.clear();
    multi.tipo=CASCADA;
    multi.label_ref.clear();
    multi.tipo_regla.clear();
    ui->setupUi(this);
    ui->Informacion->clear();
}

Conf_Multi::~Conf_Multi()
{
    delete ui;
}

void Conf_Multi::on_Anadir_clicked()
{
    int id;
    if(ui->Distancias_2->isChecked())
        id=DISTANCIAS;
    if(ui->GAUSS_2->isChecked())
        id=GAUSSIANO;
    if(ui->Histograma_2->isChecked())
        id=HISTOGRAMA;
    if(ui->KNN_2->isChecked())
        id=KNN;
    if(ui->Neuronal_2->isChecked())
        id=NEURONAL;
    if(ui->SVM_2->isChecked())
        id=C_SVM;
    if(ui->RTrees_2->isChecked())
        id=RTREES;
    if(ui->DTrees_2->isChecked())
        id=DTREES;
    if(ui->Boosting_2->isChecked())
        id=BOOSTING;
//    if(ui->GBT_2->isChecked())
//        id=GBT;
    if(ui->EM_2->isChecked())
        id=EXP_MAX;
//    if(ui->ERT_2->isChecked())
//        id=ERTREES;
    id_clasificadores.push_back(id);
    texto<<"Clasificador tipo: ";
    if(id==DISTANCIAS)
        texto<<"DISTANCIAS"<<endl;
    else if(id==GAUSSIANO)
        texto<<"GAUSSIANO"<<endl;
    else if(id==CASCADA_CLAS)
        texto<<"CASCADA_CLAS"<<endl;
    else if(id==HISTOGRAMA)
        texto<<"HISTOGRAMA"<<endl;
    else if(id==KNN)
        texto<<"KNN"<<endl;
    else if(id==NEURONAL)
        texto<<"NEURONAL"<<endl;
    else if(id==C_SVM)
        texto<<"SVM"<<endl;
    else if(id==RTREES)
        texto<<"RTREES"<<endl;
    else if(id==DTREES)
        texto<<"DTREES"<<endl;
    else if(id==BOOSTING)
        texto<<"BOOSTING"<<endl;
//    else if(id==GBT)
//        texto<<"GBT"<<endl;
    else if(id==EXP_MAX)
        texto<<"EXPECTATION MAXIMIZATION"<<endl;
//    else if(id==ERTREES)
//        texto<<"ERTREES"<<endl;
    ui->Informacion->setText(QString::fromStdString(texto.str()));
    ui->Anadir_2->setEnabled(true);
    ui->Anadir->setEnabled(false);
    if(ui->Cascada_2->isChecked())
        ui->Aceptar_2->setEnabled(true);
    else if(ui->Votacion_2->isChecked())
        ui->Aceptar_2->setEnabled(false);
}

void Conf_Multi::on_Anadir_2_clicked()
{
    if(ui->Cascada_2->isChecked()){
        if(ui->Etiq_Ref->value()==0){
            QMessageBox msgBox;
            msgBox.setText("ERROR: No se puede utilizar la etiqueta 0");
            msgBox.exec();
            return;
        }
        multi.label_ref.push_back(ui->Etiq_Ref->value());
        if(ui->Regla->currentIndex()==0){
            multi.tipo_regla.push_back(IGUAL);
            texto<<"Regla: IGUAL a "<<ui->Etiq_Ref->value();
        }
        if(ui->Regla->currentIndex()==1){
            multi.tipo_regla.push_back(DISTINTO);
            texto<<"Regla: DISTINTO a "<<ui->Etiq_Ref->value();
        }
        if(ui->Regla->currentIndex()==2){
            multi.tipo_regla.push_back(MAYOR);
            texto<<"Regla: MAYOR a "<<ui->Etiq_Ref->value();
        }
        if(ui->Regla->currentIndex()==3){
            multi.tipo_regla.push_back(MENOR);
            texto<<"Regla: MENOR a "<<ui->Etiq_Ref->value();
        }
        ui->Aceptar_2->setEnabled(false);
    }
    else if(ui->Votacion_2->isChecked()){
        multi.w_clasif.push_back(ui->Peso->value());
        texto<<"Peso="<<ui->Peso->value();
        ui->Aceptar_2->setEnabled(true);
    }
    texto<<endl;
    ui->Informacion->setText(QString::fromStdString(texto.str()));
    ui->Anadir->setEnabled(true);
    ui->Anadir_2->setEnabled(false);
}

void Conf_Multi::on_Cascada_2_clicked()
{
    texto.str("");
    ui->Informacion->clear();
    id_clasificadores.clear();
    ui->Anadir->setEnabled(true);
    ui->Anadir_2->setEnabled(false);
    multi.tipo=CASCADA;
    multi.label_ref.clear();
    multi.tipo_regla.clear();
    ui->Regla->setEnabled(true);
    ui->label_3->setEnabled(true);
    ui->Etiq_Ref->setEnabled(true);
    ui->label->setEnabled(false);
    ui->Peso->setEnabled(false);
}

void Conf_Multi::on_Votacion_2_clicked()
{
    texto.str("");
    ui->Informacion->clear();
    id_clasificadores.clear();
    ui->Anadir->setEnabled(true);
    ui->Anadir_2->setEnabled(false);
    multi.tipo=VOTACION;
    multi.w_clasif.clear();
    ui->Regla->setEnabled(false);
    ui->label_3->setEnabled(false);
    ui->Etiq_Ref->setEnabled(false);
    ui->label->setEnabled(true);
    ui->Peso->setEnabled(true);
}

void Conf_Multi::on_reset_2_clicked()
{
    texto.str("");
    ui->Informacion->clear();
    id_clasificadores.clear();
    ui->Anadir->setEnabled(true);
    ui->Anadir_2->setEnabled(false);
    multi.label_ref.clear();
    multi.tipo_regla.clear();
    ui->Regla->setEnabled(true);
    ui->label_3->setEnabled(true);
    ui->Etiq_Ref->setEnabled(true);
    ui->label->setEnabled(false);
    ui->Peso->setEnabled(false);
}

void Conf_Multi::on_Aceptar_2_clicked()
{
//    MainWindow *window=(MainWindow*) punt;
//    window->id_clasificadores=id_clasificadores;
//    window->Multi_tipo=multi;
//    delete this;
}

