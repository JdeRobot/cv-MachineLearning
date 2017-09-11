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

#include "con_multi.h"
#include "ui_con_multi.h"

Con_Multi::Con_Multi(void *puntero, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Con_Multi)
{
    punt=puntero;
    texto.str("");
    id_clasificadores.clear();
    nombres.clear();
    multi.tipo=CASCADA;
    multi.label_ref.clear();
    multi.tipo_regla.clear();
    ui->setupUi(this);
    ui->Informacion->clear();
}

Con_Multi::~Con_Multi()
{
    delete ui;
}

void Con_Multi::on_Anadir_clicked()
{
    QString direccion=ui->Direccion_Carga->displayText();
    std::string Dir=direccion.toStdString();
    int pos=0;
    for(uint i=0; i<Dir.size(); i++){
        if(Dir[i]=='/')
            pos=i;
    }
    std::string nombre;
    for(uint i=pos+1; i<Dir.size(); i++)
        nombre=nombre+Dir[i];
    string archivo=Dir+"/Clasificador.xml";
    cv::FileStorage archivo_r(archivo,CV_STORAGE_READ);
    int id;
    if(archivo_r.isOpened()){
        archivo_r["Tipo"]>>id;
    }
    else{
        QMessageBox msgBox;
        msgBox.setText("ERROR: La carpeta no tiene la estructura utilizada por el sistema");
        msgBox.exec();
        return;
    }
    id_clasificadores.push_back(id);
    nombres.push_back(nombre);
    texto<<"Clasificador: "<<nombre<<"      Tipo: ";
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
    else if(id==EXP_MAX)
        texto<<"EXPECTATION MAXIMIZATION"<<endl;
//    else if(id==GBT)
//        texto<<"GBT"<<endl;
//    else if(id==ERTREES)
//        texto<<"ERTREES"<<endl;
    ui->Informacion->setText(QString::fromStdString(texto.str()));
    ui->Anadir_2->setEnabled(true);
    ui->Anadir->setEnabled(false);
    if(ui->Cascada->isChecked())
        ui->Aceptar->setEnabled(true);
    else if(ui->Votacion->isChecked())
        ui->Aceptar->setEnabled(false);
}

void Con_Multi::on_Anadir_2_clicked()
{
    if(ui->Cascada->isChecked()){
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
        ui->Aceptar->setEnabled(false);
    }
    else if(ui->Votacion->isChecked()){
        multi.w_clasif.push_back(ui->Peso->value());
        texto<<"Peso="<<ui->Peso->value();
        ui->Aceptar->setEnabled(true);
    }
    texto<<endl;
    ui->Informacion->setText(QString::fromStdString(texto.str()));
    ui->Anadir->setEnabled(true);
    ui->Anadir_2->setEnabled(false);
}

void Con_Multi::on_Cascada_clicked()
{
    texto.str("");
    ui->Informacion->clear();
    id_clasificadores.clear();
    nombres.clear();
    ui->Anadir->setEnabled(true);
    ui->Anadir_2->setEnabled(false);
    multi.tipo=CASCADA;
    multi.label_ref.clear();
    multi.tipo_regla.clear();
    ui->Regla->setEnabled(true);
    ui->label->setEnabled(true);
    ui->Etiq_Ref->setEnabled(true);
    ui->label_2->setEnabled(false);
    ui->Peso->setEnabled(false);
}

void Con_Multi::on_Votacion_clicked()
{
    texto.str("");
    ui->Informacion->clear();
    id_clasificadores.clear();
    nombres.clear();
    ui->Anadir->setEnabled(true);
    ui->Anadir_2->setEnabled(false);
    multi.tipo=VOTACION;
    multi.w_clasif.clear();
    ui->Regla->setEnabled(false);
    ui->label->setEnabled(false);
    ui->Etiq_Ref->setEnabled(false);
    ui->label_2->setEnabled(true);
    ui->Peso->setEnabled(true);
}

void Con_Multi::on_toolButton_clicked()
{
    QString filename= QFileDialog::getExistingDirectory(
                this,
                tr("SELECCIONAR CARPETA"),
                QDir::currentPath()+"/../Data/Configuracion");
    ui->Direccion_Carga->setText(filename);
}

void Con_Multi::on_reset_clicked()
{
    texto.str("");
    ui->Informacion->clear();
    id_clasificadores.clear();
    nombres.clear();
    ui->Anadir->setEnabled(true);
    ui->Anadir_2->setEnabled(false);
    multi.label_ref.clear();
    multi.tipo_regla.clear();
    ui->Regla->setEnabled(true);
    ui->label->setEnabled(true);
    ui->Etiq_Ref->setEnabled(true);
    ui->label_2->setEnabled(false);
    ui->Peso->setEnabled(false);
}

void Con_Multi::on_Aceptar_clicked()
{
    MainWindow *window=(MainWindow*) punt;
    window->id_clasificadores=id_clasificadores;
    window->nombres=nombres;
    window->Multi_tipo=multi;
    delete this;
}
