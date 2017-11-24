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

#ifndef CON_MULTI_H
#define CON_MULTI_H

#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>
#include "../Clasificadores/multiclasificador.h"
#include "mainwindow.h"

using namespace MLT;

namespace Ui {
class Con_Multi;
}

class Con_Multi : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit Con_Multi(void *puntero, QWidget *parent = 0);
    ~Con_Multi();
    
private slots:
    void on_Anadir_clicked();

    void on_Anadir_2_clicked();

    void on_Cascada_clicked();

    void on_Votacion_clicked();

    void on_toolButton_clicked();

    void on_reset_clicked();

    void on_Aceptar_clicked();

private:
    Ui::Con_Multi *ui;
    MultiClasificador::Multi_type multi;
    vector<int> id_clasificadores;
    vector<string> nombres;
    stringstream texto;
    void *punt;
};

#endif // CON_MULTI_H
