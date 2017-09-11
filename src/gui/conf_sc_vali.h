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

#ifndef CONF_SC_VALI_H
#define CONF_SC_VALI_H

#include <QMainWindow>
#include "gui/mainwindow.h"

using namespace MLT;


namespace Ui {
class Conf_SC_Vali;
}

class Conf_SC_Vali : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit Conf_SC_Vali(void *puntero, QWidget *parent = 0);
    ~Conf_SC_Vali();
    
private slots:
    void on_Iniciar_2_clicked();

private:
    Ui::Conf_SC_Vali *ui;
    void *punt;
    vector<int> id_clasificadores;
};

#endif // CONF_SC_VALI_H
