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

#ifndef CLASIFICADOR_H
#define CLASIFICADOR_H

#include <opencv2/opencv.hpp>
#include "../Herramientas/auxiliares.h"
#include "../Herramientas/dimensionalidad.h"
#ifdef GUI
#include <ui_mainwindow.h>
#endif

namespace MLT {
    enum
    {
        DISTANCIAS          = 0,
        GAUSSIANO           = 1,
        CASCADA_CLAS        = 2,
        HISTOGRAMA          = 3,
        KNN                 = 4,
        NEURONAL            = 5,
        C_SVM               = 6,
        RTREES              = 7,
        DTREES              = 8,
        BOOSTING            = 9,
        EXP_MAX             = 11,
        MICLASIFICADOR      = 33,
        MULTICLASIFICADOR   = 100
    };

    class Clasificador
    {
    public:
        Clasificador() { }
        virtual ~Clasificador() {}

        int virtual Autotrain(vector<Mat> /*data*/, vector<float> /*labels*/, Dimensionalidad::Reducciones /*reduc*/, Generacion::Info_Datos /*info*/, bool /*save*/=true) = 0;
        int virtual Autoclasificacion(vector<Mat> /*data*/, vector<float>& /*labels*/, bool /*reducir*/, bool /*read*/) = 0;
        int virtual SaveData() = 0;
        int virtual ReadData() = 0;

        int getTipoClasificador() const { return tipoClasificador; }
        int getTipoDato() const { return tipoDato; }
        int getNumeroEtiquetas() const { return numeroEtiquetas; }

        int getVentanaX() const { return ventanaX; }
        int getVentanaY() const { return ventanaY; }
        int getVentanaOX() const { return ventanaOX; }
        int getVentanaOY() const { return ventanaOX; }

        string getNombre() const { return nombre; }

    private:
        void virtual Entrenamiento(Mat trainingDataMat, Mat labelsMat) = 0;
        float virtual Clasificacion(Mat Data) = 0;

    protected:
        int tipoClasificador;
        int tipoDato;
        int numeroEtiquetas;

        int ventanaX;
        int ventanaY;
        int ventanaOX;
        int ventanaOY;

        string nombre;

    #ifdef GUI
    public:
        int getProgreso() const { return _progreso; }
        void setProgreso(int progreso) { _progreso = progreso; }

        int getMaxProgreso() const { return _maxProgreso; }
        void setMaxProgreso(int maxProgreso) { _maxProgreso = maxProgreso; }

        int getBaseProgreso() const { return _baseProgreso; }
        void setBaseProgreso(int baseProgreso) { _baseProgreso = baseProgreso; }

        int getTotalProgreso() const { return _totalProgreso; }
        void setTotalProgreso(int totalProgreso) { _totalProgreso = totalProgreso; }

        Ui::MainWindow* getWindow() const { return _window; }
        void setWindow(Ui::MainWindow* window) { _window = window; }

    protected:
        int _progreso;
        int _maxProgreso;
        int _baseProgreso;
        int _totalProgreso;

        Ui::MainWindow *_window;
    #endif
    };
}

#endif // CLASIFICADOR_H
