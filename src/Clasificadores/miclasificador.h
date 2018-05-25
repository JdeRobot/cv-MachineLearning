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

#ifndef MICLASIFICADOR_H
#define MICLASIFICADOR_H

#include "../Herramientas/dimensionalidad.h"
#include "../Extraccion_Caracteristicas/Caracteristicas.h"
#include "Clasificadores.h"


namespace MLT{
    class MiClasificador: public Clasificador{
    public:
        MiClasificador(string Nombre="");
        ~MiClasificador();

        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true) override;
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read) override;
        int SaveData() override;
        int ReadData() override;

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat) override;
        float Clasificacion(Mat Data) override;
        Size tam_imagen;

    /*******************************************************************************************/
        // Pon aqui los clasificadores que utilizarás
        // Ejemplo:
        // Clasificador clasif;
        // Clasificador_ERTrees ERTREES;
        // Clasificador_SVM SVM;
    /*****************************************************************************************/

    };
}

#endif // MICLASIFICADOR_H
