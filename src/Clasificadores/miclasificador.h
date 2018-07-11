#ifndef MICLASIFICADOR_H
#define MICLASIFICADOR_H

#include "clasificador.h"
#include "../Extraccion_Caracteristicas/Caracteristicas.h"
#include "Clasificadores.h"


namespace MLT{
    class MiClasificador: public Clasificador{
    public:
        MiClasificador(string Nombre="");
        ~MiClasificador();
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);
        Size tam_imagen;

    /*******************************************************************************************/
        //Pon aqui los clasificadores que utilizarás
        //Ejemplo:
        //Clasificador clasif;
//        Clasificador_ERTrees ERTREES;
//        Clasificador_SVM SVM;

    /*****************************************************************************************/

    };
}

#endif // MICLASIFICADOR_H
