#ifndef BUSQUEDA_H
#define BUSQUEDA_H

#include <opencv2/opencv.hpp>
#include "auxiliares.h"
#include "dimensionalidad.h"
#include "../Clasificadores/Clasificadores.h"
#include "../Clasificadores/multiclasificador.h"
#include "../Extraccion_Caracteristicas/Caracteristicas.h"
#include "../Clasificadores/miclasificador.h"

using namespace std;
using namespace cv;

namespace MLT {
    class Busqueda
    {
    public:
        Busqueda(Clasificador *clasificador,int Tipo_Descriptor, Descriptor *descriptor, MultiClasificador::Multi_type *multitipo=0);
        Busqueda(MultiClasificador *clasificador,int Tipo_Descriptor, Descriptor *descriptor, MultiClasificador::Multi_type *multitipo=0);
        int Textura(Mat src, Size tam_base, int escalas, int salto, int rotate, bool relleno, Mat &OUT);
        int Posicion(Mat src, Size tam_base, int escalas, int salto, int rotate, bool juntar_recuadros, bool solapamiento, bool aislamiento, float distancia_recuadros, int rotacion_recuadros, vector<RotatedRect> &recuadros, vector<float> &Labels);
        Descriptor *descrip;

        int error;

    private:
        Clasificador *clasificador;
        MultiClasificador *Multi;

        int tipo,tipo_dato;
        int numero_etiquetas,ventana_x,ventana_y;

        MultiClasificador::Multi_type Tipo_Multi;
    };
}

#endif // BUSQUEDA_H
