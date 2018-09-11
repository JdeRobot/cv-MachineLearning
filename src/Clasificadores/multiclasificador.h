#ifndef MULTICLASIFICADOR_H
#define MULTICLASIFICADOR_H

#include <dirent.h>
#include "Clasificadores.h"
#include <thread>


using namespace std;
using namespace cv;

namespace MLT {
    enum
    {
        CASCADA=0,
        VOTACION=1
    };

    enum
    {
        IGUAL=0,
        DISTINTO=1,
        MENOR=2,
        MAYOR=3
    };

    class MultiClasificador{
    public:
        struct Multi_type {
           int tipo;
           vector<int> identificadores;
           vector<string> nombres;
           vector<int> tipo_regla;
           vector<float> label_ref;
           vector<float> w_clasif;

           Multi_type(){
               tipo=CASCADA;
               identificadores.clear();
               nombres.clear();
               tipo_regla.clear();
               label_ref.clear();
               w_clasif.clear();
           }
        };


        MultiClasificador(vector<Clasificador*> Clasificadores);
        int Cascada(vector<Mat> Data, vector<int> tipo_regla, vector<float> labels_ref, vector<float> &Labels);
        int Votacion(vector<Mat> Data, vector<float> w_clasif, vector<float> &Labels);

        int numero_etiquetas,ventana_x,ventana_y,tipo_dato;

#ifdef GUI
    int progreso;
    int total_progreso;
    int error;
    bool running;
#endif

    private:
        void update_progress();
        bool classifier_running;

        vector<Clasificador*> clasificadores;
        vector<int> ventanas_x,ventanas_y,n_etiquetas,tipos_dato;

        bool Error;
    };
}

#endif // MULTICLASIFICADOR_H
