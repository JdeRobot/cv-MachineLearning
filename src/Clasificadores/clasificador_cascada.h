#ifndef CLASIFICADOR_CASCADA_H
#define CLASIFICADOR_CASCADA_H

#include <dirent.h>
#include "clasificador.h"
#include <fstream>
#include <unistd.h>

using namespace std;
using namespace cv;

namespace MLT {
    class Clasificador_Cascada: public Clasificador{
    public:
        Clasificador_Cascada(string Nombre="", string FeatureType="HAAR", bool ejec_script=true, int NumPos=1000, int NumNeg=1000, string Mode="BASIC", int NumStage=10,float MinHitRate=0.995, float MaxFalseAlarmRate=0.5, float WeightTrimRate=0.95, int MaxWeakCount=2, int MaxDepth=1, string Bt="GAB", int PrecalcValBufSize=512, int PrecalcidxBufSize=512);
        ~Clasificador_Cascada();
        int Parametrizar(string FeatureType,bool ejec_script, int NumPos, int NumNeg, string Mode, int NumStage,float MinHitRate, float MaxFalseAlarmRate, float WeightTrimRate, int MaxWeakCount, int MaxDepth, string Bt, int PrecalcValBufSize, int PrecalcidxBufSize);
        int Autotrain(vector<Mat> Data, vector<float> Labels, Dimensionalidad::Reducciones reduc, Generacion::Info_Datos info, bool save=true);
        int Autoclasificacion(vector<Mat> Data, vector<float> &Labels, bool reducir, bool read);
        int Save_Data();
        int Read_Data();

        string featureType;
        int numPos;
        int numNeg;
        string mode;
        int numStages;
        float minHitRate;
        float maxFalseAlarmRate;
        float weightTrimRate;
        int maxWeakCount;
        int maxDepth;
        string bt;
        int precalcValBufSize;
        int precalcIdxBufSize;




    private:
        void Entrenamiento(Mat trainingDataMat, Mat labelsMat);
        float Clasificacion(Mat Data);

        cv::CascadeClassifier Cascade;
        bool ejecutar_script;
        Dimensionalidad::Reducciones reduccion;
    };
}

#endif // CLASIFICADOR_CASCADA_H
