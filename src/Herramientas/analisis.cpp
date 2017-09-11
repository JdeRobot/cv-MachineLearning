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

#include "analisis.h"

bool ord (MLT::Analisis::Ratios_data A, MLT::Analisis::Ratios_data B) {
    return A.valor_x<B.valor_x;
}

MLT::Analisis::Analisis(){}

int MLT::Analisis::Confusion(std::vector<float> Etiquetas, std::vector<float> Resultados, Mat &Confusion, float &error){
    error=0;
    if(Etiquetas.size()!=Resultados.size()){
        cout<<"ERROR en Confusion: Tamaño de Etiquetas y Resultados distintos"<<endl;
        return 1;
    }
    for(uint i=0; i<Resultados.size(); i++){
#ifdef WARNINGS
        if(Resultados[i]==0){
            cout<<"WARNING en Confusion: Etiquetas con valor igual 0"<<endl;
        }
#endif
        if(Resultados[i]<-1){
            cout<<"ERROR en Confusion: Etiquetas con valor menor -1"<<endl;
            return 1;
        }
    }
#ifdef WARNINGS
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0)
            cout<<"WARNING en Confusion: Etiquetas con valor 0"<<endl;
    }
#endif
    Auxiliares ax;
    bool negativa;
    int num_etiq=ax.numero_etiquetas(Etiquetas,negativa);

    error=0;
    cv::Mat conf=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]!=0 && Resultados[i]!=0){
            if(Etiquetas[i]!=Resultados[i]){
                error=error+1;
            }
            int x=Resultados[i];
            int y=Etiquetas[i];
            if(negativa){
                if(x==-1)
                    x=x+1;
                if(y==-1)
                    y=y+1;
            }
            else{
                x=x-1;
                y=y-1;
            }
            if(x<0){
                cout<<"ERROR en Confusion: La configuracion de etiquetado en Etiquetas y Resultados es distinta"<<endl;
                return 1;
            }
            conf.row(x).col(y)=conf.row(x).col(y)+1;
        }
    }
    error=error/Etiquetas.size();
    conf.copyTo(Confusion);
    return 0;
}

int MLT::Analisis::Ratios(std::vector<float> Etiquetas, std::vector<float> Resultados, vector<Ratios_data> &Rat){
    Rat.clear();
    if(Etiquetas.size()!=Resultados.size()){
        cout<<"ERROR en Ratios: Tamaño de Etiquetas y Resultados distintos"<<endl;
        return 1;
    }
    for(uint i=0; i<Resultados.size(); i++){
#ifdef WARNINGS
        if(Resultados[i]==0)
            cout<<"WARNING en Ratios: Etiquetas con valor igual 0"<<endl;
#endif
        if(Resultados[i]<-1){
            cout<<"ERROR en Ratios: Etiquetas con valor menor -1"<<endl;
            return 1;
        }
    }
#ifdef WARNINGS
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0)
            cout<<"WARING en Ratios: Etiquetas con valor 0"<<endl;
    }
#endif
    Auxiliares ax;
    bool negativa;
    int num_etiq=ax.numero_etiquetas(Etiquetas,negativa);
    for(int i=0; i<num_etiq; i++){
        float ref=0;
        if(negativa==true){
            if(i==0)
                ref=-1.0;
            else
                ref=i;
        }
        else
            ref=i+1;
        Ratios_data datos;
        datos.valor_x=ref;
        for(uint j=0; j<Etiquetas.size(); j++){
            if(Etiquetas[j]!=0 && Resultados[j]!=0){
                if(Etiquetas[j]==ref){
                    if(Etiquetas[j]==Resultados[j])
                        datos.VP=datos.VP+1;
                    if(Etiquetas[j]!=Resultados[j])
                        datos.FN=datos.FN+1;
                }
                else if(Resultados[j]==ref){
                    datos.FP=datos.FP+1;
                }
                else if(Etiquetas[j]!=ref && Resultados[j]!=ref)
                    datos.VN=datos.VN+1;
            }
        }
        if(datos.FP+datos.VN==0)
            datos.FAR=0;
        else
            datos.FAR=datos.FP/(datos.FP+datos.VN);
        if(datos.FN+datos.VP==0)
            datos.FRR=0;
        else
            datos.FRR=datos.FN/(datos.FN+datos.VP);
        datos.TAR=1-datos.FRR;
        datos.TRR=1-datos.FAR;
        if((datos.VP+datos.FP)!=0)
            datos.PPV=datos.VP/(datos.VP+datos.FP);
        if((datos.VN+datos.FN)!=0)
            datos.NPV=datos.VN/(datos.VN+datos.FN);
        datos.FDR=1-datos.PPV;
        datos.F1=(2*datos.VP)/(2*datos.VP+datos.FP+datos.FN);
        datos.INFORMEDNESS=datos.TAR+datos.TAR-1;
        datos.MARKEDNESS=datos.PPV+datos.NPV-1;
        datos.EXP_ERROR=0.5*datos.FAR+0.5*datos.FRR;
        if(datos.FAR!=0)
            datos.LR_POS=datos.TAR/datos.FAR;
        if(datos.TRR!=0)
            datos.LR_NEG=datos.FRR/datos.TRR;
        if(datos.LR_NEG!=0)
            datos.DOR=datos.LR_POS/datos.LR_NEG;
        datos.ACC=(datos.VP+datos.VN)/(datos.VP+datos.VN+datos.FN+datos.FP);
        datos.PREVALENCE=(datos.VP+datos.FN)/(datos.VP+datos.VN+datos.FN+datos.FP);
        Rat.push_back(datos);
    }
    return 0;
}

int MLT::Analisis::Ratios_Histograma(std::vector<Mat> Datos, std::vector<float> Etiquetas, std::vector<float> Resultados, int num_barras, vector<vector<Ratios_data> > &Hist_Rat){
    int e=0;
    Hist_Rat.clear();
    if(Datos.empty()){
        cout<<"ERROR en Ratios_Histograma: Tamaño de Datos igual a cero"<<endl;
        return 1;
    }
    if(Datos.size()!=Etiquetas.size()){
        cout<<"ERROR en Ratios_Histograma: Tamaño de Datos y Etiquetas distinto"<<endl;
        return 1;
    }
    if(Etiquetas.size()!=Resultados.size()){
        cout<<"ERROR en Ratios_Histograma: Tamaño de Etiquetas y Resultados distinto"<<endl;
        return 1;
    }
    for(uint i=0; i<Resultados.size(); i++){
#ifdef WARNINGS
        if(Resultados[i]==0)
            cout<<"WARNING en Ratios_Histograma: Etiquetas con valor igual 0"<<endl;
#endif
        if(Resultados[i]<-1){
            cout<<"ERROR en Ratios_Histograma: Etiquetas con valor menor -1"<<endl;
            return 1;
        }
    }
#ifdef WARNINGS
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0)
            cout<<"WARNING en Ratios_Histograma: Etiquetas con valor 0"<<endl;
    }
#endif
    vector<vector<Mat> > Hist;
    vector<vector<int> > pos_bar;
    e=Histograma(Datos,Etiquetas,num_barras,Hist,pos_bar);
    if(e==1){
        cout<<"ERROR en Ratios_Histograma: Error en Histograma"<<endl;
        return 1;
    }
    Auxiliares ax;
    bool negativa;
    int num_etiq=ax.numero_etiquetas(Etiquetas,negativa);
    vector<vector<Ratios_data> > Rat;
    for(int j=0; j<Datos[0].rows*Datos[0].cols; j++){
        vector<Ratios_data> rat;
        for(int i=0; i<num_barras*num_etiq; i++){
            Ratios_data datos;
            for(int k=0; k<num_etiq; k++){
                float ref=0;
                if(negativa==true){
                    if(k==0)
                        ref=-1.0;
                    else
                        ref=k;
                }
                else
                    ref=k+1;
                for(uint l=0; l<Etiquetas.size(); l++){
                    if(Etiquetas[l]!=0 && Resultados[l]!=0){
                        if(pos_bar[l][j]==i){
                            if(Etiquetas[l]==ref){
                                if(Etiquetas[l]==Resultados[l])
                                    datos.VP=datos.VP+1;
                                if(Etiquetas[l]!=Resultados[l])
                                    datos.FN=datos.FN+1;
                            }
                            else if(Resultados[l]==ref){
                                datos.FP=datos.FP+1;
                            }
                            else if(Etiquetas[l]!=ref && Resultados[l]!=ref)
                                datos.VN=datos.VN+1;
                        }
                    }
                }
            }
            if(datos.FP+datos.VN==0)
                datos.FAR=0;
            else
                datos.FAR=(float)datos.FP/(float)(datos.FP+datos.VN);
            if(datos.FN+datos.VP==0)
                datos.FRR=0;
            else
                datos.FRR=(float)datos.FN/(float)(datos.FN+datos.VP);
            datos.TAR=1-datos.FRR;
            datos.TRR=1-datos.FAR;
            if((datos.VP+datos.FP)!=0)
                datos.PPV=datos.VP/(datos.VP+datos.FP);
            if((datos.VN+datos.FN)!=0)
                datos.NPV=datos.VN/(datos.VN+datos.FN);
            datos.FDR=1-datos.PPV;
            datos.F1=(2*datos.VP)/(2*datos.VP+datos.FP+datos.FN);
            datos.INFORMEDNESS=datos.TAR+datos.TAR-1;
            datos.MARKEDNESS=datos.PPV+datos.NPV-1;
            datos.EXP_ERROR=0.5*datos.FAR+0.5*datos.FRR;
            if(datos.FAR!=0)
                datos.LR_POS=datos.TAR/datos.FAR;
            if(datos.TRR!=0)
                datos.LR_NEG=datos.FRR/datos.TRR;
            if(datos.LR_NEG!=0)
                datos.DOR=datos.LR_POS/datos.LR_NEG;
            datos.ACC=(datos.VP+datos.VN)/(datos.VP+datos.VN+datos.FN+datos.FP);
            datos.PREVALENCE=(datos.VP+datos.FN)/(datos.VP+datos.VN+datos.FN+datos.FP);
            rat.push_back(datos);
        }
        Rat.push_back(rat);
    }


    for(uint i=0; i<Rat.size();i++){
        vector<Ratios_data> Ratios_Barras;
        for(uint j=0; j<Rat[i].size(); j++){
            int pos_etiqueta=(int)floor(j/num_barras);
            int bar=j-(pos_etiqueta*num_barras);
            float valor_barra=Hist[pos_etiqueta][i].at<float>(0,bar);
            Ratios_data ratios_barra;
            ratios_barra.valor_x=valor_barra;
            ratios_barra.VP=Rat[i][j].VP;
            ratios_barra.VN=Rat[i][j].VN;
            ratios_barra.FN=Rat[i][j].FN;
            ratios_barra.FP=Rat[i][j].FP;
            ratios_barra.FAR=Rat[i][j].FAR;
            ratios_barra.FRR=Rat[i][j].FRR;
            ratios_barra.TAR=Rat[i][j].TAR;
            ratios_barra.TRR=Rat[i][j].TRR;
            ratios_barra.PPV=Rat[i][j].PPV;
            ratios_barra.NPV=Rat[i][j].NPV;
            ratios_barra.FDR=Rat[i][j].FDR;
            ratios_barra.F1=Rat[i][j].F1;
            ratios_barra.INFORMEDNESS=Rat[i][j].INFORMEDNESS;
            ratios_barra.MARKEDNESS=Rat[i][j].MARKEDNESS;
            ratios_barra.EXP_ERROR=Rat[i][j].EXP_ERROR;
            ratios_barra.LR_NEG=Rat[i][j].LR_NEG;
            ratios_barra.LR_POS=Rat[i][j].LR_POS;
            ratios_barra.DOR=Rat[i][j].DOR;
            ratios_barra.ACC=Rat[i][j].ACC;
            ratios_barra.PREVALENCE=Rat[i][j].PREVALENCE;
            Ratios_Barras.push_back(ratios_barra);
        }
        sort(Ratios_Barras.begin(), Ratios_Barras.end(), ord);
        vector<bool> iguales;
        for(uint j=1; j<Rat[i].size(); j++){
            if(Ratios_Barras[j-1].valor_x==Ratios_Barras[j].valor_x){
                iguales.push_back(true);
            }
            else
                iguales.push_back(false);
        }
        Ratios_data conta;
        conta.VP=Ratios_Barras[0].VP;
        conta.VN=Ratios_Barras[0].VN;
        conta.FN=Ratios_Barras[0].FN;
        conta.FP=Ratios_Barras[0].FP;
        conta.FAR=Ratios_Barras[0].FAR;
        conta.FRR=Ratios_Barras[0].FRR;
        conta.TAR=Ratios_Barras[0].TAR;
        conta.TRR=Ratios_Barras[0].TRR;
        conta.PPV=Ratios_Barras[0].PPV;
        conta.NPV=Ratios_Barras[0].NPV;
        conta.FDR=Ratios_Barras[0].FDR;
        conta.F1=Ratios_Barras[0].F1;
        conta.INFORMEDNESS=Ratios_Barras[0].INFORMEDNESS;
        conta.MARKEDNESS=Ratios_Barras[0].MARKEDNESS;
        conta.EXP_ERROR=Ratios_Barras[0].EXP_ERROR;
        conta.LR_NEG=Ratios_Barras[0].LR_NEG;
        conta.LR_POS=Ratios_Barras[0].LR_POS;
        conta.DOR=Ratios_Barras[0].DOR;
        conta.ACC=Ratios_Barras[0].ACC;
        conta.PREVALENCE=Ratios_Barras[0].PREVALENCE;
        vector<Ratios_data> Ratios_Barras_mej;
        for(uint j=0; j<iguales.size(); j++){
            if(iguales[j]==true){
                conta.VP=conta.VP+Ratios_Barras[j+1].VP;
                conta.VN=conta.VN+Ratios_Barras[j+1].VN;
                conta.FN=conta.FN+Ratios_Barras[j+1].FN;
                conta.FP=conta.FP+Ratios_Barras[j+1].FP;
                conta.FAR=conta.FAR+Ratios_Barras[j+1].FAR;
                conta.FRR=conta.FRR+Ratios_Barras[j+1].FRR;
                conta.TAR=conta.TAR+Ratios_Barras[j+1].TAR;
                conta.TRR=conta.TRR+Ratios_Barras[j+1].TRR;
                conta.PPV=conta.PPV+Ratios_Barras[j+1].PPV;
                conta.NPV=conta.NPV+Ratios_Barras[j+1].NPV;
                conta.FDR=conta.FDR+Ratios_Barras[j+1].FDR;
                conta.F1=conta.F1+Ratios_Barras[j+1].F1;
                conta.INFORMEDNESS=conta.INFORMEDNESS+Ratios_Barras[j+1].INFORMEDNESS;
                conta.MARKEDNESS=conta.MARKEDNESS+Ratios_Barras[j+1].MARKEDNESS;
                conta.EXP_ERROR=conta.EXP_ERROR+Ratios_Barras[j+1].EXP_ERROR;
                conta.LR_NEG=conta.LR_NEG+Ratios_Barras[j+1].LR_NEG;
                conta.LR_POS=conta.LR_POS+Ratios_Barras[j+1].LR_POS;
                conta.DOR=conta.DOR+Ratios_Barras[j+1].DOR;
                conta.ACC=conta.ACC+Ratios_Barras[j+1].ACC;
                conta.PREVALENCE=conta.PREVALENCE+Ratios_Barras[j+1].PREVALENCE;
            }
            else{
                conta.valor_x=Ratios_Barras[j].valor_x;
                conta.VP=Ratios_Barras[j+1].VP;
                conta.VN=Ratios_Barras[j+1].VN;
                conta.FN=Ratios_Barras[j+1].FN;
                conta.FP=Ratios_Barras[j+1].FP;
                conta.FAR=Ratios_Barras[j+1].FAR;
                conta.FRR=Ratios_Barras[j+1].FRR;
                conta.TAR=Ratios_Barras[j+1].TAR;
                conta.TRR=Ratios_Barras[j+1].TRR;
                conta.PPV=Ratios_Barras[j+1].PPV;
                conta.NPV=Ratios_Barras[j+1].NPV;
                conta.FDR=Ratios_Barras[j+1].FDR;
                conta.F1=Ratios_Barras[j+1].F1;
                conta.INFORMEDNESS=Ratios_Barras[j+1].INFORMEDNESS;
                conta.MARKEDNESS=Ratios_Barras[j+1].MARKEDNESS;
                conta.EXP_ERROR=Ratios_Barras[j+1].EXP_ERROR;
                conta.LR_NEG=Ratios_Barras[j+1].LR_NEG;
                conta.LR_POS=Ratios_Barras[j+1].LR_POS;
                conta.DOR=Ratios_Barras[j+1].DOR;
                conta.ACC=Ratios_Barras[j+1].ACC;
                conta.PREVALENCE=Ratios_Barras[j+1].PREVALENCE;
                Ratios_Barras_mej.push_back(conta);
            }
        }
        Hist_Rat.push_back(Ratios_Barras_mej);
    }
    return 0;
}

int MLT::Analisis::Estadisticos(vector<Mat> Datos, vector<float> Etiquetas, vector<Mat> &Medias, vector<Mat> &Des_Tipics, vector<vector<Mat> > &D_prime){
    if(Datos.size()==0){
        cout<<"ERROR en Estadisticos: No hay datos"<<endl;
        return 1;
    }
    if(Etiquetas.size()==0){
        cout<<"ERROR en Estadisticos: No hay Etiquetas"<<endl;
        return 1;
    }
    if(Datos.size()!=Etiquetas.size()){
        cout<<"ERROR en Estadisticos: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
#ifdef WARNINGS
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0)
            cout<<"WARNING en Estadisticos: Etiquetas con valor 0"<<endl;
    }
#endif
    vector<Mat> Varianzas;
    Auxiliares ax;
    Mat data;
    ax.Image2Lexic(Datos,data);
    bool neg;
    int num_etiq=ax.numero_etiquetas(Etiquetas,neg);
    for(int i=0; i<num_etiq; i++){
        Mat med=Mat::zeros(1,data.cols,CV_32FC1);
        Mat var=Mat::zeros(1,data.cols,CV_32FC1);
        Medias.push_back(med);
        Varianzas.push_back(var);
    }
    vector<int> num(num_etiq);
    for(int i=0; i<num_etiq; i++)
        num[i]=0;
    for(int i=0; i<data.rows; i++){
        if(Etiquetas[i]!=0){
            if(neg && Etiquetas[i]==-1){
                Medias[0]=Medias[0]+data.row(i);
                num[0]=num[0]+1;
            }
            else if(neg && Etiquetas[i]!=-1){
                Medias[Etiquetas[i]]=Medias[Etiquetas[i]]+data.row(i);
                num[Etiquetas[i]]=num[Etiquetas[i]]+1;
            }
            else if(!neg){
                Medias[Etiquetas[i]-1]=Medias[Etiquetas[i]-1]+data.row(i);
                num[Etiquetas[i]-1]=num[Etiquetas[i]-1]+1;
            }
        }
    }
    for(uint i=0; i<Medias.size(); i++)
        Medias[i]=Medias[i]/num[i];

    for(int i=0; i<data.rows; i++){
        if(Etiquetas[i]!=0){
            if(neg && Etiquetas[i]==-1){
                Mat A=data.row(i)-Medias[0];
                Varianzas[0]=Varianzas[0]+A.mul(A);
            }
            else if(neg && Etiquetas[i]!=-1){
                Mat A=data.row(i)-Medias[Etiquetas[i]];
                Varianzas[Etiquetas[i]]=Varianzas[Etiquetas[i]]+A.mul(A);
            }
            else if(!neg){
                Mat A=data.row(i)-Medias[Etiquetas[i]-1];
                Varianzas[Etiquetas[i]-1]=Varianzas[Etiquetas[i]-1]+A.mul(A);
            }
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        Varianzas[i]=Varianzas[i]/num[i];
        Des_Tipics.push_back(Varianzas[i]);
        for(int j=0; j<Varianzas[i].cols; j++){
            Des_Tipics[i].at<float>(0,j)=sqrt(Varianzas[i].at<float>(0,j));
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        vector<Mat> D_Prime;
        for(uint j=0; j<Medias.size(); j++){
            Mat d_prime=Mat::zeros(1,data.cols,CV_32FC1);
            if(i==j){
                D_Prime.push_back(d_prime);
            }
            else{
                for(int k=0; k<d_prime.cols; k++){
                    d_prime.at<float>(0,k)=abs((Medias[i].at<float>(0,k)-Medias[j].at<float>(0,k))/sqrt(Varianzas[i].at<float>(0,k)+Varianzas[j].at<float>(0,k)));
                }
                D_Prime.push_back(d_prime);
            }
        }
        D_prime.push_back(D_Prime);
    }
    return 0;
}

int MLT::Analisis::Estadisticos(Mat Datos, vector<float> Etiquetas, vector<Mat> &Medias, vector<Mat> &Des_Tipics, vector<vector<Mat> > &D_prime){
    if(Datos.empty()){
        cout<<"ERROR en Estadisticos: No hay datos"<<endl;
        return 1;
    }
    if(Etiquetas.size()==0){
        cout<<"ERROR en Estadisticos: No hay Etiquetas"<<endl;
        return 1;
    }
    if((uint)Datos.rows!=Etiquetas.size()){
        cout<<"ERROR en Estadisticos: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
#ifdef WARNINGS
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0)
            cout<<"WARNING en Estadisticos: Etiquetas con valor 0"<<endl;
    }
#endif
    vector<Mat> Varianzas;
    Auxiliares ax;
    Mat data;
    Datos.copyTo(data);
    bool neg;
    int num_etiq=ax.numero_etiquetas(Etiquetas,neg);
    for(int i=0; i<num_etiq; i++){
        Mat med=Mat::zeros(1,data.cols,CV_32FC1);
        Mat var=Mat::zeros(1,data.cols,CV_32FC1);
        Medias.push_back(med);
        Varianzas.push_back(var);
    }
    vector<int> num(num_etiq);
    for(int i=0; i<num_etiq; i++)
        num[i]=0;
    for(int i=0; i<data.rows; i++){
        if(Etiquetas[i]!=0){
            if(neg && Etiquetas[i]==-1){
                Medias[0]=Medias[0]+data.row(i);
                num[0]=num[0]+1;
            }
            else if(neg && Etiquetas[i]!=-1){
                Medias[Etiquetas[i]]=Medias[Etiquetas[i]]+data.row(i);
                num[Etiquetas[i]]=num[Etiquetas[i]]+1;
            }
            else if(!neg){
                Medias[Etiquetas[i]-1]=Medias[Etiquetas[i]-1]+data.row(i);
                num[Etiquetas[i]-1]=num[Etiquetas[i]-1]+1;
            }
        }
    }
    for(uint i=0; i<Medias.size(); i++)
        Medias[i]=Medias[i]/num[i];
    for(int i=0; i<data.rows; i++){
        if(Etiquetas[i]!=0){
            if(neg && Etiquetas[i]==-1){
                Mat A=data.row(i)-Medias[0];
                Varianzas[0]=Varianzas[0]+A.mul(A);
            }
            else if(neg && Etiquetas[i]!=-1){
                Mat A=data.row(i)-Medias[Etiquetas[i]];
                Varianzas[Etiquetas[i]]=Varianzas[Etiquetas[i]]+A.mul(A);
            }
            else if(!neg){
                Mat A=data.row(i)-Medias[Etiquetas[i]-1];
                Varianzas[Etiquetas[i]-1]=Varianzas[Etiquetas[i]-1]+A.mul(A);
            }
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        Varianzas[i]=Varianzas[i]/num[i];
        Des_Tipics.push_back(Varianzas[i]);
        for(int j=0; j<Varianzas[i].cols; j++){
            Des_Tipics[i].at<float>(0,j)=sqrt(Varianzas[i].at<float>(0,j));
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        vector<Mat> D_Prime;
        for(uint j=0; j<Medias.size(); j++){
            Mat d_prime=Mat::zeros(1,data.cols,CV_32FC1);
            if(i==j){
                D_Prime.push_back(d_prime);
            }
            else{
                for(int k=0; k<d_prime.cols; k++){
                    d_prime.at<float>(0,k)=abs((Medias[i].at<float>(0,k)-Medias[j].at<float>(0,k))/sqrt(Varianzas[i].at<float>(0,k)+Varianzas[j].at<float>(0,k)));
                }
                D_Prime.push_back(d_prime);
            }
        }
        D_prime.push_back(D_Prime);
    }
    return 0;
}

int MLT::Analisis::Covarianza(vector<Mat> Datos, vector<float> Etiquetas, vector<Mat> &Covarianzas){
    if(Datos.size()==0){
        cout<<"ERROR en Covarianza: No hay datos"<<endl;
        return 1;
    }
    if(Etiquetas.size()==0){
        cout<<"ERROR en Covarianza: No hay Etiquetas"<<endl;
        return 1;
    }
    if(Datos.size()!=Etiquetas.size()){
        cout<<"ERROR en Covarianza: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
#ifdef WARNINGS
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0)
            cout<<"WARNING en Covarianza: Etiquetas con valor 0"<<endl;
    }
#endif
    Auxiliares ax;
    Mat data;
    ax.Image2Lexic(Datos,data);
    data.convertTo(data,CV_32FC1);
    bool neg;
    int num_etiq=ax.numero_etiquetas(Etiquetas,neg);
    vector<int> num(num_etiq);
    for(uint i=0; i<num.size(); i++)
        num[i]=0;
    for(int i=0; i<data.rows; i++){
        if(Etiquetas[i]!=0){
            if(neg && Etiquetas[i]==-1)
                num[0]++;
            else if(neg && Etiquetas[i]!=-1)
                num[Etiquetas[i]]++;
            else
                num[Etiquetas[i]-1]++;
        }
    }
    vector<Mat> dat_label;
    for(int i=0; i<num_etiq; i++){
        Mat Zeros=Mat::zeros(num[i],data.cols,CV_32FC1);
        dat_label.push_back(Zeros);
    }
    vector<Mat> Medias;
    for(int i=0; i<num_etiq; i++){
        Mat Zeros=Mat::zeros(1,data.cols,CV_32FC1);
        Medias.push_back(Zeros);
    }
    vector<int> cont(num_etiq);
    for(int i=0; i<num_etiq; i++){
        cont[i]=0;
    }
    for(int i=0; i<data.rows; i++){
        if(Etiquetas[i]!=0){
            if(neg && Etiquetas[i]==-1){
                data.row(i).copyTo(dat_label[0].row(cont[0]));
                Medias[0]=Medias[0]+data.row(i);
                cont[0]++;
            }
            else if(neg && Etiquetas[i]!=-1){
                data.row(i).copyTo(dat_label[Etiquetas[i]].row(cont[Etiquetas[i]]));
                Medias[Etiquetas[i]]=Medias[Etiquetas[i]]+data.row(i);
                cont[Etiquetas[i]]++;
            }
            else{
                data.row(i).copyTo(dat_label[Etiquetas[i]-1].row(cont[Etiquetas[i]-1]));
                Medias[Etiquetas[i]-1]=Medias[Etiquetas[i]-1]+data.row(i);
                cont[Etiquetas[i]-1]++;
            }
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        Medias[i]=Medias[i]/num[i];
    }
    for(uint i=0; i<dat_label.size(); i++){
        for(int j=0; j<dat_label[i].rows; j++){
            dat_label[i].row(j)=dat_label[i].row(j)-Medias[i].row(0);
        }
        Mat aux=(1/(float)num[i])*dat_label[i].t()*dat_label[i];
        Covarianzas.push_back(aux);
    }
    return 0;
}

int MLT::Analisis::Covarianza(Mat Datos, vector<float> Etiquetas, vector<Mat> &Covarianzas){
    if(Datos.empty()){
        cout<<"ERROR en Covarianza: No hay datos"<<endl;
        return 1;
    }
    if(Etiquetas.size()==0){
        cout<<"ERROR en Covarianza: No hay Etiquetas"<<endl;
        return 1;
    }
    if((uint)Datos.rows!=Etiquetas.size()){
        cout<<"ERROR en Covarianza: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
#ifdef WARINGS
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0)
            cout<<"WARNING en Covarianza: Etiquetas con valor 0"<<endl;
    }
#endif
    Auxiliares ax;
    Mat data;
    Datos.copyTo(data);
    data.convertTo(data,CV_32FC1);
    bool neg;
    int num_etiq=ax.numero_etiquetas(Etiquetas,neg);
    vector<int> num(num_etiq);
    for(uint i=0; i<num.size(); i++)
        num[i]=0;
    for(int i=0; i<data.rows; i++){
        if(Etiquetas[i]!=0){
            if(neg && Etiquetas[i]==-1)
                num[0]++;
            else if(neg && Etiquetas[i]!=-1)
                num[Etiquetas[i]]++;
            else
                num[Etiquetas[i]-1]++;
        }
    }
    vector<Mat> dat_label;
    for(int i=0; i<num_etiq; i++){
        Mat Zeros=Mat::zeros(num[i],data.cols,CV_32FC1);
        dat_label.push_back(Zeros);
    }
    vector<Mat> Medias;
    for(int i=0; i<num_etiq; i++){
        Mat Zeros=Mat::zeros(1,data.cols,CV_32FC1);
        Medias.push_back(Zeros);
    }
    vector<int> cont(num_etiq);
    for(int i=0; i<num_etiq; i++){
        cont[i]=0;
    }
    for(int i=0; i<data.rows; i++){
        if(Etiquetas[i]!=0){
            if(neg && Etiquetas[i]==-1){
                data.row(i).copyTo(dat_label[0].row(cont[0]));
                Medias[0]=Medias[0]+data.row(i);
                cont[0]++;
            }
            else if(neg && Etiquetas[i]!=-1){
                data.row(i).copyTo(dat_label[Etiquetas[i]].row(cont[Etiquetas[i]]));
                Medias[Etiquetas[i]]=Medias[Etiquetas[i]]+data.row(i);
                cont[Etiquetas[i]]++;
            }
            else{
                data.row(i).copyTo(dat_label[Etiquetas[i]-1].row(cont[Etiquetas[i]-1]));
                Medias[Etiquetas[i]-1]=Medias[Etiquetas[i]-1]+data.row(i);
                cont[Etiquetas[i]-1]++;
            }
        }
    }
    for(uint i=0; i<Medias.size(); i++){
        Medias[i]=Medias[i]/num[i];
    }
    for(uint i=0; i<dat_label.size(); i++){
        for(int j=0; j<dat_label[i].rows; j++){
            dat_label[i].row(j)=dat_label[i].row(j)-Medias[i].row(0);
        }
        Mat aux=(1/(float)num[i])*dat_label[i].t()*dat_label[i];
        Covarianzas.push_back(aux);
    }
    return 0;
}

int MLT::Analisis::Histograma(vector<Mat> Datos, vector<float> Etiquetas, int Num_Barras, vector<vector<Mat> > &His, vector<vector<int> > &pos_barra){
    if(Datos.size()==0){
        cout<<"ERROR en Histograma: No hay datos"<<endl;
        return 1;
    }
    if(Etiquetas.size()==0){
        cout<<"ERROR en Histograma: No hay Etiquetas"<<endl;
        return 1;
    }
    if(Datos.size()!=Etiquetas.size()){
        cout<<"ERROR en Histograma: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
#ifdef WARNINGS
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0)
            cout<<"WARNING en Histograma: Etiquetas con valor 0"<<endl;
    }
#endif
    Auxiliares aux;
    bool negativa;
    int num=aux.numero_etiquetas(Etiquetas,negativa);
    Mat Dat;
    aux.Image2Lexic(Datos,Dat);
    for(int i=0; i<num; i++){
        vector<Mat> Hist;
        for(int j=0; j<Dat.cols; j++){
            Mat zer=Mat::zeros(2,Num_Barras,CV_32FC1);
            Hist.push_back(zer);
        }
        His.push_back(Hist);
    }
    pos_barra.clear();
    for(int i=0; i<Dat.rows; i++){
        vector<int> dimensiones;
        for(int j=0; j<Dat.cols; j++){
            dimensiones.push_back(0);
        }
        pos_barra.push_back(dimensiones);
    }
    for(int i=0; i<num; i++){
        int etiqueta=0;
        if(negativa && i==0)
            etiqueta=-1.0;
        else if(negativa && i>0)
            etiqueta=i;
        else if(negativa==false)
            etiqueta=i+1;
        Mat Max=Mat::zeros(1,Dat.cols,CV_32FC1);
        Mat Min=Mat::zeros(1,Dat.cols,CV_32FC1)+999999;
        for(int j=0; j<Dat.rows;j++){
            if(Etiquetas[j]!=0){
                if(Etiquetas[j]==etiqueta){
                    for(int k=0; k<Dat.cols; k++){
                        if(Max.at<float>(0,k)<Dat.at<float>(j,k))
                            Max.at<float>(0,k)=Dat.at<float>(j,k);
                        if(Min.at<float>(0,k)>Dat.at<float>(j,k))
                            Min.at<float>(0,k)=Dat.at<float>(j,k);
                    }
                }
            }
        }
        Mat Saltos=(Max-Min)/Num_Barras;
        for(int j=0; j<Dat.rows;j++){
            if(Etiquetas[j]!=0){
                if(Etiquetas[j]==etiqueta){
                    for(int k=0; k<Dat.cols; k++){
                        int pos=0;
                        if(Saltos.at<float>(0,k)>0){
                            float posicion=(Dat.at<float>(j,k)-Min.at<float>(0,k))/Saltos.at<float>(0,k);
                            int pos=(int)floor(posicion);
                            if(pos==Num_Barras)
                                pos=Num_Barras-1;
                        }
                        pos_barra[j][k]=(i*Num_Barras)+pos;
                        His[i][k].at<float>(1,pos)++;
                    }
                }
            }
        }
        for(int j=0; j<Dat.cols; j++){
            for(int k=0; k<Num_Barras; k++)
                His[i][j].at<float>(0,k)=((Min.at<float>(0,j)+(k*Saltos.at<float>(0,j)))+(Min.at<float>(0,j)+((k+1)*Saltos.at<float>(0,j))))/2.0;
        }
    }
    return 0;
}

int MLT::Analisis::Ellipse_Error(vector<Mat> Datos, vector<float> Etiquetas, vector<int> dimensiones, vector<Ellipse_data> &Elipses){
    int e=0;
    if(Datos.size()==0){
        cout<<"ERROR en Ellipse_Error: No hay datos"<<endl;
        return 1;
    }
    if(Etiquetas.size()==0){
        cout<<"ERROR en Ellipse_Error: No hay Etiquetas"<<endl;
        return 1;
    }
    if(Datos.size()!=Etiquetas.size()){
        cout<<"ERROR en Ellipse_Error: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
    if(dimensiones.size()!=2){
        cout<<"ERROR en Ellipse_Error: El número de dimensiones es distintos de 1->No se puede sacar la elipse de error"<<endl;
        return 1;
    }
#ifdef WARNINGS
    for(uint i=0; i<Etiquetas.size(); i++){
        if(Etiquetas[i]==0)
            cout<<"WARNING en Ellipse_Error: Etiquetas con valor 0"<<endl;
    }
#endif
    vector<Mat> Medias,Varianzas,Covarianzas;
    vector<vector<Mat> > D_Prime;
    e=Estadisticos(Datos,Etiquetas,Medias,Varianzas, D_Prime);
    if(e==1){
        cout<<"ERROR en Ellipse_Error: Error en Estadisticos"<<endl;
        return 1;
    }
    e=Covarianza(Datos,Etiquetas,Covarianzas);
    if(e==1){
        cout<<"ERROR en Ellipse_Error: Error en Covarianza"<<endl;
        return 1;
    }
    for(uint i=0; i<Medias.size(); i++){
        Mat covarianza=Mat::zeros(2,2,CV_32FC1);
        covarianza.at<float>(0,0)=Covarianzas[i].at<float>(dimensiones[0]-1,dimensiones[0]-1);
        covarianza.at<float>(1,1)=Covarianzas[i].at<float>(dimensiones[1]-1,dimensiones[1]-1);
        covarianza.at<float>(0,1)=Covarianzas[i].at<float>(dimensiones[0]-1,dimensiones[1]-1);
        covarianza.at<float>(1,0)=Covarianzas[i].at<float>(dimensiones[1]-1,dimensiones[0]-1);
        Mat eigenvalues,eigenvectors;
        cv::eigen(covarianza,eigenvalues,eigenvectors);
        Analisis::Ellipse_data elipse;
        elipse.big=sqrt(eigenvalues.at<float>(0,0));
        elipse.small=sqrt(eigenvalues.at<float>(1,0));
        elipse.angle=180*(atan2(covarianza.at<float>(0,1),pow(elipse.big,2)-covarianza.at<float>(0,0))/CV_PI);
        elipse.media_x=Medias[i].at<float>(0,dimensiones[0]-1);
        elipse.media_y=Medias[i].at<float>(0,dimensiones[1]-1);
        Elipses.push_back(elipse);
    }
    return 0;
}
