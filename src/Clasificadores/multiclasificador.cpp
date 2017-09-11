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

#include "multiclasificador.h"


//MULTICLASIFICADOR REQUIERE TENER UN ARCHIVO DE CONFIGURACIÓN PARA LOS CLASIFICADORES YA GENERADO.
MLT::MultiClasificador::MultiClasificador(vector<Clasificador *> Clasificadores){
    Error=false;
    if(Clasificadores.size()==0){
        Error=true;
        cout<<"ERROR en Multiclasificador: Clasificadores vacio"<<endl;
    }

    for(uint i=0; i<Clasificadores.size(); i++){
        Clasificadores[i]->Read_Data();
        ventanas_x.push_back(Clasificadores[i]->ventana_x);
        ventanas_y.push_back(Clasificadores[i]->ventana_y);
        n_etiquetas.push_back(Clasificadores[i]->numero_etiquetas);
        tipos_dato.push_back(Clasificadores[i]->tipo_dato);
    }
    ventana_x=ventanas_x[0];
    ventana_y=ventanas_y[0];
    numero_etiquetas=n_etiquetas[0];
    tipo_dato=tipos_dato[0];
    for(uint i=1; i<ventanas_x.size(); i++){
        if(ventana_x!=ventanas_x[i]  ||  ventana_y!=ventanas_y[i] || numero_etiquetas!=n_etiquetas[i] || tipo_dato!=tipos_dato[i]){
            cout<<"ERROR en Multiclasificador: Los clasificadores no se han entrenado con los mismos datos"<<endl;
            Error=true;
        }
    }
    clasificadores=Clasificadores;
}


int MLT::MultiClasificador::Cascada(vector<Mat> Data, vector<int> tipo_regla, vector<float> labels_ref, vector<float> &Labels){
    int e=0;
    if(Error==true){
        cout<<"ERROR en Multiclasificador: No se ha inicializado bien la clase"<<endl;
        return 1;
    }
    if(Data.size()==0){
        cout<<"ERROR en Multiclasificador Cascada: Data esta vacio"<<endl;
        return 1;
    }
    if(labels_ref.size()!=tipo_regla.size()){
        cout<<"ERROR en Multiclasificador Cascada: El numero de reglas y labels de referencia es distinto"<<endl;
        return 1;
    }
    if(clasificadores.size()-1!=tipo_regla.size()){
        cout<<"ERROR en Multiclasificador Cascada: El numero de reglas no coincide con el numero de clasificadores establecidos"<<endl;
        return 1;
    }
    for(uint i=0; i<tipo_regla.size(); i++){
        if(tipo_regla[i]!=IGUAL && tipo_regla[i]!=DISTINTO && tipo_regla[i]!=MENOR && tipo_regla[i]!=MAYOR){
            cout<<"ERROR en Multiclasificador Cascada: El tipo de regla es erróneo"<<endl;
            return 1;
        }
    }
#ifdef GUI
    for(int i=0; i<clasificadores.size();i++){
        clasificadores[i]->progreso=progreso;
        clasificadores[i]->max_progreso=max_progreso;
        clasificadores[i]->base_progreso=base_progreso;
        clasificadores[i]->total_progreso=total_progreso;
        clasificadores[i]->window=window;
    }
#endif
    vector<vector<float> > labels;
    for(uint i=0; i<clasificadores.size(); i++){
        vector<float> l;
        e=clasificadores[i]->Autoclasificacion(Data,l,false,false);
        if(e==1){
            cout<<"Error en "+clasificadores[i]->nombre+": Error en Autoclasificacion"<<endl;
            return 1;
        }
        labels.push_back(l);
    }
    for(uint i=0; i<labels[0].size(); i++){
        bool pasa=true;
        uint pos=0;
        while(pasa && pos!=labels.size()){
            float label=labels[pos][i];
            if(pos+1==labels.size()){
                Labels.push_back(label);
            }
            else{
                float label_ref=labels_ref[pos];
                if(tipo_regla[pos]==IGUAL){
                    if(label==label_ref)
                        pasa=true;
                    else{
                        pasa=false;
                        Labels.push_back(label);
                    }
                }
                if(tipo_regla[pos]==DISTINTO){
                    if(label!=label_ref)
                        pasa=true;
                    else{
                        pasa=false;
                        Labels.push_back(label);
                    }
                }
                if(tipo_regla[pos]==MENOR){
                    if(label<label_ref)
                        pasa=true;
                    else{
                        pasa=false;
                        Labels.push_back(label);
                    }
                }
                if(tipo_regla[pos]==MAYOR){
                    if(label>label_ref)
                        pasa=true;
                    else{
                        pasa=false;
                        Labels.push_back(label);
                    }
                }
            }
            pos++;
        }
    }
    return 0;
}

int MLT::MultiClasificador::Votacion(vector<Mat> Data, vector<float> w_clasif, vector<float> &Labels){
    int e=0;
    if(Error==true){
        cout<<"ERROR en Multiclasificador: No se ha inicializado bien la clase"<<endl;
        return 1;
    }
    if(clasificadores.size()!=w_clasif.size()){
        cout<<"ERROR en Multiclasificador Votacion: El número de clasificadores es distinto al de w_clasif"<<endl;
        return 1;
    }
    if(Data.size()==0){
        cout<<"ERROR en Multiclasificador Votacion: Data esta vacio"<<endl;
        return 1;
    }
#ifdef GUI
    for(int i=0; i<clasificadores.size();i++){
        clasificadores[i]->progreso=progreso;
        clasificadores[i]->max_progreso=max_progreso;
        clasificadores[i]->base_progreso=base_progreso;
        clasificadores[i]->total_progreso=total_progreso;
        clasificadores[i]->window=window;
    }
#endif
    vector<vector<float> > labels;
    for(uint i=0; i<clasificadores.size(); i++){
        vector<float> l;
        e=clasificadores[i]->Autoclasificacion(Data,l,false,false);
        if(e==1){
            cout<<"Error en "+clasificadores[i]->nombre+": Error en Autoclasificacion"<<endl;
            return 1;
        }
        labels.push_back(l);
    }
    for(uint i=0; i<labels[0].size(); i++){
        int num_etiq=0;
        bool negativa=false;
        for(uint j=0; j<labels.size(); j++){
            if(labels[j][i]<0)
                negativa=true;
            if(num_etiq<labels[j][i])
                num_etiq=labels[j][i];
        }
        if(negativa)
            num_etiq=num_etiq+1;
        vector<float> pesos(num_etiq);
        for(uint j=0; j<pesos.size(); j++)
            pesos[j]=0.f;
        for(uint j=0; j<labels.size(); j++){
            float label=labels[j][i];
            if(negativa && label<0)
                pesos[label+1]=pesos[label+1]+w_clasif[j];
            else if(negativa && label>0)
                pesos[label]=pesos[label]+w_clasif[j];
            else if(!negativa && label!=0)
                pesos[label-1]=pesos[label-1]+w_clasif[j];
            else if(label==0){
#ifdef WARNINGS
                cout<<"WARNING: El clasificador "<<clasificadores[j]->nombre<<" ha dado un resultado erróneo (0)"<<endl;
#endif
            }
        }
        int pos=0;
        float maximo=0;
        for(uint j=0; j<pesos.size(); j++){
            if(pesos[j]>maximo){
                maximo=pesos[j];
                pos=j;
            }
        }
        if(negativa && pos==0)
            Labels.push_back(pos-1);
        else if(negativa && pos>0)
            Labels.push_back(pos);
        else
            Labels.push_back(pos+1);
    }
    return 0;
}
