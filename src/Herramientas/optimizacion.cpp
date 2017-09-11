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
#include "optimizacion.h"

MLT::Optimizacion::Optimizacion(){}


int MLT::Optimizacion::Validation(vector<Mat> Datos, vector<float> Labels, int Porcentaje_validation, int id_clasificador, Parametros parame, float &Error, Mat &Confusion, vector<Analisis::Ratios_data> &Ratios){
    int e=0;
    if(Datos.size()==0){
        cout<<"ERROR en Validation: No hay datos"<<endl;
        return 1;
    }
    if(Labels.size()==0){
        cout<<"ERROR en Validation: No hay Etiquetas"<<endl;
        return 1;
    }
    if(Datos.size()!=Labels.size()){
        cout<<"ERROR en Validation: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
    for(uint i=0; i<Labels.size(); i++){
        if(Labels[i]==0){
            cout<<"ERROR en Validation: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    Dimensionalidad::Reducciones reduc;
    Generacion::Info_Datos info;
    info.Tam_X=Datos[0].cols;
    info.Tam_Y=Datos[0].rows;
    vector<Mat> dat;
    dat=Datos;
    vector<float> lab;
    lab=Labels;
    vector<Mat> datos_entrena;
    vector<Mat> datos_valida;
    vector<float> labels_entrena;
    vector<float> labels_valida;
    uint num_val=round(Datos.size()*Porcentaje_validation/100);
    while (dat.size()>Datos.size()-num_val){
        int pos=rand() %dat.size();
        datos_valida.push_back(dat[pos]);
        labels_valida.push_back(lab[pos]);
        dat.erase(dat.begin()+pos);
        lab.erase(lab.begin()+pos);
    }
    datos_entrena=dat;
    labels_entrena=lab;
    vector<float> result;
    if(id_clasificador==DISTANCIAS){
        Clasificador_Distancias D("Validation");
#ifdef GUI
    D.progreso=progreso;
    D.max_progreso=max_progreso;
    D.base_progreso=base_progreso;
    D.total_progreso=total_progreso;
    D.window=window;
#endif
        e=D.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Distancias"<<endl;
            return 1;
        }
        e=D.Autoclasificacion(datos_valida,result,false,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_Distancias"<<endl;
            return 1;
        }
    }
    else if(id_clasificador==GAUSSIANO){
        Clasificador_Gaussiano G("Validation");
#ifdef GUI
    G.progreso=progreso;
    G.max_progreso=max_progreso;
    G.base_progreso=base_progreso;
    G.total_progreso=total_progreso;
    G.window=window;
#endif
        e=G.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Gaussiano"<<endl;
            return 1;
        }
        e=G.Autoclasificacion(datos_valida,result,false,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_Gaussiano"<<endl;
            return 1;
        }
    }
//    else if(id_clasificador==CASCADA_CLAS){
//        Clasificador_Cascada HA("Validation");
//#ifdef GUI
//    HA.progreso=progreso;
//    HA.max_progreso=max_progreso;
//    HA.base_progreso=base_progreso;
//    HA.total_progreso=total_progreso;
//    HA.window=window;
//#endif
//        e=HA.Autotrain(datos_entrena,labels_entrena,false);
//        if(e==1){
//            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Histograma"<<endl;
//            return 1;
//        }
//        e=HA.Autoclasificacion(datos_valida,result,false);
//        if(e==1){
//            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_Histograma"<<endl;
//            return 1;
//        }
//    }
    else if(id_clasificador==HISTOGRAMA){
        Clasificador_Histograma H("Validation",parame.Hist_tam_celda);
#ifdef GUI
    H.progreso=progreso;
    H.max_progreso=max_progreso;
    H.base_progreso=base_progreso;
    H.total_progreso=total_progreso;
    H.window=window;
#endif
        e=H.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Histograma"<<endl;
            return 1;
        }
        e=H.Autoclasificacion(datos_valida,result,false,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_Histograma"<<endl;
            return 1;
        }
    }
    else if(id_clasificador==KNN){
        Clasificador_KNN K("Validation",parame.KNN_k, parame.KNN_regression);
#ifdef GUI
    K.progreso=progreso;
    K.max_progreso=max_progreso;
    K.base_progreso=base_progreso;
    K.total_progreso=total_progreso;
    K.window=window;
#endif
        e=K.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_KNN"<<endl;
            return 1;
        }
        e=K.Autoclasificacion(datos_valida,result,false,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_KNN"<<endl;
            return 1;
        }
    }
    else if(id_clasificador==NEURONAL){
        Clasificador_Neuronal N("Validation", parame.Neuronal_layerSize, parame.Neuronal_Method, parame.Neuronal_Function, parame.Neuronal_bp_dw_scale,
                                parame.Neuronal_bp_moment_scale, parame.Neuronal_rp_dw0, parame.Neuronal_rp_dw_max, parame.Neuronal_rp_dw_min,
                                parame.Neuronal_rp_dw_minus, parame.Neuronal_rp_dw_plus, parame.Neuronal_fparam1, parame.Neuronal_fparam2);
#ifdef GUI
    N.progreso=progreso;
    N.max_progreso=max_progreso;
    N.base_progreso=base_progreso;
    N.total_progreso=total_progreso;
    N.window=window;
#endif
        e=N.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Neuronal"<<endl;
            return 1;
        }
        e=N.Autoclasificacion(datos_valida,result,false,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_Neuronal"<<endl;
            return 1;
        }
    }
    else if(id_clasificador==C_SVM){
        Clasificador_SVM S("Validation",parame.SVM_train, parame.SVM_Type, parame.SVM_kernel_type, parame.SVM_class_weights, parame.SVM_degree,
                           parame.SVM_gamma, parame.SVM_coef0, parame.SVM_C,parame.SVM_nu, parame.SVM_p);
#ifdef GUI
    S.progreso=progreso;
    S.max_progreso=max_progreso;
    S.base_progreso=base_progreso;
    S.total_progreso=total_progreso;
    S.window=window;
#endif
        e=S.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_SVM"<<endl;
            return 1;
        }
        e=S.Autoclasificacion(datos_valida,result,false,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_SVM"<<endl;
            return 1;
        }
    }
    else if(id_clasificador==RTREES){
        Clasificador_RTrees T("Validation",parame.RTrees_max_depth, parame.RTrees_min_sample_count, parame.RTrees_regression_accuracy,
                             parame.RTrees_use_surrogates, parame.RTrees_max_categories, parame.RTrees_cv_folds, parame.RTrees_use_1se_rule,
                             parame.RTrees_truncate_pruned_tree, parame.RTrees_priors,parame.RTrees_calc_var_importance, parame.RTrees_native_vars);
#ifdef GUI
    T.progreso=progreso;
    T.max_progreso=max_progreso;
    T.base_progreso=base_progreso;
    T.total_progreso=total_progreso;
    T.window=window;
#endif
        e=T.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Tree"<<endl;
            return 1;
        }
        vector<float> result;
        e=T.Autoclasificacion(datos_valida,result,false,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_Tree"<<endl;
            return 1;
        }
    }
    else if(id_clasificador==DTREES){
        Clasificador_DTrees T2("Validation",parame.DTrees_max_depth, parame.DTrees_min_sample_count, parame.DTrees_regression_accuracy,
                               parame.DTrees_use_surrogates, parame.DTrees_max_categories, parame.DTrees_cv_folds, parame.DTrees_use_1se_rule,
                               parame.DTrees_truncate_pruned_tree, parame.DTrees_priors);
#ifdef GUI
    T2.progreso=progreso;
    T2.max_progreso=max_progreso;
    T2.base_progreso=base_progreso;
    T2.total_progreso=total_progreso;
    T2.window=window;
#endif
        e=T2.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Tree2"<<endl;
            return 1;
        }
        e=T2.Autoclasificacion(datos_valida,result,false,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_Tree2"<<endl;
            return 1;
        }
    }
    else if(id_clasificador==BOOSTING){
        Clasificador_Boosting B("Validation",parame.Boosting_boost_type,parame.Boosting_weak_count,parame.Boosting_weight_trim_rate,parame.Boosting_max_depth,parame.Boosting_use_surrogates,parame.Boosting_priors);
#ifdef GUI
    B.progreso=progreso;
    B.max_progreso=max_progreso;
    B.base_progreso=base_progreso;
    B.total_progreso=total_progreso;
    B.window=window;
#endif
        e=B.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Boosting"<<endl;
            return 1;
        }
        e=B.Autoclasificacion(datos_valida,result,false,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_Boosting"<<endl;
            return 1;
        }
    }
//    else if(id_clasificador==GBT){
//        Clasificador_GBT GB("Validation",parame.GBT_loss_function_type,parame.GBT_weak_count,parame.GBT_shrinkage,parame.GBT_subsample_portion,parame.GBT_max_depth, parame.GBT_use_surrogates);
//#ifdef GUI
//    GB.progreso=progreso;
//    GB.max_progreso=max_progreso;
//    GB.base_progreso=base_progreso;
//    GB.total_progreso=total_progreso;
//    GB.window=window;
//#endif
//        e=GB.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
//        if(e==1){
//            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_GBT"<<endl;
//            return 1;
//        }
//        e=GB.Autoclasificacion(datos_valida,result,false,false);
//        if(e==1){
//            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_GBT"<<endl;
//            return 1;
//        }
//    }
    else if(id_clasificador==EXP_MAX){
        Clasificador_EM E("Validation",parame.EM_nclusters, parame.EM_covMatType);
#ifdef GUI
    E.progreso=progreso;
    E.max_progreso=max_progreso;
    E.base_progreso=base_progreso;
    E.total_progreso=total_progreso;
    E.window=window;
#endif
        e=E.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_EM"<<endl;
            return 1;
        }
        e=E.Autoclasificacion(datos_valida,result,false,false);
        if(e==1){
            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_EM"<<endl;
            return 1;
        }
    }
//    else if(id_clasificador==ERTREES){
//        Clasificador_ERTrees ER("Validation",parame.ERTrees_max_depth, parame.ERTrees_min_sample_count, parame.ERTrees_regression_accuracy, parame.ERTrees_use_surrogates, parame.ERTrees_max_categories, parame.ERTrees_cv_folds, parame.ERTrees_use_1se_rule, parame.ERTrees_truncate_pruned_tree, parame.ERTrees_priors, parame.ERTrees_calc_var_importance, parame.ERTrees_native_vars);
//#ifdef GUI
//    ER.progreso=progreso;
//    ER.max_progreso=max_progreso;
//    ER.base_progreso=base_progreso;
//    ER.total_progreso=total_progreso;
//    ER.window=window;
//#endif
//        e=ER.Autotrain(datos_entrena,labels_entrena,reduc,info,false);
//        if(e==1){
//            cout<<"ERROR en Validation: Error en Autotrain en Clasificador_ERTrees"<<endl;
//            return 1;
//        }
//        e=ER.Autoclasificacion(datos_valida,result,false,false);
//        if(e==1){
//            cout<<"ERROR en Validation: Error en Autoclasificacion en Clasificador_ERTrees"<<endl;
//            return 1;
//        }
//    }
//    else {
//        cout<<"ERROR en Validation: id_clasificador erroneo"<<endl;
//        return 1;
//    }
    Analisis an;
    Mat Conf;
    float error=0;
    e=an.Confusion(labels_valida,result,Conf,error);
    if(e==1){
        cout<<"ERROR en Validation: Error en Confusion"<<endl;
        return 1;
    }
    vector<Analisis::Ratios_data> ratios;
    an.Ratios(labels_valida,result,ratios);
    Ratios=ratios;
    Auxiliares aux;
    bool negativa=false;
    aux.numero_etiquetas(Labels,negativa);
    cout<<"Ratios= "<<endl;
    for(uint i=0; i<ratios.size(); i++){
        int etiqueta;
        if(negativa){
            if(i==0)
                etiqueta=-1;
            else
                etiqueta=i;
        }
        else
            etiqueta=i+1;
        cout<<"Label="<<etiqueta<<endl;
        cout<<"VP="<<ratios[i].VP<<endl;
        cout<<"VN="<<ratios[i].VN<<endl;
        cout<<"FP="<<ratios[i].FP<<endl;
        cout<<"FN="<<ratios[i].FN<<endl;
        cout<<"FAR="<<ratios[i].FAR<<endl;
        cout<<"FRR="<<ratios[i].FRR<<endl;
        cout<<"TAR="<<ratios[i].TAR<<endl;
        cout<<"TRR="<<ratios[i].TRR<<endl;
        cout<<"EXP_ERROR="<<ratios[i].EXP_ERROR<<endl;
        cout<<endl;
    }
    Error=error;
    Conf.copyTo(Confusion);
    cout<<"Error= "<<error<<endl;
    cout<<"Matriz Confusion= "<<endl<<Conf<<endl;
    return 0;
}


int MLT::Optimizacion::Validation(vector<Mat> Datos, vector<float> Labels, int Porcentaje_validation, vector<int> id_clasif, Parametros parame, MultiClasificador::Multi_type multi, float &Error, Mat &Confusion, vector<Analisis::Ratios_data> &Ratios){
    int e=0;
    if(Datos.size()==0){
        cout<<"ERROR en Validation: No hay datos"<<endl;
        return 1;
    }
    if(Labels.size()==0){
        cout<<"ERROR en Validation: No hay Etiquetas"<<endl;
        return 1;
    }
    if(Datos.size()!=Labels.size()){
        cout<<"ERROR en Validation: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
    if(id_clasif.size()==0){
        cout<<"ERROR en Validation: id_clasif vacio"<<endl;
        return 1;
    }
    for(uint i=0; i<id_clasif.size(); i++){
        if(id_clasif[i]<0 || id_clasif[i]>NUM_CLASIF){
            cout<<"ERROR en Validation: id_clasif erroneo"<<endl;
            return 1;
        }
    }
    for(uint i=0; i<Labels.size(); i++){
        if(Labels[i]==0){
            cout<<"ERROR en Validation: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    Dimensionalidad::Reducciones reduc;
    Generacion::Info_Datos info;
    info.Tam_X=Datos[0].cols;
    info.Tam_Y=Datos[0].rows;
    vector<Mat> dat;
    dat=Datos;
    vector<float> lab;
    lab=Labels;
    vector<Mat> datos_entrena;
    vector<Mat> datos_valida;
    vector<float> labels_entrena;
    vector<float> labels_valida;
    uint num_val=round(Datos.size()*Porcentaje_validation/100);
    while (dat.size()>Datos.size()-num_val){
        int pos=rand() %dat.size();
        datos_valida.push_back(dat[pos]);
        labels_valida.push_back(lab[pos]);
        dat.erase(dat.begin()+pos);
        lab.erase(lab.begin()+pos);
    }
    datos_entrena=dat;
    labels_entrena=lab;
    vector<string> Nombres(id_clasif.size());
    for(uint i=0; i<Nombres.size(); i++){
        stringstream num;
        num<<i;
        Nombres[i]="Validation"+num.str();
    }
    vector<Clasificador*> clasificadores;
    for(uint i=0; i<id_clasif.size(); i++){
        if(id_clasif[i]==DISTANCIAS){
            Clasificador_Distancias *D=new Clasificador_Distancias(Nombres[i]);
            e=D->Autotrain(datos_entrena,labels_entrena,reduc,info,true);
            if(e==1){
                cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Distancias"<<endl;
                return 1;
            }
            clasificadores.push_back(D);
#ifdef GUI
    D->progreso=progreso;
    D->max_progreso=max_progreso;
    D->base_progreso=base_progreso;
    D->total_progreso=total_progreso;
    D->window=window;
#endif
        }
        if(id_clasif[i]==GAUSSIANO){
            Clasificador_Gaussiano *G=new Clasificador_Gaussiano(Nombres[i]);
            e=G->Autotrain(datos_entrena,labels_entrena,reduc,info,true);
            if(e==1){
                cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Gaussiano"<<endl;
                return 1;
            }
            clasificadores.push_back(G);
#ifdef GUI
    G->progreso=progreso;
    G->max_progreso=max_progreso;
    G->base_progreso=base_progreso;
    G->total_progreso=total_progreso;
    G->window=window;
#endif
        }
        if(id_clasif[i]==HISTOGRAMA){
            Clasificador_Histograma *H=new Clasificador_Histograma(Nombres[i],parame.Hist_tam_celda);
            e=H->Autotrain(datos_entrena,labels_entrena,reduc,info,true);
            if(e==1){
                cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Histograma"<<endl;
                return 1;
            }
            clasificadores.push_back(H);
#ifdef GUI
    H->progreso=progreso;
    H->max_progreso=max_progreso;
    H->base_progreso=base_progreso;
    H->total_progreso=total_progreso;
    H->window=window;
#endif
        }
        if(id_clasif[i]==KNN){
            Clasificador_KNN *K=new Clasificador_KNN(Nombres[i],parame.KNN_k, parame.KNN_regression);
            e=K->Autotrain(datos_entrena,labels_entrena,reduc,info,true);
            if(e==1){
                cout<<"ERROR en Validation: Error en Autotrain en Clasificador_KNN"<<endl;
                return 1;
            }
            clasificadores.push_back(K);
#ifdef GUI
    K->progreso=progreso;
    K->max_progreso=max_progreso;
    K->base_progreso=base_progreso;
    K->total_progreso=total_progreso;
    K->window=window;
#endif
        }
        if(id_clasif[i]==NEURONAL){
            Clasificador_Neuronal *N=new Clasificador_Neuronal(Nombres[i],parame.Neuronal_layerSize, parame.Neuronal_Method, parame.Neuronal_Function, parame.Neuronal_bp_dw_scale,
                                    parame.Neuronal_bp_moment_scale, parame.Neuronal_rp_dw0, parame.Neuronal_rp_dw_max, parame.Neuronal_rp_dw_min,
                                    parame.Neuronal_rp_dw_minus, parame.Neuronal_rp_dw_plus, parame.Neuronal_fparam1, parame.Neuronal_fparam2);
            e=N->Autotrain(datos_entrena,labels_entrena,reduc,info,true);
            if(e==1){
                cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Neuronal"<<endl;
                return 1;
            }
            clasificadores.push_back(N);
#ifdef GUI
    N->progreso=progreso;
    N->max_progreso=max_progreso;
    N->base_progreso=base_progreso;
    N->total_progreso=total_progreso;
    N->window=window;
#endif
        }
        if(id_clasif[i]==C_SVM){
            Clasificador_SVM *S=new Clasificador_SVM(Nombres[i],parame.SVM_train, parame.SVM_Type, parame.SVM_kernel_type, parame.SVM_class_weights, parame.SVM_degree,
                               parame.SVM_gamma, parame.SVM_coef0, parame.SVM_C,parame.SVM_nu, parame.SVM_p);
            e=S->Autotrain(datos_entrena,labels_entrena,reduc,info,true);
            if(e==1){
                cout<<"ERROR en Validation: Error en Autotrain en Clasificador_SVM"<<endl;
                return 1;
            }
            clasificadores.push_back(S);
#ifdef GUI
    S->progreso=progreso;
    S->max_progreso=max_progreso;
    S->base_progreso=base_progreso;
    S->total_progreso=total_progreso;
    S->window=window;
#endif
        }
        if(id_clasif[i]==RTREES){
            Clasificador_RTrees *T=new Clasificador_RTrees(Nombres[i],parame.RTrees_max_depth, parame.RTrees_min_sample_count, parame.RTrees_regression_accuracy,
                                 parame.RTrees_use_surrogates, parame.RTrees_max_categories, parame.RTrees_cv_folds, parame.RTrees_use_1se_rule,
                                 parame.RTrees_truncate_pruned_tree, parame.RTrees_priors,parame.RTrees_calc_var_importance, parame.RTrees_native_vars);
            e=T->Autotrain(datos_entrena,labels_entrena,reduc,info,true);
            if(e==1){
                cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Tree"<<endl;
                return 1;
            }
            clasificadores.push_back(T);
#ifdef GUI
    T->progreso=progreso;
    T->max_progreso=max_progreso;
    T->base_progreso=base_progreso;
    T->total_progreso=total_progreso;
    T->window=window;
#endif
        }
        if(id_clasif[i]==DTREES){
            Clasificador_DTrees *T2=new Clasificador_DTrees(Nombres[i],parame.DTrees_max_depth, parame.DTrees_min_sample_count, parame.DTrees_regression_accuracy,
                                   parame.DTrees_use_surrogates, parame.DTrees_max_categories, parame.DTrees_cv_folds, parame.DTrees_use_1se_rule,
                                   parame.DTrees_truncate_pruned_tree, parame.DTrees_priors);
            e=T2->Autotrain(datos_entrena,labels_entrena,reduc,info,true);
            if(e==1){
                cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Tree2"<<endl;
                return 1;
            }
            clasificadores.push_back(T2);
#ifdef GUI
    T2->progreso=progreso;
    T2->max_progreso=max_progreso;
    T2->base_progreso=base_progreso;
    T2->total_progreso=total_progreso;
    T2->window=window;
#endif

        }
        if(id_clasif[i]==BOOSTING){
            Clasificador_Boosting *B=new Clasificador_Boosting(Nombres[i],parame.Boosting_boost_type,parame.Boosting_weak_count,parame.Boosting_weight_trim_rate,parame.Boosting_max_depth,parame.Boosting_use_surrogates,parame.Boosting_priors);
            e=B->Autotrain(datos_entrena,labels_entrena,reduc,info,false);
            if(e==1){
                cout<<"ERROR en Validation: Error en Autotrain en Clasificador_Boosting"<<endl;
                return 1;
            }
            clasificadores.push_back(B);
#ifdef GUI
    B->progreso=progreso;
    B->max_progreso=max_progreso;
    B->base_progreso=base_progreso;
    B->total_progreso=total_progreso;
    B->window=window;
#endif
        }
        if(id_clasif[i]==EXP_MAX){
            Clasificador_EM *E=new Clasificador_EM(Nombres[i],parame.EM_nclusters, parame.EM_covMatType);
            e=E->Autotrain(datos_entrena,labels_entrena,reduc,info,false);
            if(e==1){
                cout<<"ERROR en Validation: Error en Autotrain en Clasificador_EM"<<endl;
                return 1;
            }
            clasificadores.push_back(E);
#ifdef GUI
    E->progreso=progreso;
    E->max_progreso=max_progreso;
    E->base_progreso=base_progreso;
    E->total_progreso=total_progreso;
    E->window=window;
#endif

        }
    }

    vector<float> result;
    MultiClasificador mul(clasificadores);
#ifdef GUI
    mul.progreso=progreso;
    mul.max_progreso=max_progreso;
    mul.base_progreso=base_progreso;
    mul.total_progreso=total_progreso;
    mul.window=window;
#endif
    if(multi.tipo==CASCADA)
        e=mul.Cascada(datos_valida,multi.tipo_regla,multi.label_ref,result);
    if(e==1){
        cout<<"ERROR en Validation: Error en Cascada"<<endl;
        return 1;
    }
    else if(multi.tipo==VOTACION)
        e=mul.Votacion(datos_valida,multi.w_clasif,result);
    if(e==1){
        cout<<"ERROR en Validation: Error en Votacion"<<endl;
        return 1;
    }
    Analisis an;
    vector<Analisis::Ratios_data> ratios;
    an.Ratios(labels_valida,result,ratios);
    Ratios=ratios;
    Auxiliares aux;
    bool negativa=false;
    aux.numero_etiquetas(Labels,negativa);
    cout<<"Ratios= "<<endl;
    for(uint i=0; i<ratios.size(); i++){
        int etiqueta;
        if(negativa){
            if(i==0)
                etiqueta=-1;
            else
                etiqueta=i;
        }
        else
            etiqueta=i+1;
        cout<<"Label="<<etiqueta<<endl;
        cout<<"VP="<<ratios[i].VP<<endl;
        cout<<"VN="<<ratios[i].VN<<endl;
        cout<<"FP="<<ratios[i].FP<<endl;
        cout<<"FN="<<ratios[i].FN<<endl;
        cout<<"FAR="<<ratios[i].FAR<<endl;
        cout<<"FRR="<<ratios[i].FRR<<endl;
        cout<<"TAR="<<ratios[i].TAR<<endl;
        cout<<"TRR="<<ratios[i].TRR<<endl;
        cout<<"EXP_ERROR="<<ratios[i].EXP_ERROR<<endl;
        cout<<endl;
    }
    Mat Conf;
    float error=0;
    e=an.Confusion(labels_valida,result,Conf,error);
    if(e==1){
        cout<<"ERROR en Validation: Error en Confusion"<<endl;
        return 1;
    }
    Error=error;
    Conf.copyTo(Confusion);
    cout<<"Error= "<<error<<endl;
    cout<<"Matriz Confusion= "<<endl<<Conf<<endl;
    return 0;
}

int MLT::Optimizacion::Cross_Validation(vector<Mat> Datos, vector<float> Labels, int Num_Folds, int Tam_Fold, int id_clasificador, Parametros inicio, Parametros fin, Parametros salto, Parametros &parametros, float &Error, cv::Mat &Confus){
    int e=0;
    Auxiliares ax;
    if(Datos.size()==0){
        cout<<"ERROR en Cross_Validation: No hay datos"<<endl;
        return 1;
    }
    if(Labels.size()==0){
        cout<<"ERROR en Cross_Validation: No hay Etiquetas"<<endl;
        return 1;
    }
    if(Datos.size()!=Labels.size()){
        cout<<"ERROR en Cross_Validation: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
    if((uint)Num_Folds*Tam_Fold>Datos.size()){
        cout<<"ERROR en Cross_Validation: Numero de datos menor de lo que se pide para el proceso"<<endl;
        return 1;
    }
    for(uint i=0; i<Labels.size(); i++){
        if(Labels[i]==0){
            cout<<"ERROR en Cross_Validation: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    Dimensionalidad::Reducciones reduc;
    Generacion::Info_Datos info;
    info.Tam_X=Datos[0].cols;
    info.Tam_Y=Datos[0].rows;
    Analisis an;
    vector<Mat> dat;
    dat=Datos;
    vector<float> lab;
    lab=Labels;
    vector<bool> cogidos;
    for(uint i=0; i<Datos.size(); i++)
        cogidos.push_back(false);
    vector<vector<Mat> > datos_entrena;
    vector<vector<Mat> > datos_valida;
    vector<vector<float> > labels_entrena;
    vector<vector<float> > labels_valida;
    for(int i=0; i<Num_Folds; i++){
        vector<bool> cog;
        for(uint j=0; j<Datos.size(); j++)
            cog.push_back(false);
        vector<Mat> dat_entrena;
        vector<Mat> dat_valida;
        vector<float> lab_entrena;
        vector<float> lab_valida;
        int contador=0;
        while (contador<Tam_Fold){
            int pos=rand() %dat.size();
            if(cogidos[pos]==false){
                dat_valida.push_back(dat[pos]);
                lab_valida.push_back(lab[pos]);
                cogidos[pos]=true;
                cog[pos]=true;
                contador++;
            }
        }
        for(uint j=0; j<cog.size(); j++){
            if(cog[j]==false){
                dat_entrena.push_back(dat[j]);
                lab_entrena.push_back(lab[j]);
            }
        }
        datos_entrena.push_back(dat_entrena);
        labels_entrena.push_back(lab_entrena);
        datos_valida.push_back(dat_valida);
        labels_valida.push_back(lab_valida);
    }
    if(id_clasificador==DISTANCIAS){
#ifdef WARNINGS
        std::cout << "WARNING: clasificador_distancias no tiene parametros se hara un Validation"<< std::endl;
#endif
        float porcentaje=round(100*((float)Tam_Fold/Datos.size()));
        vector<Analisis::Ratios_data> Ratios;
        Validation(Datos,Labels,porcentaje,DISTANCIAS,inicio, Error, Confus, Ratios);
    }
    else if(id_clasificador==GAUSSIANO){
#ifdef WARNINGS
        std::cout << "WARNING: clasificador_gaussiano no tiene parametros se hara un Validation"<< std::endl;
#endif
        float porcentaje=round(100*((float)Tam_Fold/Datos.size()));
        vector<Analisis::Ratios_data> Ratios;
        Validation(Datos,Labels,porcentaje,GAUSSIANO,inicio, Error, Confus, Ratios);
    }
    else if(id_clasificador==CASCADA_CLAS){
        cout<<endl<<endl<<"WARNING en Cross_Validation: NO SE HA IMPLEMENTADO AUN EL Clasificador_Cascada EN Cross_Validation"<<endl;
//#ifdef WARNINGS
//        std::cout << "WARNING: clasificador_haar no esta implementado se utilizara clasificador_gaussiano que no tiene parametros se hara un Validation"<< std::endl;
//#endif
//        float porcentaje=round(100*((float)Tam_Fold/Datos.size()));
//        Mat Confusion;
//        vector<Analisis::Ratios_data> Ratios;
//        Validation(Datos,Labels,porcentaje,CASCADA_CLAS,inicio,reduc, Error, Confusion, Ratios);
    }
    else if(id_clasificador==HISTOGRAMA){
        //tam_celda
        if(inicio.Hist_tam_celda>=fin.Hist_tam_celda){
            cout<<"ERROR en Cross_Validation: Los parametros de inicio y fin son los mismos o los de inicio son mayores a los de fin"<<endl;
            return 1;
        }
        bool negativa;
        int num_etiq=ax.numero_etiquetas(Labels,negativa);
        Confus=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
        float min_error=99999999;
        while(inicio.Hist_tam_celda<=fin.Hist_tam_celda){
            float total_error=0;
            cv::Mat Total_Confusion=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
            for(int i=0; i<Num_Folds; i++){
                Clasificador_Histograma H("Cross_Validation",inicio.Hist_tam_celda);
#ifdef GUI
    H.progreso=progreso;
    H.max_progreso=max_progreso;
    H.base_progreso=base_progreso;
    H.total_progreso=total_progreso;
    H.window=window;
#endif
                e=H.Autotrain(datos_entrena[i],labels_entrena[i],reduc,info,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autotrain en Clasificador_Histograma"<<endl;
                    return 1;
                }
                vector<float> result;
                e=H.Autoclasificacion(datos_valida[i],result,false,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autoclasificacion en Clasificador_Histograma"<<endl;
                    return 1;
                }
                Mat Conf;
                float error=0;
                e=an.Confusion(labels_valida[i],result,Conf,error);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Confusion"<<endl;
                    return 1;
                }
                Total_Confusion=Total_Confusion+Conf;
                total_error=total_error+error;
            }
            total_error=total_error/Num_Folds;
            if(total_error<=min_error){
                Total_Confusion.copyTo(Confus);
                min_error=total_error;
                parametros=inicio;
            }
            inicio.Hist_tam_celda=inicio.Hist_tam_celda+salto.Hist_tam_celda;
        }
        Error=min_error;
        cout<<"Parametros optimizados"<<endl;
        cout<<"Hist_tam_celdea= "<<parametros.Hist_tam_celda<<endl;
        cout<<"Error= "<<min_error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    else if(id_clasificador==KNN){
        //K
        if(inicio.KNN_k>=fin.KNN_k){
            cout<<"ERROR en Cross_Validation: Los parametros de inicio y fin son los mismos o los de inicio son mayores a los de fin"<<endl;
            return 1;
        }
        bool negativa;
        int num_etiq=ax.numero_etiquetas(Labels,negativa);
        Confus=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
        float min_error=99999999;
        while(inicio.KNN_k<=fin.KNN_k){
            float total_error=0;
            cv::Mat Total_Confusion=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
            for(int i=0; i<Num_Folds; i++){
                Clasificador_KNN KNN("Cross_Validation",inicio.KNN_k,inicio.KNN_regression);
#ifdef GUI
    KNN.progreso=progreso;
    KNN.max_progreso=max_progreso;
    KNN.base_progreso=base_progreso;
    KNN.total_progreso=total_progreso;
    KNN.window=window;
#endif
                e=KNN.Autotrain(datos_entrena[i],labels_entrena[i],reduc,info,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autotrain en Clasificador_KNN"<<endl;
                    return 1;
                }
                vector<float> result;
                e=KNN.Autoclasificacion(datos_valida[i],result,false,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autoclasificacion en Clasificador_KNN"<<endl;
                    return 1;
                }
                Mat Conf;
                float error=0;
                e=an.Confusion(labels_valida[i],result,Conf,error);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Confusion"<<endl;
                    return 1;
                }
                Total_Confusion=Total_Confusion+Conf;
                total_error=total_error+error;
            }
            total_error=total_error/Num_Folds;
            if(total_error<min_error){
                Total_Confusion.copyTo(Confus);
                min_error=total_error;
                parametros=inicio;
            }
            inicio.KNN_k=inicio.KNN_k+salto.KNN_k;
        }
        Error=min_error;
        cout<<"Parametros optimizados"<<endl;
        cout<<"KNN_k= "<<parametros.KNN_k<<endl;
        cout<<"Error= "<<min_error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    else if(id_clasificador==NEURONAL){
//bp_dw_scale, bp_moment_scale, rp_dw0, rp_dw_max, rp_dw_min, rp_dw_minus, rp_dw_plus, fparam1, fparam2
        if(inicio.Neuronal_bp_dw_scale>=fin.Neuronal_bp_dw_scale && inicio.Neuronal_bp_moment_scale>=fin.Neuronal_bp_moment_scale && inicio.Neuronal_rp_dw0>=fin.Neuronal_rp_dw0 && inicio.Neuronal_rp_dw_max>=fin.Neuronal_rp_dw_max
                && inicio.Neuronal_rp_dw_min>=fin.Neuronal_rp_dw_min && inicio.Neuronal_rp_dw_minus>=fin.Neuronal_rp_dw_minus && inicio.Neuronal_rp_dw_plus>=fin.Neuronal_rp_dw_plus && inicio.Neuronal_fparam1>=fin.Neuronal_fparam1 && inicio.Neuronal_fparam2>=fin.Neuronal_fparam2){
            cout<<"ERROR en Cross_Validation: Los parametros de inicio y fin son los mismos o los de inicio son mayores a los de fin"<<endl;
            return 1;
        }
        bool negativa;
        int num_etiq=ax.numero_etiquetas(Labels,negativa);
        Confus=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
        float min_error=99999999;
        Parametros param=inicio;
        while(param.Neuronal_bp_dw_scale<=fin.Neuronal_bp_dw_scale && param.Neuronal_bp_moment_scale<=fin.Neuronal_bp_moment_scale && param.Neuronal_rp_dw0<=fin.Neuronal_rp_dw0 && param.Neuronal_rp_dw_max<=fin.Neuronal_rp_dw_max && param.Neuronal_rp_dw_min<=fin.Neuronal_rp_dw_min && param.Neuronal_rp_dw_minus<=fin.Neuronal_rp_dw_minus && param.Neuronal_rp_dw_plus<=fin.Neuronal_rp_dw_plus && param.Neuronal_fparam1<=fin.Neuronal_fparam1 && param.Neuronal_fparam2<=fin.Neuronal_fparam2){
            float total_error=0;
            cv::Mat Total_Confusion=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
            for(int i=0; i<Num_Folds; i++){
                Clasificador_Neuronal Neur("Cross_Validation",param.Neuronal_layerSize, param.Neuronal_Method, param.Neuronal_Function, param.Neuronal_bp_dw_scale, param.Neuronal_bp_moment_scale, param.Neuronal_rp_dw0, param.Neuronal_rp_dw_max, param.Neuronal_rp_dw_min, param.Neuronal_rp_dw_minus, param.Neuronal_rp_dw_plus, param.Neuronal_fparam1, param.Neuronal_fparam2);
#ifdef GUI
    Neur.progreso=progreso;
    Neur.max_progreso=max_progreso;
    Neur.base_progreso=base_progreso;
    Neur.total_progreso=total_progreso;
    Neur.window=window;
#endif
                e=Neur.Autotrain(datos_entrena[i],labels_entrena[i],reduc,info,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autotrain en Clasificador_Neuronal"<<endl;
                    return 1;
                }
                vector<float> result;
                e=Neur.Autoclasificacion(datos_valida[i],result,false,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autoclasificacion en Clasificador_Neuronal"<<endl;
                    return 1;
                }
                Mat Conf;
                float error=0;
                e=an.Confusion(labels_valida[i],result,Conf,error);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Confusion"<<endl;
                    return 1;
                }
                Total_Confusion=Total_Confusion+Conf;
                total_error=total_error+error;
            }
            total_error=total_error/Num_Folds;
            if(total_error<min_error){
                Total_Confusion.copyTo(Confus);
                min_error=total_error;
                parametros=param;
            }
            param.Neuronal_bp_dw_scale=param.Neuronal_bp_dw_scale+salto.Neuronal_bp_dw_scale;
            if(param.Neuronal_bp_dw_scale>fin.Neuronal_bp_dw_scale){
                param.Neuronal_bp_dw_scale=inicio.Neuronal_bp_dw_scale;
                param.Neuronal_bp_moment_scale=param.Neuronal_bp_moment_scale+salto.Neuronal_bp_moment_scale;
            }
            if(param.Neuronal_bp_moment_scale>fin.Neuronal_bp_moment_scale){
                param.Neuronal_bp_moment_scale=inicio.Neuronal_bp_moment_scale;
                param.Neuronal_rp_dw0=param.Neuronal_rp_dw0+salto.Neuronal_rp_dw0;
            }
            if(param.Neuronal_rp_dw0>fin.Neuronal_rp_dw0){
                param.Neuronal_rp_dw0=inicio.Neuronal_rp_dw0;
                param.Neuronal_rp_dw_max=param.Neuronal_rp_dw_max+salto.Neuronal_rp_dw_max;
            }
            if(param.Neuronal_rp_dw_max>fin.Neuronal_rp_dw_max){
                param.Neuronal_rp_dw_max=inicio.Neuronal_rp_dw_max;
                param.Neuronal_rp_dw_min=param.Neuronal_rp_dw_min+salto.Neuronal_rp_dw_min;
            }
            if(param.Neuronal_rp_dw_min>fin.Neuronal_rp_dw_min){
                param.Neuronal_rp_dw_min=inicio.Neuronal_rp_dw_min;
                param.Neuronal_rp_dw_minus=param.Neuronal_rp_dw_minus+salto.Neuronal_rp_dw_minus;
            }
            if(param.Neuronal_rp_dw_minus>fin.Neuronal_rp_dw_minus){
                param.Neuronal_rp_dw_minus=inicio.Neuronal_rp_dw_minus;
                param.Neuronal_rp_dw_plus=param.Neuronal_rp_dw_plus+salto.Neuronal_rp_dw_plus;
            }
            if(param.Neuronal_rp_dw_plus>fin.Neuronal_rp_dw_plus){
                param.Neuronal_rp_dw_plus=inicio.Neuronal_rp_dw_plus;
                param.Neuronal_fparam1=param.Neuronal_fparam1+salto.Neuronal_fparam1;
            }
            if(param.Neuronal_fparam1>fin.Neuronal_fparam1){
                param.Neuronal_fparam1=inicio.Neuronal_fparam1;
                param.Neuronal_fparam2=param.Neuronal_fparam2+salto.Neuronal_fparam2;
            }
        }
        Error=min_error;
        cout<<"Parametros optimizados"<<endl;
        cout<<"Neuronal_bp_dw_scale= "<<parametros.Neuronal_bp_dw_scale<<endl;
        cout<<"Neuronal_bp_moment_scale= "<<parametros.Neuronal_bp_moment_scale<<endl;
        cout<<"Neuronal_rp_dw0= "<<parametros.Neuronal_rp_dw0<<endl;
        cout<<"Neuronal_rp_dw_max= "<<parametros.Neuronal_rp_dw_max<<endl;
        cout<<"Neuronal_rp_dw_min= "<<parametros.Neuronal_rp_dw_min<<endl;
        cout<<"Neuronal_rp_dw_minus= "<<parametros.Neuronal_rp_dw_minus<<endl;
        cout<<"Neuronal_rp_dw_plus= "<<parametros.Neuronal_rp_dw_plus<<endl;
        cout<<"Neuronal_fparam1= "<<parametros.Neuronal_fparam1<<endl;
        cout<<"Neuronal_fparam2= "<<parametros.Neuronal_fparam2<<endl;
        cout<<"Error= "<<min_error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    else if(id_clasificador==C_SVM){
//        C, gamma, p, nu, coef0, degree
#ifdef WARNINGS
        std::cout << "WARNING: Opencv ya tiene un sistema de optimización de los parametros para el Clasificador_SVM llamado train_auto"<< std::endl;
#endif
        if(inicio.SVM_C>=fin.SVM_C && inicio.SVM_gamma>=fin.SVM_gamma && inicio.SVM_p>=fin.SVM_p && inicio.SVM_nu>=fin.SVM_nu && inicio.SVM_coef0>=fin.SVM_coef0 && inicio.SVM_degree>=fin.SVM_degree){
            cout<<"ERROR en Cross_Validation: Los parametros de inicio y fin son los mismos o los de inicio son mayores a los de fin"<<endl;
            return 1;
        }
        bool negativa;
        int num_etiq=ax.numero_etiquetas(Labels,negativa);
        Confus=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
        float min_error=99999999;
        Parametros param=inicio;
        while(param.SVM_C<=fin.SVM_C && param.SVM_gamma<=fin.SVM_gamma && param.SVM_p<=fin.SVM_p && param.SVM_nu<=fin.SVM_nu && param.SVM_coef0<=fin.SVM_coef0 && param.SVM_degree<=fin.SVM_degree){
            float total_error=0;
            cv::Mat Total_Confusion=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
            for(int i=0; i<Num_Folds; i++){
                Clasificador_SVM SVM("Cross_Validation",1, param.SVM_Type, param.SVM_kernel_type, param.SVM_class_weights, param.SVM_degree, param.SVM_gamma, param.SVM_coef0, param.SVM_C, param.SVM_nu, param.SVM_p);
#ifdef GUI
    SVM.progreso=progreso;
    SVM.max_progreso=max_progreso;
    SVM.base_progreso=base_progreso;
    SVM.total_progreso=total_progreso;
    SVM.window=window;
#endif
                e=SVM.Autotrain(datos_entrena[i],labels_entrena[i],reduc,info,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autotrain en Clasificador_SVM"<<endl;
                    return 1;
                }
                vector<float> result;
                e=SVM.Autoclasificacion(datos_valida[i],result,false,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autoclasificacion en Clasificador_SVM"<<endl;
                    return 1;
                }
                Mat Conf;
                float error=0;
                e=an.Confusion(labels_valida[i],result,Conf,error);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Confusion"<<endl;
                    return 1;
                }
                Total_Confusion=Total_Confusion+Conf;
                total_error=total_error+error;
            }
            total_error=total_error/Num_Folds;
            if(total_error<min_error){
                Total_Confusion.copyTo(Confus);
                min_error=total_error;
                parametros=param;
            }
            param.SVM_C=param.SVM_C+salto.SVM_C;
            if(param.SVM_C>fin.SVM_C){
                param.SVM_C=inicio.SVM_C;
                param.SVM_gamma=param.SVM_gamma+salto.SVM_gamma;
            }
            if(param.SVM_gamma>fin.SVM_gamma){
                param.SVM_gamma=inicio.SVM_gamma;
                param.SVM_p=param.SVM_p+salto.SVM_p;
            }
            if(param.SVM_p>fin.SVM_p){
                param.SVM_p=inicio.SVM_p;
                param.SVM_nu=param.SVM_nu+salto.SVM_nu;
            }
            if(param.SVM_nu>fin.SVM_nu){
                param.SVM_nu=inicio.SVM_nu;
                param.SVM_coef0=param.SVM_coef0+salto.SVM_coef0;
            }
            if(param.SVM_coef0>fin.SVM_coef0){
                param.SVM_coef0=inicio.SVM_coef0;
                param.SVM_degree=param.SVM_degree+salto.SVM_degree;
            }
        }
        Error=min_error;
        cout<<"Parametros optimizados"<<endl;
        cout<<"SVM_C= "<<parametros.SVM_C<<endl;
        cout<<"SVM_gamma= "<<parametros.SVM_gamma<<endl;
        cout<<"SVM_p= "<<parametros.SVM_p<<endl;
        cout<<"SVM_nu= "<<parametros.SVM_nu<<endl;
        cout<<"SVM_coef0= "<<parametros.SVM_coef0<<endl;
        cout<<"SVM_degree= "<<parametros.SVM_degree<<endl;
        cout<<"Error= "<<min_error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    else if(id_clasificador==RTREES){
//        max_depth, min_sample_count, regression_accuracy, max_categories, cv_folds,native_vars
        if(inicio.RTrees_max_depth>=fin.RTrees_max_depth && inicio.RTrees_min_sample_count>=fin.RTrees_min_sample_count && inicio.RTrees_regression_accuracy>=fin.RTrees_regression_accuracy && inicio.RTrees_max_categories>=fin.RTrees_max_categories
                && inicio.RTrees_cv_folds>=fin.RTrees_cv_folds && inicio.RTrees_native_vars>=fin.RTrees_native_vars){
            cout<<"ERROR en Cross_Validation: Los parametros de inicio y fin son los mismos o los de inicio son mayores a los de fin"<<endl;
            return 1;
        }
        bool negativa;
        int num_etiq=ax.numero_etiquetas(Labels,negativa);
        Confus=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
        float min_error=99999999;
        Parametros param=inicio;
        while(param.RTrees_max_depth<=fin.RTrees_max_depth && param.RTrees_min_sample_count<=fin.RTrees_min_sample_count && param.RTrees_regression_accuracy<=fin.RTrees_regression_accuracy && param.RTrees_max_categories<=fin.RTrees_max_categories && param.RTrees_cv_folds<=fin.RTrees_cv_folds && param.RTrees_native_vars<=fin.RTrees_native_vars){
            float total_error=0;
            cv::Mat Total_Confusion=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
            for(int i=0; i<Num_Folds; i++){
                Clasificador_RTrees Trees("Cross_Validation",param.RTrees_max_depth, param.RTrees_min_sample_count, param.RTrees_regression_accuracy, param.RTrees_use_surrogates, param.RTrees_max_categories, param.RTrees_cv_folds, param.RTrees_use_1se_rule, param.RTrees_truncate_pruned_tree, param.RTrees_priors,param.RTrees_calc_var_importance, param.RTrees_native_vars);
#ifdef GUI
    Trees.progreso=progreso;
    Trees.max_progreso=max_progreso;
    Trees.base_progreso=base_progreso;
    Trees.total_progreso=total_progreso;
    Trees.window=window;
#endif
                e=Trees.Autotrain(datos_entrena[i],labels_entrena[i],reduc,info,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autotrain en Clasificador_RTrees"<<endl;
                    return 1;
                }
                vector<float> result;
                e=Trees.Autoclasificacion(datos_valida[i],result,false,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autoclasificacion en Clasificador_RTrees"<<endl;
                    return 1;
                }
                Mat Conf;
                float error=0;
                e=an.Confusion(labels_valida[i],result,Conf,error);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Confusion"<<endl;
                    return 1;
                }
                Total_Confusion=Total_Confusion+Conf;
                total_error=total_error+error;
            }
            total_error=total_error/Num_Folds;
            if(total_error<min_error){
                Total_Confusion.copyTo(Confus);
                min_error=total_error;
                parametros=param;
            }
            param.RTrees_max_depth=param.RTrees_max_depth+salto.RTrees_max_depth;
            if(param.RTrees_max_depth>fin.RTrees_max_depth){
                param.RTrees_max_depth=inicio.RTrees_max_depth;
                param.RTrees_min_sample_count=param.RTrees_min_sample_count+salto.RTrees_min_sample_count;
            }
            if(param.RTrees_min_sample_count>fin.RTrees_min_sample_count){
                param.RTrees_min_sample_count=inicio.RTrees_min_sample_count;
                param.RTrees_regression_accuracy=param.RTrees_regression_accuracy+salto.RTrees_regression_accuracy;
            }
            if(param.RTrees_regression_accuracy>fin.RTrees_regression_accuracy){
                param.RTrees_regression_accuracy=inicio.RTrees_regression_accuracy;
                param.RTrees_max_categories=param.RTrees_max_categories+salto.RTrees_max_categories;
            }
            if(param.RTrees_max_categories>fin.RTrees_max_categories){
                param.RTrees_max_categories=inicio.RTrees_max_categories;
                param.RTrees_cv_folds=param.RTrees_cv_folds+salto.RTrees_cv_folds;
            }
            if(param.RTrees_cv_folds>fin.RTrees_cv_folds){
                param.RTrees_cv_folds=inicio.RTrees_cv_folds;
                param.RTrees_native_vars=param.RTrees_native_vars+salto.RTrees_native_vars;
            }
        }
        Error=min_error;
        cout<<"Parametros optimizados"<<endl;
        cout<<"RTrees_max_depth= "<<parametros.RTrees_max_depth<<endl;
        cout<<"RTrees_min_sample_count= "<<parametros.RTrees_min_sample_count<<endl;
        cout<<"RTrees_regression_accuracy= "<<parametros.RTrees_regression_accuracy<<endl;
        cout<<"RTrees_max_categories= "<<parametros.RTrees_max_categories<<endl;
        cout<<"RTrees_cv_folds= "<<parametros.RTrees_cv_folds<<endl;
        cout<<"RTrees_native_vars= "<<parametros.RTrees_native_vars<<endl;
        cout<<"Error= "<<min_error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    else if(id_clasificador==DTREES){
//        max_depth, min_sample_count, regression_accuracy, max_categories, cv_folds
        if(inicio.DTrees_max_depth>=fin.DTrees_max_depth && inicio.DTrees_min_sample_count>=fin.DTrees_min_sample_count && inicio.DTrees_regression_accuracy>=fin.DTrees_regression_accuracy && inicio.DTrees_max_categories>=fin.DTrees_max_categories
                && inicio.DTrees_cv_folds>=fin.DTrees_cv_folds){
            cout<<"ERROR en Cross_Validation: Los parametros de inicio y fin son los mismos o los de inicio son mayores a los de fin"<<endl;
            return 1;
        }
        bool negativa;
        int num_etiq=ax.numero_etiquetas(Labels,negativa);
        Confus=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
        float min_error=99999999;
        Parametros param=inicio;
        while(param.DTrees_max_depth<=fin.DTrees_max_depth && param.DTrees_min_sample_count<=fin.DTrees_min_sample_count && param.DTrees_regression_accuracy<=fin.DTrees_regression_accuracy && param.DTrees_max_categories<=fin.DTrees_max_categories && param.DTrees_cv_folds<=fin.DTrees_cv_folds){
            float total_error=0;
            cv::Mat Total_Confusion=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
            for(int i=0; i<Num_Folds; i++){
                Clasificador_DTrees DT("Cross_Validation",param.DTrees_max_depth, param.DTrees_min_sample_count, param.DTrees_regression_accuracy, param.DTrees_use_surrogates, param.DTrees_max_categories, param.DTrees_cv_folds, param.DTrees_use_1se_rule, param.DTrees_truncate_pruned_tree, param.DTrees_priors);
#ifdef GUI
    DT.progreso=progreso;
    DT.max_progreso=max_progreso;
    DT.base_progreso=base_progreso;
    DT.total_progreso=total_progreso;
    DT.window=window;
#endif
                e=DT.Autotrain(datos_entrena[i],labels_entrena[i],reduc,info,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autotrain en Clasificador_DTrees"<<endl;
                    return 1;
                }
                vector<float> result;
                e=DT.Autoclasificacion(datos_valida[i],result,false,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autoclasificacion en Clasificador_DTrees"<<endl;
                    return 1;
                }
                Mat Conf;
                float error=0;
                e=an.Confusion(labels_valida[i],result,Conf,error);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Confusion"<<endl;
                    return 1;
                }
                Total_Confusion=Total_Confusion+Conf;
                total_error=total_error+error;
            }
            total_error=total_error/Num_Folds;
            if(total_error<min_error){
                Total_Confusion.copyTo(Confus);
                min_error=total_error;

            }
            param.DTrees_max_depth=param.DTrees_max_depth+salto.DTrees_max_depth;
            if(param.DTrees_max_depth>fin.DTrees_max_depth){
                param.DTrees_max_depth=inicio.DTrees_max_depth;
                param.DTrees_min_sample_count=param.DTrees_min_sample_count+salto.DTrees_min_sample_count;
            }
            if(param.DTrees_min_sample_count>fin.DTrees_min_sample_count){
                param.DTrees_min_sample_count=inicio.DTrees_min_sample_count;
                param.DTrees_regression_accuracy=param.DTrees_regression_accuracy+salto.DTrees_regression_accuracy;
            }
            if(param.DTrees_regression_accuracy>fin.DTrees_regression_accuracy){
                param.DTrees_regression_accuracy=inicio.DTrees_regression_accuracy;
                param.DTrees_max_categories=param.DTrees_max_categories+salto.DTrees_max_categories;
            }
            if(param.DTrees_max_categories>fin.DTrees_max_categories){
                param.DTrees_max_categories=inicio.DTrees_max_categories;
                param.DTrees_cv_folds=param.DTrees_cv_folds+salto.DTrees_cv_folds;
            }
        }
        Error=min_error;
        cout<<"Parametros optimizados"<<endl;
        cout<<"DTrees_max_depth= "<<parametros.DTrees_max_depth<<endl;
        cout<<"DTrees_min_sample_count= "<<parametros.DTrees_min_sample_count<<endl;
        cout<<"DTrees_regression_accuracy= "<<parametros.DTrees_regression_accuracy<<endl;
        cout<<"DTrees_max_categories= "<<parametros.DTrees_max_categories<<endl;
        cout<<"DTrees_cv_folds= "<<parametros.DTrees_cv_folds<<endl;
        cout<<"Error= "<<min_error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    else if(id_clasificador==BOOSTING){
//        max_depth, weak_count
        if(inicio.Boosting_max_depth>=fin.Boosting_max_depth && inicio.Boosting_weak_count>=fin.Boosting_weak_count){
            cout<<"ERROR en Cross_Validation: Los parametros de inicio y fin son los mismos o los de inicio son mayores a los de fin"<<endl;
            return 1;
        }
        bool negativa;
        int num_etiq=ax.numero_etiquetas(Labels,negativa);
        Confus=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
        float min_error=99999999;
        Parametros param=inicio;
        while(param.Boosting_max_depth<=fin.Boosting_max_depth && param.Boosting_weak_count<=fin.Boosting_weak_count){
            float total_error=0;
            cv::Mat Total_Confusion=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
            for(int i=0; i<Num_Folds; i++){
                Clasificador_Boosting B("Cross_Validation",param.Boosting_boost_type,param.Boosting_weak_count,param.Boosting_weight_trim_rate,param.Boosting_max_depth,param.Boosting_use_surrogates,param.Boosting_priors);
#ifdef GUI
    B.progreso=progreso;
    B.max_progreso=max_progreso;
    B.base_progreso=base_progreso;
    B.total_progreso=total_progreso;
    B.window=window;
#endif
                e=B.Autotrain(datos_entrena[i],labels_entrena[i],reduc,info,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autotrain en Clasificador_Boosting"<<endl;
                    return 1;
                }
                vector<float> result;
                e=B.Autoclasificacion(datos_valida[i],result,false,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autoclasificacion en Clasificador_Boosting"<<endl;
                    return 1;
                }
                Mat Conf;
                float error=0;
                e=an.Confusion(labels_valida[i],result,Conf,error);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Confusion"<<endl;
                    return 1;
                }
                Total_Confusion=Total_Confusion+Conf;
                total_error=total_error+error;
            }
            total_error=total_error/Num_Folds;
            if(total_error<min_error){
                Total_Confusion.copyTo(Confus);
                min_error=total_error;

            }
            param.Boosting_max_depth=param.Boosting_max_depth+salto.Boosting_max_depth;
            if(param.Boosting_max_depth>fin.Boosting_max_depth){
                param.Boosting_max_depth=inicio.Boosting_max_depth;
                param.Boosting_weak_count=param.Boosting_weak_count+salto.Boosting_weak_count;
            }
        }
        Error=min_error;
        cout<<"Parametros optimizados"<<endl;
        cout<<"Boosting_max_depth= "<<parametros.Boosting_max_depth<<endl;
        cout<<"Boosting_weak_count= "<<parametros.Boosting_weak_count<<endl;
        cout<<"Error= "<<min_error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
//    else if(id_clasificador==GBT){
////        weak_count, shrinkage, max_depth
//        if(inicio.GBT_weak_count>=fin.GBT_weak_count && inicio.GBT_shrinkage>=fin.GBT_shrinkage && inicio.GBT_max_depth>=fin.GBT_max_depth){
//            cout<<"ERROR en Cross_Validation: Los parametros de inicio y fin son los mismos o los de inicio son mayores a los de fin"<<endl;
//            return 1;
//        }
//        bool negativa;
//        int num_etiq=ax.numero_etiquetas(Labels,negativa);
//        Confus=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
//        float min_error=99999999;
//        Parametros param=inicio;
//        while(param.GBT_weak_count<=fin.GBT_weak_count && param.GBT_shrinkage<=fin.GBT_shrinkage && param.GBT_max_depth<=fin.GBT_max_depth){
//            float total_error=0;
//            cv::Mat Total_Confusion=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
//            for(int i=0; i<Num_Folds; i++){
//                Clasificador_GBT GB("Cross_Validation",param.GBT_loss_function_type,param.GBT_weak_count,param.GBT_shrinkage,param.GBT_subsample_portion,param.GBT_max_depth, param.GBT_use_surrogates);
//#ifdef GUI
//    GB.progreso=progreso;
//    GB.max_progreso=max_progreso;
//    GB.base_progreso=base_progreso;
//    GB.total_progreso=total_progreso;
//    GB.window=window;
//#endif
//                e=GB.Autotrain(datos_entrena[i],labels_entrena[i],reduc,info,false);
//                if(e==1){
//                    cout<<"ERROR en Cross_Validation: Error en Autotrain en Clasificador_GBT"<<endl;
//                    return 1;
//                }
//                vector<float> result;
//                e=GB.Autoclasificacion(datos_valida[i],result,false,false);
//                if(e==1){
//                    cout<<"ERROR en Cross_Validation: Error en Autoclasificacion en Clasificador_GBT"<<endl;
//                    return 1;
//                }
//                Mat Conf;
//                float error=0;
//                e=an.Confusion(labels_valida[i],result,Conf,error);
//                if(e==1){
//                    cout<<"ERROR en Cross_Validation: Error en Confusion"<<endl;
//                    return 1;
//                }
//                Total_Confusion=Total_Confusion+Conf;
//                total_error=total_error+error;
//            }
//            total_error=total_error/Num_Folds;
//            if(total_error<min_error){
//                Total_Confusion.copyTo(Confus);
//                min_error=total_error;
//                parametros=param;
//            }
//            param.GBT_weak_count=param.GBT_weak_count+salto.GBT_weak_count;
//            if(param.GBT_weak_count>fin.GBT_weak_count){
//                param.GBT_weak_count=inicio.GBT_weak_count;
//                param.GBT_shrinkage=param.GBT_shrinkage+salto.GBT_shrinkage;
//            }
//            if(param.GBT_shrinkage>fin.GBT_shrinkage){
//                param.GBT_shrinkage=inicio.GBT_shrinkage;
//                param.GBT_max_depth=param.GBT_max_depth+salto.GBT_max_depth;
//            }

//        }
//        Error=min_error;
//        cout<<"Parametros optimizados"<<endl;
//        cout<<"GBT_weak_count= "<<parametros.GBT_weak_count<<endl;
//        cout<<"GBT_shrinkage= "<<parametros.GBT_shrinkage<<endl;
//        cout<<"GBT_max_depth= "<<parametros.GBT_max_depth<<endl;
//        cout<<"Error= "<<min_error<<endl;
//        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
//    }
    else if(id_clasificador==EXP_MAX){
//        nclusters
        if(inicio.EM_nclusters>=fin.EM_nclusters){
            cout<<"ERROR en Cross_Validation: Los parametros de inicio y fin son los mismos o los de inicio son mayores a los de fin"<<endl;
            return 1;
        }
        bool negativa;
        int num_etiq=ax.numero_etiquetas(Labels,negativa);
        Confus=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
        float min_error=99999999;
        while(inicio.EM_nclusters<=fin.EM_nclusters){
            float total_error=0;
            cv::Mat Total_Confusion=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
            for(int i=0; i<Num_Folds; i++){
                Clasificador_EM E("Validation",inicio.EM_nclusters, inicio.EM_covMatType);
#ifdef GUI
    E.progreso=progreso;
    E.max_progreso=max_progreso;
    E.base_progreso=base_progreso;
    E.total_progreso=total_progreso;
    E.window=window;
#endif
                e=E.Autotrain(datos_entrena[i],labels_entrena[i],reduc,info,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autotrain en Clasificador_EM"<<endl;
                    return 1;
                }
                vector<float> result;
                e=E.Autoclasificacion(datos_valida[i],result,false,false);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Autoclasificacion en Clasificador_EM"<<endl;
                    return 1;
                }
                Mat Conf;
                float error=0;
                e=an.Confusion(labels_valida[i],result,Conf,error);
                if(e==1){
                    cout<<"ERROR en Cross_Validation: Error en Confusion"<<endl;
                    return 1;
                }
                Total_Confusion=Total_Confusion+Conf;
                total_error=total_error+error;
            }
            total_error=total_error/Num_Folds;
            if(total_error<min_error){
                Total_Confusion.copyTo(Confus);
                min_error=total_error;
                parametros=inicio;
            }
            inicio.EM_nclusters=inicio.EM_nclusters+salto.EM_nclusters;
        }
        Error=min_error;
        cout<<"Parametros optimizados"<<endl;
        cout<<"EM_nclusters= "<<parametros.EM_nclusters<<endl;
        cout<<"Error= "<<min_error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
//    else if(id_clasificador==ERTREES){
////        max_depth, min_sample_count, regression_accuracy, max_categories, cv_folds,native_vars
//        if(inicio.ERTrees_max_depth>=fin.ERTrees_max_depth && inicio.ERTrees_min_sample_count>=fin.ERTrees_min_sample_count && inicio.ERTrees_regression_accuracy>=fin.ERTrees_regression_accuracy && inicio.ERTrees_max_categories>=fin.ERTrees_max_categories
//                && inicio.ERTrees_cv_folds>=fin.ERTrees_cv_folds && inicio.ERTrees_native_vars>=fin.ERTrees_native_vars){
//            cout<<"ERROR en Cross_Validation: Los parametros de inicio y fin son los mismos o los de inicio son mayores a los de fin"<<endl;
//            return 1;
//        }
//        bool negativa;
//        int num_etiq=ax.numero_etiquetas(Labels,negativa);
//        Confus=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
//        float min_error=99999999;
//        Parametros param=inicio;
//        while(param.RTrees_max_depth<=fin.RTrees_max_depth && param.RTrees_min_sample_count<=fin.RTrees_min_sample_count && param.RTrees_regression_accuracy<=fin.RTrees_regression_accuracy && param.RTrees_max_categories<=fin.RTrees_max_categories && param.RTrees_cv_folds<=fin.RTrees_cv_folds && param.RTrees_native_vars<=fin.RTrees_native_vars){
//            float total_error=0;
//            cv::Mat Total_Confusion=Mat::zeros(num_etiq,num_etiq,CV_32FC1);
//            for(int i=0; i<Num_Folds; i++){
//                Clasificador_ERTrees ERT("Cross_Validation",param.ERTrees_max_depth, param.ERTrees_min_sample_count, param.ERTrees_regression_accuracy, param.ERTrees_use_surrogates, param.ERTrees_max_categories, param.ERTrees_cv_folds, param.ERTrees_use_1se_rule, param.ERTrees_truncate_pruned_tree, param.ERTrees_priors,param.ERTrees_calc_var_importance, param.ERTrees_native_vars);
//#ifdef GUI
//    ERT.progreso=progreso;
//    ERT.max_progreso=max_progreso;
//    ERT.base_progreso=base_progreso;
//    ERT.total_progreso=total_progreso;
//    ERT.window=window;
//#endif
//                e=ERT.Autotrain(datos_entrena[i],labels_entrena[i],reduc,info,false);
//                if(e==1){
//                    cout<<"ERROR en Cross_Validation: Error en Autotrain en Clasificador_ERTrees"<<endl;
//                    return 1;
//                }
//                vector<float> result;
//                e=ERT.Autoclasificacion(datos_valida[i],result,false,false);
//                if(e==1){
//                    cout<<"ERROR en Cross_Validation: Error en Autoclasificacion en Clasificador_ERTrees"<<endl;
//                    return 1;
//                }
//                Mat Conf;
//                float error=0;
//                e=an.Confusion(labels_valida[i],result,Conf,error);
//                if(e==1){
//                    cout<<"ERROR en Cross_Validation: Error en Confusion"<<endl;
//                    return 1;
//                }
//                Total_Confusion=Total_Confusion+Conf;
//                total_error=total_error+error;
//            }
//            total_error=total_error/Num_Folds;
//            if(total_error<min_error){
//                Total_Confusion.copyTo(Confus);
//                min_error=total_error;
//                parametros=param;
//            }
//            param.ERTrees_max_depth=param.ERTrees_max_depth+salto.ERTrees_max_depth;
//            if(param.ERTrees_max_depth>fin.ERTrees_max_depth){
//                param.ERTrees_max_depth=inicio.ERTrees_max_depth;
//                param.ERTrees_min_sample_count=param.ERTrees_min_sample_count+salto.ERTrees_min_sample_count;
//            }
//            if(param.ERTrees_min_sample_count>fin.ERTrees_min_sample_count){
//                param.ERTrees_min_sample_count=inicio.ERTrees_min_sample_count;
//                param.ERTrees_regression_accuracy=param.ERTrees_regression_accuracy+salto.ERTrees_regression_accuracy;
//            }
//            if(param.ERTrees_regression_accuracy>fin.ERTrees_regression_accuracy){
//                param.ERTrees_regression_accuracy=inicio.ERTrees_regression_accuracy;
//                param.ERTrees_max_categories=param.ERTrees_max_categories+salto.ERTrees_max_categories;
//            }
//            if(param.ERTrees_max_categories>fin.ERTrees_max_categories){
//                param.ERTrees_max_categories=inicio.ERTrees_max_categories;
//                param.ERTrees_cv_folds=param.ERTrees_cv_folds+salto.ERTrees_cv_folds;
//            }
//            if(param.ERTrees_cv_folds>fin.ERTrees_cv_folds){
//                param.ERTrees_cv_folds=inicio.ERTrees_cv_folds;
//                param.ERTrees_native_vars=param.ERTrees_native_vars+salto.ERTrees_native_vars;
//            }
//        }
//        Error=min_error;
//        cout<<"Parametros optimizados"<<endl;
//        cout<<"ERTrees_max_depth= "<<parametros.ERTrees_max_depth<<endl;
//        cout<<"ERTrees_min_sample_count= "<<parametros.ERTrees_min_sample_count<<endl;
//        cout<<"ERTrees_regression_accuracy= "<<parametros.ERTrees_regression_accuracy<<endl;
//        cout<<"ERTrees_max_categories= "<<parametros.ERTrees_max_categories<<endl;
//        cout<<"ERTrees_cv_folds= "<<parametros.ERTrees_cv_folds<<endl;
//        cout<<"ERTrees_native_vars= "<<parametros.ERTrees_native_vars<<endl;
//        cout<<"Error= "<<min_error<<endl;
//        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
//    }
    else{
        cout<<"ERROR en Cross_Validation: id_clasificador erroneo"<<endl;
        return 1;
    }
    return 0;
}

int MLT::Optimizacion::Super_Cross_Validation(vector<Mat> Datos, vector<float> Labels, int Num_Folds, int Tam_Fold, vector<int> &id_clasificador, Parametros inicio, Parametros fin, Parametros salto, Parametros &parametros, float &Error, Mat &Confus){
    int e=0;
    if(Datos.size()==0){
        cout<<"ERROR en Super_Cross_Validation: No hay datos"<<endl;
        return 1;
    }
    if(Labels.size()==0){
        cout<<"ERROR en Super_Cross_Validation: No hay Etiquetas"<<endl;
        return 1;
    }
    if(Datos.size()!=Labels.size()){
        cout<<"ERROR en Super_Cross_Validation: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
    if((uint)Num_Folds*Tam_Fold>Datos.size()){
        cout<<"ERROR en Super_Cross_Validation: Numero de datos menor de lo que se pide para el proceso"<<endl;
        return 1;
    }
    if(id_clasificador.size()==0){
        cout<<"ERROR en Super_Cross_Validation: Numero de clasificadores elegido igual a cero"<<endl;
        return 1;
    }
    for(uint i=0; i<Labels.size(); i++){
        if(Labels[i]==0){
            cout<<"ERROR en Super_Cross_Validation: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    vector<bool> clasificadores(NUM_CLASIF);
    for(uint i=0; i<clasificadores.size(); i++)
        clasificadores[i]=false;
    for(uint i=0; i<id_clasificador.size(); i++){
        if(id_clasificador[i]<0 || id_clasificador[i]>NUM_CLASIF){
            cout<<"ERROR en Super_Cross_Validation: id_clasificador erroneo"<<endl;
            return 1;
        }
        clasificadores[id_clasificador[i]]=true;
    }
    Error=999999999;
    int id=0;
    if(clasificadores[0]==true){
        cout<<endl<<endl<<"Clasificador_Distancias"<<endl;
        Parametros param;
        float error;
        Mat confus;
        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,DISTANCIAS,inicio,fin,salto,param, error, confus);
        if(e==1){
            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
        }
        if(error<Error){
            Error=error;
            Confus=confus;
            id=DISTANCIAS;
            parametros=param;
        }
    }
    if(clasificadores[1]==true){
        cout<<endl<<endl<<"Clasificador_Gaussiano"<<endl;
        Parametros param;
        float error;
        Mat confus;
        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,GAUSSIANO,inicio,fin,salto,param, error, confus);
        if(e==1){
            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
        }
        if(error<Error){
            Error=error;
            Confus=confus;
            id=GAUSSIANO;
            parametros=param;
        }
    }
    if(clasificadores[2]==true){
        cout<<endl<<endl<<"WARNING en Super_Cross_Validation: NO SE HA IMPLEMENTADO AUN EL Clasificador_Cascada EN Super_Cross_Validation"<<endl;
//        cout<<endl<<endl<<"Clasificador_HAAR"<<endl;
//        Parametros param;
//        float error;
//        Mat confus;
//        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,CASCADA_CLAS,reduc,inicio,fin,salto,param, error, confus);
//        if(e==1){
//            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
//        }
//        if(error<min_error){
//            Error=error;
//            Confus=confus;
//            id=CASCADA_CLAS;
//            parametros=param;
//        }
    }
    if(clasificadores[3]==true){
        cout<<endl<<endl<<"Clasificador_Histograma"<<endl;
        Parametros param;
        float error;
        Mat confus;
        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,HISTOGRAMA,inicio,fin,salto,param, error, confus);
        if(e==1){
            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
        }
        if(error<Error){
            Error=error;
            Confus=confus;
            id=HISTOGRAMA;
            parametros=param;
        }
    }
    if(clasificadores[4]==true){
        cout<<endl<<endl<<"Clasificador_KNN"<<endl;
        Parametros param;
        float error;
        Mat confus;
        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,KNN,inicio,fin,salto,param, error, confus);
        if(e==1){
            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
        }
        if(error<Error){
            Error=error;
            Confus=confus;
            id=KNN;
            parametros=param;
        }
    }
    if(clasificadores[5]==true){
        cout<<endl<<endl<<"Clasificador_Neuronal"<<endl;
        Parametros param;
        float error;
        Mat confus;
        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,NEURONAL,inicio,fin,salto,param, error, confus);
        if(e==1){
            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
        }
        if(error<Error){
            Error=error;
            Confus=confus;
            id=NEURONAL;
            parametros=param;
        }
    }
    if(clasificadores[6]==true){
        cout<<endl<<endl<<"Clasificador_SVM"<<endl;
        Parametros param;
        float error;
        Mat confus;
        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,C_SVM,inicio,fin,salto,param, error, confus);
        if(e==1){
            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
        }
        if(error<Error){
            Error=error;
            Confus=confus;
            id=C_SVM;
            parametros=param;
        }
    }
    if(clasificadores[7]==true){
        cout<<endl<<endl<<"Clasificador_RTrees"<<endl;
        Parametros param;
        float error;
        Mat confus;
        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,RTREES,inicio,fin,salto,param, error, confus);
        if(e==1){
            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
        }
        if(error<Error){
            Error=error;
            Confus=confus;
            id=RTREES;
            parametros=param;
        }
    }
    if(clasificadores[8]==true){
        cout<<endl<<endl<<"Clasificador_DTrees"<<endl;
        Parametros param;
        float error;
        Mat confus;
        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,DTREES,inicio,fin,salto,param, error, confus);
        if(e==1){
            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
        }
        if(error<Error){
            Error=error;
            Confus=confus;
            id=DTREES;
            parametros=param;
        }
    }
    if(clasificadores[9]==true){
        cout<<endl<<endl<<"Clasificador_Boosting"<<endl;
        Parametros param;
        float error;
        Mat confus;
        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,BOOSTING,inicio,fin,salto,param, error, confus);
        if(e==1){
            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
        }
        if(error<Error){
            Error=error;
            Confus=confus;
            id=BOOSTING;
            parametros=param;
        }
    }
//    if(clasificadores[10]==true){
//        cout<<endl<<endl<<"Clasificador_GBT"<<endl;
//        Parametros param;
//        float error;
//        Mat confus;
//        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,GBT,inicio,fin,salto,param, error, confus);
//        if(e==1){
//            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
//        }
//        if(error<Error){
//            Error=error;
//            Confus=confus;
//            id=GBT;
//            parametros=param;
//        }
//    }
    if(clasificadores[11]==true){
        cout<<endl<<endl<<"Clasificador_EM"<<endl;
        Parametros param;
        float error;
        Mat confus;
        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,EXP_MAX,inicio,fin,salto,param, error, confus);
        if(e==1){
            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
        }
        if(error<Error){
            Error=error;
            Confus=confus;
            id=EXP_MAX;
            parametros=param;
        }
    }
//    if(clasificadores[12]==true){
//        cout<<endl<<endl<<"Clasificador_ERTrees"<<endl;
//        Parametros param;
//        float error;
//        Mat confus;
//        e=Cross_Validation(Datos,Labels,Num_Folds,Tam_Fold,ERTREES,inicio,fin,salto,param, error, confus);
//        if(e==1){
//            cout<<"ERROR en Super_Cross_Validation: Error en Cross_Validation"<<endl;
//        }
//        if(error<Error){
//            Error=error;
//            Confus=confus;
//            id=ERTREES;
//            parametros=param;
//        }
//    }
    id_clasificador.clear();
    cout<<endl<<endl;
    if(id==DISTANCIAS){
        id_clasificador.push_back(DISTANCIAS);
        cout<<"El mejor clasificador es Clasificador_Distancias"<<endl;
    }
    if(id==GAUSSIANO){
        id_clasificador.push_back(GAUSSIANO);
        cout<<"El mejor clasificador es Clasificador_Gaussiano"<<endl;
    }
//    if(id==CASCADA_CLAS){
//        id_clasificador.push_back(CASCADA_CLAS);
//        cout<<"El mejor clasificador es Clasificador_HAAR"<<endl;
//    }
    if(id==HISTOGRAMA){
        id_clasificador.push_back(HISTOGRAMA);
        cout<<"El mejor clasificador es Clasificador_Histograma"<<endl;
        cout<<"Parametros optimizados"<<endl;
        cout<<"Hist_tam_celdea= "<<parametros.Hist_tam_celda<<endl;
        cout<<"Error= "<<Error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    if(id==KNN){
        id_clasificador.push_back(KNN);
        cout<<"El mejor clasificador es Clasificador_KNN"<<endl;
        cout<<"Parametros optimizados"<<endl;
        cout<<"KNN_k= "<<parametros.KNN_k<<endl;
        cout<<"Error= "<<Error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    if(id==NEURONAL){
        id_clasificador.push_back(NEURONAL);
        cout<<"El mejor clasificador es Clasificador_Neuronal"<<endl;
        cout<<"Neuronal_rp_dw0= "<<parametros.Neuronal_rp_dw0<<endl;
        cout<<"Neuronal_rp_dw_max= "<<parametros.Neuronal_rp_dw_max<<endl;
        cout<<"Neuronal_rp_dw_min= "<<parametros.Neuronal_rp_dw_min<<endl;
        cout<<"Neuronal_rp_dw_minus= "<<parametros.Neuronal_rp_dw_minus<<endl;
        cout<<"Neuronal_rp_dw_plus= "<<parametros.Neuronal_rp_dw_plus<<endl;
        cout<<"Neuronal_fparam1= "<<parametros.Neuronal_fparam1<<endl;
        cout<<"Neuronal_fparam2= "<<parametros.Neuronal_fparam2<<endl;
        cout<<"Error= "<<Error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    if(id==C_SVM){
        id_clasificador.push_back(C_SVM);
        cout<<"El mejor clasificador es Clasificador_SVM"<<endl;
        cout<<"Parametros optimizados"<<endl;
        cout<<"SVM_C= "<<parametros.SVM_C<<endl;
        cout<<"SVM_gamma= "<<parametros.SVM_gamma<<endl;
        cout<<"SVM_p= "<<parametros.SVM_p<<endl;
        cout<<"SVM_nu= "<<parametros.SVM_nu<<endl;
        cout<<"SVM_coef0= "<<parametros.SVM_coef0<<endl;
        cout<<"SVM_degree= "<<parametros.SVM_degree<<endl;
        cout<<"Error= "<<Error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    if(id==RTREES){
        id_clasificador.push_back(RTREES);
        cout<<"El mejor clasificador es Clasificador_RTrees"<<endl;
        cout<<"Parametros optimizados"<<endl;
        cout<<"RTrees_max_depth= "<<parametros.RTrees_max_depth<<endl;
        cout<<"RTrees_min_sample_count= "<<parametros.RTrees_min_sample_count<<endl;
        cout<<"RTrees_regression_accuracy= "<<parametros.RTrees_regression_accuracy<<endl;
        cout<<"RTrees_max_categories= "<<parametros.RTrees_max_categories<<endl;
        cout<<"RTrees_cv_folds= "<<parametros.RTrees_cv_folds<<endl;
        cout<<"RTrees_native_vars= "<<parametros.RTrees_native_vars<<endl;
        cout<<"Error= "<<Error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    if(id==DTREES){
        id_clasificador.push_back(DTREES);
        cout<<"El mejor clasificador es Clasificador_DTrees"<<endl;
        cout<<"Parametros optimizados"<<endl;
        cout<<"DTrees_max_depth= "<<parametros.DTrees_max_depth<<endl;
        cout<<"DTrees_min_sample_count= "<<parametros.DTrees_min_sample_count<<endl;
        cout<<"DTrees_regression_accuracy= "<<parametros.DTrees_regression_accuracy<<endl;
        cout<<"DTrees_max_categories= "<<parametros.DTrees_max_categories<<endl;
        cout<<"DTrees_cv_folds= "<<parametros.DTrees_cv_folds<<endl;
        cout<<"Error= "<<Error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
    if(id==BOOSTING){
        id_clasificador.push_back(BOOSTING);
        cout<<"El mejor clasificador es Clasificador_Boosting"<<endl;
        cout<<"Parametros optimizados"<<endl;
        cout<<"Boosting_max_depth= "<<parametros.Boosting_max_depth<<endl;
        cout<<"Boosting_weak_count= "<<parametros.Boosting_weak_count<<endl;
        cout<<"Error= "<<Error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
//    if(id==GBT){
//        id_clasificador.push_back(GBT);
//        cout<<"El mejor clasificador es Clasificador_GBT"<<endl;
//        cout<<"Parametros optimizados"<<endl;
//        cout<<"GBT_weak_count= "<<parametros.GBT_weak_count<<endl;
//        cout<<"GBT_shrinkage= "<<parametros.GBT_shrinkage<<endl;
//        cout<<"GBT_max_depth= "<<parametros.GBT_max_depth<<endl;
//        cout<<"Error= "<<Error<<endl;
//        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
//    }
    if(id==EXP_MAX){
        id_clasificador.push_back(EXP_MAX);
        cout<<"El mejor clasificador es Clasificador_EM"<<endl;
        cout<<"Parametros optimizados"<<endl;
        cout<<"EM_nclusters= "<<parametros.EM_nclusters<<endl;
        cout<<"Error= "<<Error<<endl;
        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
    }
//    if(id==ERTREES){
//        id_clasificador.push_back(ERTREES);
//        cout<<"El mejor clasificador es Clasificador_ERTrees"<<endl;
//        cout<<"Parametros optimizados"<<endl;
//        cout<<"ERTrees_max_depth= "<<parametros.ERTrees_max_depth<<endl;
//        cout<<"ERTrees_min_sample_count= "<<parametros.ERTrees_min_sample_count<<endl;
//        cout<<"ERTrees_regression_accuracy= "<<parametros.ERTrees_regression_accuracy<<endl;
//        cout<<"ERTrees_max_categories= "<<parametros.ERTrees_max_categories<<endl;
//        cout<<"ERTrees_cv_folds= "<<parametros.ERTrees_cv_folds<<endl;
//        cout<<"ERTrees_native_vars= "<<parametros.ERTrees_native_vars<<endl;
//        cout<<"Error= "<<Error<<endl;
//        cout<<"Matriz Confusion= "<<endl<<Confus<<endl;
//    }
    return 0;
}

int MLT::Optimizacion::Ratios_parametro(vector<Mat> Datos, vector<float> Labels, int porcentaje_validacion, string parametro, Parametros inicio, Parametros fin, Parametros salto, vector<vector<Analisis::Ratios_data> > &Ratios){
    int e=0;
    if(Datos.size()==0){
        cout<<"ERROR en Ratios_parametro: No hay datos"<<endl;
        return 1;
    }
    if(Labels.size()==0){
        cout<<"ERROR en Ratios_parametro: No hay Etiquetas"<<endl;
        return 1;
    }
    if(Datos.size()!=Labels.size()){
        cout<<"ERROR en Ratios_parametro: Numero de datos y etiquetas distinto"<<endl;
        return 1;
    }
    for(uint i=0; i<Labels.size(); i++){
        if(Labels[i]==0){
            cout<<"ERROR en Cross_Validation: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    if(parametro=="Hist_tam_celda"){
        while(inicio.Hist_tam_celda<=fin.Hist_tam_celda){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,HISTOGRAMA,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Hist_tam_celda=inicio.Hist_tam_celda+salto.Hist_tam_celda;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="KNN_k"){
        while(inicio.KNN_k<=fin.KNN_k){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            cout<<"KNN_K= "<<inicio.KNN_k<<endl;
            e=Validation(Datos,Labels,porcentaje_validacion,KNN,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.KNN_k=inicio.KNN_k+salto.KNN_k;
            Ratios.push_back(Rat);
            cout<<endl<<endl<<endl;
        }
    }
    else if(parametro=="Neuronal_bp_dw_scale"){
        while(inicio.Neuronal_bp_dw_scale<=fin.Neuronal_bp_dw_scale){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,NEURONAL,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Neuronal_bp_dw_scale=inicio.Neuronal_bp_dw_scale+salto.Neuronal_bp_dw_scale;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="Neuronal_bp_moment_scale"){
        while(inicio.Neuronal_bp_moment_scale<=fin.Neuronal_bp_moment_scale){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,NEURONAL,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Neuronal_bp_moment_scale=inicio.Neuronal_bp_moment_scale+salto.Neuronal_bp_moment_scale;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="Neuronal_fparam1"){
        while(inicio.Neuronal_fparam1<=fin.Neuronal_fparam1){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,NEURONAL,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Neuronal_fparam1=inicio.Neuronal_fparam1+salto.Neuronal_fparam1;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="Neuronal_fparam2"){
        while(inicio.Neuronal_fparam2<=fin.Neuronal_fparam2){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,NEURONAL,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Neuronal_fparam2=inicio.Neuronal_fparam2+salto.Neuronal_fparam2;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="Neuronal_rp_dw0"){
        while(inicio.Neuronal_rp_dw0<=fin.Neuronal_rp_dw0){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,NEURONAL,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Neuronal_rp_dw0=inicio.Neuronal_rp_dw0+salto.Neuronal_rp_dw0;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="Neuronal_rp_dw_max"){
        while(inicio.Neuronal_rp_dw_max<=fin.Neuronal_rp_dw_max){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,NEURONAL,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Neuronal_rp_dw_max=inicio.Neuronal_rp_dw_max+salto.Neuronal_rp_dw_max;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="Neuronal_rp_dw_min"){
        while(inicio.Neuronal_rp_dw_min<=fin.Neuronal_rp_dw_min){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,NEURONAL,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Neuronal_rp_dw_min=inicio.Neuronal_rp_dw_min+salto.Neuronal_rp_dw_min;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="Neuronal_rp_dw_minus"){
        while(inicio.Neuronal_rp_dw_minus<=fin.Neuronal_rp_dw_minus){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,NEURONAL,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Neuronal_rp_dw_minus=inicio.Neuronal_rp_dw_minus+salto.Neuronal_rp_dw_minus;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="Neuronal_rp_dw_plus"){
        while(inicio.Neuronal_rp_dw_plus<=fin.Neuronal_rp_dw_plus){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,NEURONAL,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Neuronal_rp_dw_plus=inicio.Neuronal_rp_dw_plus+salto.Neuronal_rp_dw_plus;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="SVM_degree"){
        while(inicio.SVM_degree<=fin.SVM_degree){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,C_SVM,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.SVM_degree=inicio.SVM_degree+salto.SVM_degree;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="SVM_gamma"){
        while(inicio.SVM_gamma<=fin.SVM_gamma){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,C_SVM,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.SVM_gamma=inicio.SVM_gamma+salto.SVM_gamma;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="SVM_coef0"){
        while(inicio.SVM_coef0<=fin.SVM_coef0){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,C_SVM,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.SVM_coef0=inicio.SVM_coef0+salto.SVM_coef0;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="SVM_C"){
        while(inicio.SVM_C<=fin.SVM_C){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,C_SVM,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.SVM_C=inicio.SVM_C+salto.SVM_C;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="SVM_nu"){
        while(inicio.SVM_nu<=fin.SVM_nu){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,C_SVM,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.SVM_nu=inicio.SVM_nu+salto.SVM_nu;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="SVM_p"){
        while(inicio.SVM_p<=fin.SVM_p){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,C_SVM,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.SVM_p=inicio.SVM_p+salto.SVM_p;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="RTrees_cv_folds"){
        while(inicio.RTrees_cv_folds<=fin.RTrees_cv_folds){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,RTREES,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.RTrees_cv_folds=inicio.RTrees_cv_folds+salto.RTrees_cv_folds;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="RTrees_max_categories"){
        while(inicio.RTrees_max_categories<=fin.RTrees_max_categories){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,RTREES,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.RTrees_max_categories=inicio.RTrees_max_categories+salto.RTrees_max_categories;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="RTrees_max_depth"){
        while(inicio.RTrees_max_depth<=fin.RTrees_max_depth){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,RTREES,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.RTrees_max_depth=inicio.RTrees_max_depth+salto.RTrees_max_depth;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="RTrees_min_sample_count"){
        while(inicio.RTrees_min_sample_count<=fin.RTrees_min_sample_count){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,RTREES,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.RTrees_min_sample_count=inicio.RTrees_min_sample_count+salto.RTrees_min_sample_count;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="RTrees_native_vars"){
        while(inicio.RTrees_native_vars<=fin.RTrees_native_vars){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,RTREES,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.RTrees_native_vars=inicio.RTrees_native_vars+salto.RTrees_native_vars;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="RTrees_regression_accuracy"){
        while(inicio.RTrees_regression_accuracy<=fin.RTrees_regression_accuracy){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,RTREES,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.RTrees_regression_accuracy=inicio.RTrees_regression_accuracy+salto.RTrees_regression_accuracy;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="DTrees_cv_folds"){
        while(inicio.DTrees_cv_folds<=fin.DTrees_cv_folds){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,DTREES,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.DTrees_cv_folds=inicio.DTrees_cv_folds+salto.DTrees_cv_folds;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="DTrees_max_categories"){
        while(inicio.DTrees_max_categories<=fin.DTrees_max_categories){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,DTREES,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.DTrees_max_categories=inicio.DTrees_max_categories+salto.DTrees_max_categories;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="DTrees_max_depth"){
        while(inicio.DTrees_max_depth<=fin.DTrees_max_depth){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,DTREES,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.DTrees_max_depth=inicio.DTrees_max_depth+salto.DTrees_max_depth;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="DTrees_min_sample_count"){
        while(inicio.DTrees_min_sample_count<=fin.DTrees_min_sample_count){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,DTREES,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.DTrees_min_sample_count=inicio.DTrees_min_sample_count+salto.DTrees_min_sample_count;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="DTrees_regression_accuracy"){
        while(inicio.DTrees_regression_accuracy<=fin.DTrees_regression_accuracy){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,DTREES,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.DTrees_regression_accuracy=inicio.DTrees_regression_accuracy+salto.DTrees_regression_accuracy;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="Boosting_max_depth"){
        while(inicio.Boosting_max_depth<=fin.Boosting_max_depth){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,BOOSTING,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Boosting_max_depth=inicio.Boosting_max_depth+salto.Boosting_max_depth;
            Ratios.push_back(Rat);
        }
    }
    else if(parametro=="Boosting_weak_count"){
        while(inicio.Boosting_weak_count<=fin.Boosting_weak_count){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,BOOSTING,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.Boosting_weak_count=inicio.Boosting_weak_count+salto.Boosting_weak_count;
            Ratios.push_back(Rat);
        }
    }
//    else if(parametro=="GBT_weak_count"){
//        while(inicio.GBT_weak_count<=fin.GBT_weak_count){
//            float Error=0;
//            Mat Confusion;
//            vector<Analisis::Ratios_data> Rat;
//            e=Validation(Datos,Labels,porcentaje_validacion,GBT,inicio,Error,Confusion,Rat);
//            if(e==1){
//                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
//                return 1;
//            }
//            inicio.GBT_weak_count=inicio.GBT_weak_count+salto.GBT_weak_count;
//            Ratios.push_back(Rat);
//        }
//    }
//    else if(parametro=="GBT_shrinkage"){
//        while(inicio.GBT_shrinkage<=fin.GBT_shrinkage){
//            float Error=0;
//            Mat Confusion;
//            vector<Analisis::Ratios_data> Rat;
//            e=Validation(Datos,Labels,porcentaje_validacion,GBT,inicio,Error,Confusion,Rat);
//            if(e==1){
//                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
//                return 1;
//            }
//            inicio.GBT_shrinkage=inicio.GBT_shrinkage+salto.GBT_shrinkage;
//            Ratios.push_back(Rat);
//        }
//    }
//    else if(parametro=="GBT_max_depth"){
//        while(inicio.GBT_max_depth<=fin.GBT_max_depth){
//            float Error=0;
//            Mat Confusion;
//            vector<Analisis::Ratios_data> Rat;
//            e=Validation(Datos,Labels,porcentaje_validacion,GBT,inicio,Error,Confusion,Rat);
//            if(e==1){
//                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
//                return 1;
//            }
//            inicio.GBT_max_depth=inicio.GBT_max_depth+salto.GBT_max_depth;
//            Ratios.push_back(Rat);
//        }
//    }
    else if(parametro=="EM_nclusters"){
        while(inicio.EM_nclusters<=fin.EM_nclusters){
            float Error=0;
            Mat Confusion;
            vector<Analisis::Ratios_data> Rat;
            e=Validation(Datos,Labels,porcentaje_validacion,EXP_MAX,inicio,Error,Confusion,Rat);
            if(e==1){
                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
                return 1;
            }
            inicio.EM_nclusters=inicio.EM_nclusters+salto.EM_nclusters;
            Ratios.push_back(Rat);
        }
    }
//    else if(parametro=="ERTrees_cv_folds"){
//        while(inicio.ERTrees_cv_folds<=fin.ERTrees_cv_folds){
//            float Error=0;
//            Mat Confusion;
//            vector<Analisis::Ratios_data> Rat;
//            e=Validation(Datos,Labels,porcentaje_validacion,ERTREES,inicio,Error,Confusion,Rat);
//            if(e==1){
//                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
//                return 1;
//            }
//            inicio.ERTrees_cv_folds=inicio.ERTrees_cv_folds+salto.ERTrees_cv_folds;
//            Ratios.push_back(Rat);
//        }
//    }
//    else if(parametro=="ERTrees_max_categories"){
//        while(inicio.ERTrees_max_categories<=fin.ERTrees_max_categories){
//            float Error=0;
//            Mat Confusion;
//            vector<Analisis::Ratios_data> Rat;
//            e=Validation(Datos,Labels,porcentaje_validacion,ERTREES,inicio,Error,Confusion,Rat);
//            if(e==1){
//                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
//                return 1;
//            }
//            inicio.ERTrees_max_categories=inicio.ERTrees_max_categories+salto.ERTrees_max_categories;
//            Ratios.push_back(Rat);
//        }
//    }
//    else if(parametro=="ERTrees_max_depth"){
//        while(inicio.ERTrees_max_depth<=fin.ERTrees_max_depth){
//            float Error=0;
//            Mat Confusion;
//            vector<Analisis::Ratios_data> Rat;
//            e=Validation(Datos,Labels,porcentaje_validacion,ERTREES,inicio,Error,Confusion,Rat);
//            if(e==1){
//                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
//                return 1;
//            }
//            inicio.ERTrees_max_depth=inicio.ERTrees_max_depth+salto.ERTrees_max_depth;
//            Ratios.push_back(Rat);
//        }
//    }
//    else if(parametro=="ERTrees_min_sample_count"){
//        while(inicio.ERTrees_min_sample_count<=fin.ERTrees_min_sample_count){
//            float Error=0;
//            Mat Confusion;
//            vector<Analisis::Ratios_data> Rat;
//            e=Validation(Datos,Labels,porcentaje_validacion,ERTREES,inicio,Error,Confusion,Rat);
//            if(e==1){
//                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
//                return 1;
//            }
//            inicio.ERTrees_min_sample_count=inicio.ERTrees_min_sample_count+salto.ERTrees_min_sample_count;
//            Ratios.push_back(Rat);
//        }
//    }
//    else if(parametro=="ERTrees_native_vars"){
//        while(inicio.ERTrees_native_vars<=fin.ERTrees_native_vars){
//            float Error=0;
//            Mat Confusion;
//            vector<Analisis::Ratios_data> Rat;
//            e=Validation(Datos,Labels,porcentaje_validacion,ERTREES,inicio,Error,Confusion,Rat);
//            if(e==1){
//                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
//                return 1;
//            }
//            inicio.ERTrees_native_vars=inicio.ERTrees_native_vars+salto.ERTrees_native_vars;
//            Ratios.push_back(Rat);
//        }
//    }
//    else if(parametro=="ERTrees_regression_accuracy"){
//        while(inicio.ERTrees_regression_accuracy<=fin.ERTrees_regression_accuracy){
//            float Error=0;
//            Mat Confusion;
//            vector<Analisis::Ratios_data> Rat;
//            e=Validation(Datos,Labels,porcentaje_validacion,ERTREES,inicio,Error,Confusion,Rat);
//            if(e==1){
//                cout<<"ERROR en Ratios_parametro: Error en Validation"<<endl;
//                return 1;
//            }
//            inicio.ERTrees_regression_accuracy=inicio.ERTrees_regression_accuracy+salto.ERTrees_regression_accuracy;
//            Ratios.push_back(Rat);
//        }
//    }
    else{
        cout<<"ERROR en Ratios_parametro: El parámetro no existe"<<endl;
        return 1;
    }
    return 0;
}

