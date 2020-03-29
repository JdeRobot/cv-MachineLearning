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

#include "generacion.h"

MLT::Generacion::Generacion(){
    this->running=false;
}

int MLT::Generacion::Cargar_Imagenes(string input_directory, std::vector<cv::Mat> &Images){
    this->running=true;
    input_directory=input_directory+"/";
    string strPrefix;
    DIR    *dir_p = opendir (input_directory.c_str());
    struct dirent *dir_entry_p;
    std::vector<cv::Mat> img;
    while((dir_entry_p = readdir(dir_p)) != NULL){
        if(strcmp(dir_entry_p->d_name, ""))
            strPrefix=input_directory+dir_entry_p->d_name;
        if(strcmp(dir_entry_p->d_name, ".")!=0 && strcmp(dir_entry_p->d_name, "..")!=0){
            Mat imagen = imread(strPrefix.c_str());
            Mat Imagen;
            imagen.convertTo(Imagen,CV_32F);
            if(Imagen.empty()){
#ifdef WARNINGS
                cout<<"WARINING en Cargar_Imagenes: Intento de cargar un archivo que no es imagen o imagen esta vacia"<<endl;
#endif
            }
            else
                img.push_back(Imagen);
        }
#ifdef GUI
            this->progreso++;
#endif
    }
    Images=img;
    if(Images.size()==0){
        cout<<"ERROR en Cargar_Imagenes: Resultado de tamaño 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Guardar_Datos(string nombre, vector<Mat> Imagenes, vector<float> Labels, Info_Datos info){
    this->running=true;
    if(Imagenes.size()==0){
        cout<<"ERROR en Guardar_Datos: Imagenes esta vacio"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    if(Labels.size()==0){
        cout<<"ERROR en Guardar_Datos: Labels esta vacio"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    if(Imagenes.size()!= Labels.size()){
        cout<<"ERROR en Guardar_Datos: El tamaño de Imagenes y Labels no coincide"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    string output_directory="../Data/Imagenes/"+nombre+"/";
    string archivo_imagenes=output_directory+"Images.xml";
    string archivo_imagenes_recortes=output_directory+"Recortes.xml";
    string archivo_recortes=output_directory+"Recortes.txt";
    string archivo_info=output_directory+"Info.xml";
    ofstream f;
    string dir_img="../Data/Imagenes";
    DIR    *dir_p_o = opendir (dir_img.c_str());
    if(dir_p_o == NULL) {
        string command = "mkdir "+dir_img;
        int er=system(command.c_str());
        if(er==0)
            dir_p_o = opendir (dir_img.c_str());
    }
    DIR    *dir_p_i = opendir (output_directory.c_str());
    if(dir_p_i == NULL) {
        string command = "mkdir "+output_directory;
        int er=system(command.c_str());
        if(er==0)
            dir_p_i = opendir (output_directory.c_str());
    }
    f.open(archivo_recortes.c_str(),ofstream::out | ofstream::trunc);
    cv::FileStorage Archivo_img(archivo_imagenes,FileStorage::WRITE);
    cv::FileStorage Archivo_recortes(archivo_imagenes_recortes,FileStorage::WRITE);
    cv::FileStorage Archivo_i(archivo_info,FileStorage::WRITE);
    for(uint i=0; i<Imagenes.size(); i++){
        stringstream conta;
        ostringstream datos;
        conta<<"Imagen"<<(i+1);
        datos<<conta.str();
        datos<<" ";
        datos<<1;
        datos<<" ";
        datos<<0;
        datos<<" ";
        datos<<0;
        datos<<" ";
        datos<<Imagenes[i].cols-1;
        datos<<" ";
        datos<<Imagenes[i].rows-1;
        datos<<" ";
        datos<<0;
        datos<<" ";
        datos<<Labels[i];
        datos<<" ";
        f<<datos.str();
        f<<"\n";
        Mat imag;
        Imagenes[i].convertTo(imag,CV_32F);
        Archivo_img<<conta.str()<<imag;
        Archivo_recortes<<conta.str()<<imag;
#ifdef GUI
        this->progreso++;
#endif
    }
    Archivo_i<<"Tipo_Datos"<<info.Tipo_Datos;
    Archivo_i<<"Num_Datos"<<info.Num_Datos;
    Archivo_i<<"DS"<<info.DS;
    Archivo_i<<"D_PRIME"<<info.D_PRIME;
    Archivo_i<<"LDA"<<info.LDA;
    Archivo_i<<"PCA"<<info.PCA;
    Archivo_i<<"si_dist"<<info.si_dist;
    Archivo_i<<"si_d_prime"<<info.si_d_prime;
    Archivo_i<<"si_lda"<<info.si_lda;
    Archivo_i<<"si_pca"<<info.si_pca;
    Archivo_i<<"Tam_Orig_X"<<info.Tam_Orig_X;
    Archivo_i<<"Tam_Orig_Y"<<info.Tam_Orig_Y;
    Archivo_i<<"Tam_X"<<info.Tam_X;
    Archivo_i<<"Tam_Y"<<info.Tam_Y;
    f.close();
    Archivo_i.release();
    Archivo_img.release();
    Archivo_recortes.release();
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Cargar_Fichero(string Archivo, vector<Mat> &Imagenes, vector<float> &Labels, Info_Datos &info){
    this->running=true;
    Imagenes.clear();
    Labels.clear();
    std::ifstream input_rec(Archivo.c_str());
    if(!input_rec.is_open()){
        cout<<"ERROR en Cargar_Fichero: path ilegible"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    int pos=0;
    for(uint i=0; i<Archivo.size(); i++){
        if(Archivo[i]=='/')
            pos=i;
    }
    string archivo_i;
    for(int i=0; i<pos+1; i++){
        archivo_i=archivo_i+Archivo[i];
    }
    archivo_i=archivo_i+"Info.xml";
    cv::FileStorage archivo_r(archivo_i,FileStorage::READ);
    if(!archivo_r.isOpened()){
        cout<<"ERROR en Cargar_Fichero: No se ha podido cargar el archivo Info.xml"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    int num=0;
    archivo_r["Num_Datos"]>>num;
    this->total_progreso=num;
    string archivo_recortes;
    for(int i=0; i<pos+1; i++){
        archivo_recortes=archivo_recortes+Archivo[i];
    }
    archivo_recortes=archivo_recortes+"Recortes.xml";
    cv::FileStorage Archivo_recortes(archivo_recortes,FileStorage::READ);
    std::string line;
    line.clear();
    string path_imagen;
    vector<Rect> recortes;
    int numero_imagenes=0;
    while(std::getline(input_rec, line)){
        vector<int> espacios;
        int cont=0;
        Rect aux;
        vector<int> rotacion;
        for(uint i=0; i<line.size(); i++){
            if(line[i]==' '){
                espacios.push_back(i);
                if(espacios.size()==1){
                    for(uint j=0; j<i; j++)
                        path_imagen=path_imagen+line[j];
                    if(path_imagen==""){
                        cout<<"ERROR en Cargar_Fichero: nombre de imagen ilegible"<<endl;
                        this->running=false;
                        this->error=1;
                        return this->error;
                    }
                }
                else if(espacios.size()==2){
                    string num_recortes;
                    for(uint j=espacios[0]+1; j<i; j++)
                        num_recortes=num_recortes+line[j];
                    aux.x=0;
                    aux.y=0;
                    aux.height=0;
                    aux.width=0;
                }
                else if(espacios.size()>2 && cont==0){
                    string numero;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        numero=numero+line[j];
                    aux.x=atoi(numero.c_str());
                    cont++;
                }
                else if(espacios.size()>2 && cont==1){
                    string numero;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        numero=numero+line[j];
                    aux.y=atoi(numero.c_str());
                    cont++;
                }
                else if(espacios.size()>2 && cont==2){
                    string numero;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        numero=numero+line[j];
                    aux.width=atoi(numero.c_str());
                    cont++;
                }
                else if(espacios.size()>2 && cont==3){
                    string numero;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        numero=numero+line[j];
                    aux.height=atoi(numero.c_str());
                    recortes.push_back(aux);
                    cont++;
                }
                else if(espacios.size()>2 && cont==4){
                    string rota;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        rota=rota+line[j];
                    rotacion.push_back(atoi(rota.c_str()));
                    cont++;
                }
                else if(espacios.size()>2 && cont==5){
                    string numero;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        numero=numero+line[j];
                    if(numero=="0"){
                        cout<<"ERROR en Cargar_Fichero: etiqueta erronea (igual a 0)"<<endl;
                        this->running=false;
                        this->error=1;
                        return this->error;
                    }
                    Labels.push_back(atoi(numero.c_str()));
                    cont=0;
                    numero_imagenes++;
                    Mat imagen;
                    stringstream nombr;
                    nombr<<"Imagen"<<numero_imagenes;
                    Archivo_recortes[nombr.str()]>>imagen;
                    if(imagen.empty()){
                        cout<<"ERROR en Cargar_Fichero: No se pudo cargar Imagen"<<numero_imagenes<<endl;
                        this->running=false;
                        this->error=1;
                        return this->error;
                    }

                    Mat Imagen;
                    imagen.convertTo(Imagen,CV_32F);
                    Imagenes.push_back(Imagen);
            #ifdef GUI
                    this->progreso++;
            #endif
                }
            }
        }
        path_imagen.clear();
        line.clear();
        recortes.clear();
    }

    archivo_r["Tipo_Datos"]>>info.Tipo_Datos;
    archivo_r["Num_Datos"]>>info.Num_Datos;
    archivo_r["DS"]>>info.DS;
    archivo_r["D_PRIME"]>>info.D_PRIME;
    archivo_r["LDA"]>>info.LDA;
    archivo_r["PCA"]>>info.PCA;
    archivo_r["si_dist"]>>info.si_dist;
    archivo_r["si_d_prime"]>>info.si_d_prime;
    archivo_r["si_lda"]>>info.si_lda;
    archivo_r["si_pca"]>>info.si_pca;
    archivo_r["Tam_Orig_X"]>>info.Tam_Orig_X;
    archivo_r["Tam_Orig_Y"]>>info.Tam_Orig_Y;
    archivo_r["Tam_X"]>>info.Tam_X;
    archivo_r["Tam_Y"]>>info.Tam_Y;
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Juntar_Recortes(string nombre,string Path){
    this->running=true;

    Path=Path+"/";
    string output_directory=Path+nombre+"/";
    string archivo_imagenes=output_directory+"Images.xml";
    string archivo_imagenes_recortes=output_directory+"Recortes.xml";
    string archivo_recortes=output_directory+"Recortes.txt";
    string archivo_info=output_directory+"Info.xml";
    ofstream f;
    DIR    *dir_p2 = opendir (output_directory.c_str());
    if(dir_p2 == NULL) {
        string command = "mkdir "+output_directory;
        int er=system(command.c_str());
        if(er==1){
            cout<<"ERROR en Juntar_Recortes: Error al crear carpeta"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
    }
    DIR    *dir_p = opendir (Path.c_str());
    if(dir_p==NULL){
        this->running=false;
        this->error=1;
        return this->error;
    }
    struct dirent *dir_entry_p;
    f.open(archivo_recortes.c_str(),ofstream::out | ofstream::trunc);
    cv::FileStorage Archivo_img(archivo_imagenes,FileStorage::WRITE);
    cv::FileStorage Archivo_recortes(archivo_imagenes_recortes,FileStorage::WRITE);
    cv::FileStorage Archivo_i(archivo_info,FileStorage::WRITE);
    int cuenta_carpetas=0;
    Info_Datos info;
    Archivo_i["Tipo_Datos"]>>info.Tipo_Datos;
    Archivo_i["Num_Datos"]>>info.Num_Datos;
    Archivo_i["DS"]>>info.DS;
    Archivo_i["D_PRIME"]>>info.D_PRIME;
    Archivo_i["LDA"]>>info.LDA;
    Archivo_i["PCA"]>>info.PCA;
    Archivo_i["si_dist"]>>info.si_dist;
    Archivo_i["si_d_prime"]>>info.si_d_prime;
    Archivo_i["si_lda"]>>info.si_lda;
    Archivo_i["si_pca"]>>info.si_pca;
    Archivo_i["Tam_Orig_X"]>>info.Tam_Orig_X;
    Archivo_i["Tam_Orig_Y"]>>info.Tam_Orig_Y;
    Archivo_i["Tam_X"]>>info.Tam_X;
    Archivo_i["Tam_Y"]>>info.Tam_Y;
    int total_recortes=0;
    int cuenta_imagenes=0;
    int cuenta_recortes=0;

    while((dir_entry_p = readdir(dir_p)) != NULL){
        if(strcmp(dir_entry_p->d_name, ".")!=0 && strcmp(dir_entry_p->d_name, "..")!=0 && strcmp(dir_entry_p->d_name, nombre.c_str())!=0){
            cuenta_carpetas++;
            string input_directory=Path+dir_entry_p->d_name+"/";
            string archivo_imagenes_in=input_directory+"Images.xml";
            string archivo_imagenes_recortes_in=input_directory+"Recortes.xml";
            string archivo_recortes_in=input_directory+"Recortes.txt";
            string archivo_info_in=input_directory+"Info.xml";
            std::ifstream f_in(archivo_recortes_in.c_str());
            cv::FileStorage Archivo_img_in(archivo_imagenes_in,FileStorage::READ);
            cv::FileStorage Archivo_recortes_in(archivo_imagenes_recortes_in,FileStorage::READ);
            cv::FileStorage Archivo_i_in(archivo_info_in,FileStorage::READ);
            Info_Datos info_in;
            Archivo_i_in["Tipo_Datos"]>>info_in.Tipo_Datos;
            Archivo_i_in["Num_Datos"]>>info_in.Num_Datos;
            Archivo_i_in["DS"]>>info_in.DS;
            Archivo_i_in["D_PRIME"]>>info_in.D_PRIME;
            Archivo_i_in["LDA"]>>info_in.LDA;
            Archivo_i_in["PCA"]>>info_in.PCA;
            Archivo_i_in["si_dist"]>>info_in.si_dist;
            Archivo_i_in["si_d_prime"]>>info_in.si_d_prime;
            Archivo_i_in["si_lda"]>>info_in.si_lda;
            Archivo_i_in["si_pca"]>>info_in.si_pca;
            Archivo_i_in["Tam_Orig_X"]>>info_in.Tam_Orig_X;
            Archivo_i_in["Tam_Orig_Y"]>>info_in.Tam_Orig_Y;
            Archivo_i_in["Tam_X"]>>info_in.Tam_X;
            Archivo_i_in["Tam_Y"]>>info_in.Tam_Y;
            total_recortes=total_recortes+info_in.Num_Datos;
            if(cuenta_carpetas==1){
                info.Tipo_Datos=info_in.Tipo_Datos;
                info.Num_Datos=info_in.Num_Datos;
                info.DS=info_in.DS;
                info.D_PRIME=info_in.D_PRIME;
                info.LDA=info_in.LDA;
                info.PCA=info_in.PCA;
                info.si_dist=info_in.si_dist;
                info.si_d_prime=info_in.si_d_prime;
                info.si_lda=info_in.si_lda;
                info.si_pca=info_in.si_pca;
                info.Tam_Orig_X=info_in.Tam_Orig_X;
                info.Tam_Orig_Y=info_in.Tam_Orig_Y;
                info.Tam_X=info_in.Tam_X;
                info.Tam_Y=info_in.Tam_Y;
            }
            else{
                if((info.Tipo_Datos!=info_in.Tipo_Datos) || (info.si_dist!=info.si_dist) ||
                        (info.si_d_prime!=info_in.si_d_prime) || (info.si_lda!=info_in.si_lda) ||
                        (info.si_pca!=info_in.si_pca) || (info.Tam_Orig_X!=info_in.Tam_Orig_X) ||
                        (info.Tam_Orig_Y!=info_in.Tam_Orig_Y) || (info.Tam_X!=info_in.Tam_X) ||
                        (info.Tam_Y!=info_in.Tam_Y)){

                    string command = "rm -r -f "+output_directory;
                    int er=system(command.c_str());
                    cout<<"ERROR en Juntar_Recortes: Los datos que se intena juntar no son del mismo tipo"<<endl;
                    if (er!=0){
                        cout<<"Puede que hayan quedado archivos corruptos en el path de destino"<<endl;
                    }
                    this->running=false;
                    this->error=1;
                    return this->error;
                }
            }
            std::string line, line_params, line_nombre;
            Mat imagen;
            while( std::getline(f_in, line)){
                cuenta_imagenes++;
                int pos=0;
                for(uint n=0; n<line.size(); n++){
                    if(pos==0 && line[n]==' ')
                        pos=n;
                }
                for(int n=0; n<pos; n++)
                    line_nombre=line_nombre+line[n];
                for(uint n=pos; n<line.size(); n++)
                    line_params=line_params+line[n];
                f<<"Imagen";
                f<<cuenta_imagenes;
                f<<line_params.c_str();
                f<<"\n";
                Archivo_img_in[line_nombre.c_str()]>>imagen;
                stringstream nom_s;
                nom_s<<"Imagen";
                nom_s<<cuenta_imagenes;
                Archivo_img<<nom_s.str()<<imagen;
                line.clear();
                line_params.clear();
                line_nombre.clear();
            }
            int cuenta_ficheros=1;
            Archivo_recortes_in["Imagen1"]>>imagen;
            while(!imagen.empty()){
                cuenta_recortes++;
                stringstream nom_s;
                nom_s<<"Imagen";
                nom_s<<cuenta_recortes;
                Archivo_recortes<<nom_s.str()<<imagen;
                cuenta_ficheros++;
                stringstream nom;
                nom<<"Imagen";
                nom<<cuenta_ficheros;
                Archivo_recortes_in[nom.str()]>>imagen;
            }
            f_in.close();
            Archivo_i_in.release();
            Archivo_img_in.release();
            Archivo_recortes_in.release();
#ifdef GUI
    this->progreso++;
#endif
        }
    }
    info.Num_Datos=total_recortes;
    Archivo_i<<"Tipo_Datos"<<info.Tipo_Datos;
    Archivo_i<<"Num_Datos"<<info.Num_Datos;
    Archivo_i<<"DS"<<info.DS;
    Archivo_i<<"D_PRIME"<<info.D_PRIME;
    Archivo_i<<"LDA"<<info.LDA;
    Archivo_i<<"PCA"<<info.PCA;
    Archivo_i<<"si_dist"<<info.si_dist;
    Archivo_i<<"si_d_prime"<<info.si_d_prime;
    Archivo_i<<"si_lda"<<info.si_lda;
    Archivo_i<<"si_pca"<<info.si_pca;
    Archivo_i<<"Tam_Orig_X"<<info.Tam_Orig_X;
    Archivo_i<<"Tam_Orig_Y"<<info.Tam_Orig_Y;
    Archivo_i<<"Tam_X"<<info.Tam_X;
    Archivo_i<<"Tam_Y"<<info.Tam_Y;
    f.close();
    Archivo_i.release();
    Archivo_img.release();
    Archivo_recortes.release();
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Datos_Imagenes(string nombre, string input_directory, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save){
    this->running=true;
    if(tam_recorte.width<1 || tam_recorte.height<1){
        cout<<"ERROR en Datos_Imagenes: El tamaño del recorte debe ser mayor a 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    input_directory=input_directory+"/";
    string strPrefix;
    DIR    *dir_p = opendir (input_directory.c_str());
    struct dirent *dir_entry_p;
    std::vector<cv::Mat> img;
    std::vector<float> etiquetas;
    int cont=0;
    string output_directory="../Data/Imagenes/"+nombre+"/";
    string archivo_imagenes=output_directory+"Images.xml";
    string archivo_imagenes_recortes=output_directory+"Recortes.xml";
    string archivo_recortes=output_directory+"Recortes.txt";
    string archivo_info=output_directory+"Info.xml";
    string g=output_directory+"Config.xml";
    cv::FileStorage archivo_r(g,FileStorage::READ);
    if(archivo_r.isOpened()){
        cv::FileStorage Archivo_i(archivo_info,FileStorage::READ);
        if(!Archivo_i.isOpened()){
            cout<<"ERROR en Datos_Imagenes: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        int num;
        Archivo_i["Num_Datos"]>>num;
        Archivo_i.release();
#ifndef GUI
        total_progreso=num;
        progreso=0;
        base_progreso=0;
        max_progreso=100;
#endif
        Info_Datos inff;
        Cargar_Fichero(archivo_recortes,img,Labels,inff);

        string command = "cp "+archivo_imagenes+" "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "cp "+archivo_imagenes_recortes+" "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Datos_Imagenes: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
    }
    cv::FileStorage Archivo_img(archivo_imagenes,FileStorage::WRITE);
    cv::FileStorage Archivo_recortes(archivo_imagenes_recortes,FileStorage::WRITE);
    cv::FileStorage Archivo_i(archivo_info,FileStorage::WRITE);
    ofstream f;
    if(save){
        string dir_img="../Data/Imagenes";
        DIR    *dir_p_o = opendir (dir_img.c_str());
        if(dir_p_o == NULL) {
            string command = "mkdir "+dir_img;
            int er=system(command.c_str());
            if(er==0)
                dir_p_o = opendir (dir_img.c_str());
        }
        DIR    *dir_p_i = opendir (output_directory.c_str());
        if(dir_p_i == NULL) {
            string command = "mkdir "+output_directory;
            int er=system(command.c_str());
            if(er==0)
                dir_p_i = opendir (output_directory.c_str());
        }
        Archivo_img.open(archivo_imagenes,FileStorage::WRITE);
        Archivo_recortes.open(archivo_imagenes_recortes,FileStorage::WRITE);
        Archivo_i.open(archivo_info,FileStorage::WRITE);
    }
    if(archivo_r.isOpened()){
        cv::FileStorage aux_Images(output_directory+"aux_Images.xml",FileStorage::READ);
        FileNode nodo1 = aux_Images.root();
        for (FileNodeIterator current = nodo1.begin(); current != nodo1.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Images[nom.c_str()]>>v;
                Archivo_img<<nom<<v;
        }
        cv::FileStorage aux_Recortes(output_directory+"aux_Recortes.xml",FileStorage::READ);
        FileNode nodo2 = aux_Recortes.root();
        for (FileNodeIterator current = nodo2.begin(); current != nodo2.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Recortes[nom.c_str()]>>v;
                Archivo_recortes<<nom<<v;
        }
        aux_Images.release();
        aux_Recortes.release();
        string command = "rm "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "rm "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Datos_Imagenes: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::app);
        int num_imagen;
        archivo_r["num_imagen"]>>num_imagen;
        cont=num_imagen-1;
        for(int i=1; i<num_imagen; i++)
            dir_entry_p = readdir(dir_p);

    }
    else if(save==true)
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::trunc);
    while((dir_entry_p = readdir(dir_p)) != NULL){
        if(strcmp(dir_entry_p->d_name, ""))
            strPrefix=input_directory+dir_entry_p->d_name;
        if(strcmp(dir_entry_p->d_name, ".")!=0 && strcmp(dir_entry_p->d_name, "..")!=0){
            Mat imagen = imread(strPrefix.c_str());
            if(imagen.empty()){
#ifdef WARNINGS
                cout<<"WARNING en Datos_Imagenes: Se ha intentado abrir un archivo que no es de tipo imagen"<<endl;
#endif
            }
            else{
                cont++;
                Mat Imagen;
                imagen.convertTo(Imagen,CV_32F);
                Mat modif,modificada;
                cv::resize(Imagen,modif,tam_recorte);
                modif.convertTo(modificada,CV_32F);
                etiquetas.push_back(1);
                img.push_back(modificada);
                if(save==true){
                    ostringstream datos;
                    datos<<"Imagen";
                    datos<<cont;
                    datos<<" ";
                    datos<<1;
                    datos<<" ";
                    datos<<0;
                    datos<<" ";
                    datos<<0;
                    datos<<" ";
                    datos<<imagen.cols-1;
                    datos<<" ";
                    datos<<imagen.rows-1;
                    datos<<" ";
                    datos<<0;
                    datos<<" ";
                    datos<<1;
                    datos<<" ";
                    f<<datos.str();
                    f<<"\n";
                    ostringstream nom;
                    nom<<"Imagen";
                    nom<<cont;
                    Archivo_img<<nom.str()<<modificada;
                    Archivo_recortes<<nom.str()<<modificada;
                }
            }
        }
    }
    Labels=etiquetas;
    imagenes=img;
    destroyAllWindows();
    if(Labels.size()==0 || imagenes.size()==0 || Labels.size()!=imagenes.size()){
        cout<<"ERROR en Datos_Imagenes: El resultado de las etiquetas e imagenes no tienen el mismo tamaño o son 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    info.Tipo_Datos=0;
    info.Num_Datos=imagenes.size();
    info.DS=Mat();
    info.D_PRIME=Mat();
    info.LDA=Mat();
    info.PCA=Mat();
    info.si_dist=false;
    info.si_d_prime=false;
    info.si_lda=false;
    info.si_pca=false;
    info.Tam_Orig_X=imagenes[0].cols;
    info.Tam_Orig_Y=imagenes[0].rows;
    info.Tam_X=imagenes[0].cols;
    info.Tam_Y=imagenes[0].rows;
    if(save){
        Archivo_i<<"Tipo_Datos"<<info.Tipo_Datos;
        Archivo_i<<"Num_Datos"<<info.Num_Datos;
        Archivo_i<<"DS"<<info.DS;
        Archivo_i<<"D_PRIME"<<info.D_PRIME;
        Archivo_i<<"LDA"<<info.LDA;
        Archivo_i<<"PCA"<<info.PCA;
        Archivo_i<<"si_dist"<<info.si_dist;
        Archivo_i<<"si_d_prime"<<info.si_d_prime;
        Archivo_i<<"si_lda"<<info.si_lda;
        Archivo_i<<"si_pca"<<info.si_pca;
        Archivo_i<<"Tam_Orig_X"<<info.Tam_Orig_X;
        Archivo_i<<"Tam_Orig_Y"<<info.Tam_Orig_Y;
        Archivo_i<<"Tam_X"<<info.Tam_X;
        Archivo_i<<"Tam_Y"<<info.Tam_Y;
    }
    f.close();
    Archivo_i.release();
    Archivo_img.release();
    Archivo_recortes.release();
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Etiquetar(string nombre, string input_directory, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save){

    this->running=true;
    if(tam_recorte.width<1 || tam_recorte.height<1){
        cout<<"ERROR en Etiquetar: El tamaño del recorte debe ser mayor a 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    input_directory=input_directory+"/";
    string strPrefix;
    DIR    *dir_p = opendir (input_directory.c_str());
    struct dirent *dir_entry_p;
    std::vector<cv::Mat> img;
    std::vector<float> etiquetas;
    int cont=0;
    string output_directory="../Data/Imagenes/"+nombre+"/";
    string archivo_imagenes=output_directory+"Images.xml";
    string archivo_imagenes_recortes=output_directory+"Recortes.xml";
    string archivo_recortes=output_directory+"Recortes.txt";
    string archivo_info=output_directory+"Info.xml";
    string g=output_directory+"Config.xml";
    cv::FileStorage archivo_r(g,FileStorage::READ);
    if(archivo_r.isOpened()){
        cv::FileStorage Archivo_i(archivo_info,FileStorage::READ);
        if(!Archivo_i.isOpened()){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        int num;
        Archivo_i["Num_Datos"]>>num;
        Archivo_i.release();
#ifndef GUI
        total_progreso=num;
        progreso=0;
        base_progreso=0;
        max_progreso=100;
#endif
        Info_Datos inff;
        Cargar_Fichero(archivo_recortes,img,Labels,inff);

        string command = "cp "+archivo_imagenes+" "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "cp "+archivo_imagenes_recortes+" "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
    }
    cv::FileStorage Archivo_img(archivo_imagenes,FileStorage::WRITE);
    cv::FileStorage Archivo_recortes(archivo_imagenes_recortes,FileStorage::WRITE);
    cv::FileStorage Archivo_i(archivo_info,FileStorage::WRITE);
    ofstream f;
    if(save){
        string dir_img="../Data/Imagenes";
        DIR    *dir_p_o = opendir (dir_img.c_str());
        if(dir_p_o == NULL) {
            string command = "mkdir "+dir_img;
            int er=system(command.c_str());
            if(er==0)
                dir_p_o = opendir (dir_img.c_str());
        }
        DIR    *dir_p_i = opendir (output_directory.c_str());
        if(dir_p_i == NULL) {
            string command = "mkdir "+output_directory;
            int er=system(command.c_str());
            if(er==0)
                dir_p_i = opendir (output_directory.c_str());
        }
        Archivo_img.open(archivo_imagenes,FileStorage::WRITE);
        Archivo_recortes.open(archivo_imagenes_recortes,FileStorage::WRITE);
        Archivo_i.open(archivo_info,FileStorage::WRITE);
    }
    if(archivo_r.isOpened()){
        cv::FileStorage aux_Images(output_directory+"aux_Images.xml",FileStorage::READ);
        FileNode nodo1 = aux_Images.root();
        for (FileNodeIterator current = nodo1.begin(); current != nodo1.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Images[nom.c_str()]>>v;
                Archivo_img<<nom<<v;
        }
        cv::FileStorage aux_Recortes(output_directory+"aux_Recortes.xml",FileStorage::READ);
        FileNode nodo2 = aux_Recortes.root();
        for (FileNodeIterator current = nodo2.begin(); current != nodo2.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Recortes[nom.c_str()]>>v;
                Archivo_recortes<<nom<<v;
        }
        aux_Images.release();
        aux_Recortes.release();
        string command = "rm "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "rm "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::app);
        int num_imagen;
        archivo_r["num_imagen"]>>num_imagen;
        cont=num_imagen-1;
        for(int i=1; i<num_imagen; i++)
            dir_entry_p = readdir(dir_p);

    }
    else if(save==true)
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::trunc);
    while((dir_entry_p = readdir(dir_p)) != NULL){
        if(strcmp(dir_entry_p->d_name, ""))
            strPrefix=input_directory+dir_entry_p->d_name;
        if(strcmp(dir_entry_p->d_name, ".")!=0 && strcmp(dir_entry_p->d_name, "..")!=0){
            Mat imagen = imread(strPrefix.c_str());
            if(imagen.empty()){
#ifdef WARNINGS
                cout<<"WARNING en Etiquetar: Se ha intentado abrir un archivo que no es de tipo imagen"<<endl;
#endif
            }
            else{
                cont++;
                cout<<"Imagen= "<<dir_entry_p->d_name;
                Mat Imagen;
                imagen.convertTo(Imagen,CV_32F);
                Muestra:
                cv::imshow("Etiquetar",imagen);
                char z=waitKey(0);
                if(z=='i'){
                    Mat most=Mat::zeros(200,200,CV_8UC3);
                    most=most+Scalar(255,255,255);
                    string texto="ESC=Exit";
                    putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                    texto="Num=Labeling";
                    putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                    texto="Space=Next";
                    putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                    imshow("Info",most);
                    waitKey(1500);
                    destroyWindow("Info");
                    goto Muestra;
                }
                int a=z;
                if(a!=-1){
                    if(a==27){
                        cv::FileStorage archivo_w(g,FileStorage::WRITE);
                        archivo_w<<"num_imagen"<<cont;
                        archivo_w.release();
                        break;
                    }
                    if(a!=48 && a!=49 && a!=50 && a!=51 && a!=52 && a!=53
                             && a!=54 && a!=55 && a!=56 && a!=57 &&
                            a!=-71 && a!=-72 && a!=-73 && a!=-74 && a!=-75
                             && a!=-76 && a!=-77 && a!=-78 && a!=-79 && a!=-80 && a!=32){
                        break;
                    }
                    if(a!=32){
                        if(a<0){
                            a=a+128-'0';
                        }
                        else{
                            a=a-'0';
                        }
                        if(a==0)
                            a=-1.0;
                        Mat modif,modificada;
                        cv::resize(Imagen,modif,tam_recorte);
                        modif.convertTo(modificada,CV_32F);
                        etiquetas.push_back(a);
                        img.push_back(modificada);
                        if(save==true){
                            ostringstream datos;
                            datos<<"Imagen";
                            datos<<cont;
                            datos<<" ";
                            datos<<1;
                            datos<<" ";
                            datos<<0;
                            datos<<" ";
                            datos<<0;
                            datos<<" ";
                            datos<<imagen.cols-1;
                            datos<<" ";
                            datos<<imagen.rows-1;
                            datos<<" ";
                            datos<<0;
                            datos<<" ";
                            datos<<a;
                            datos<<" ";
                            f<<datos.str();
                            f<<"\n";
                            ostringstream nom;
                            nom<<"Imagen";
                            nom<<cont;
                            Archivo_img<<nom.str()<<modificada;
                            Archivo_recortes<<nom.str()<<modificada;
                        }
                        cout<<"       Etiqueta= "<<a<<endl;
                    }
                    else
                        cout<<endl;
                }
            }
        }
    }
    Labels=etiquetas;
    imagenes=img;
    destroyAllWindows();
    if(Labels.size()==0 || imagenes.size()==0 || Labels.size()!=imagenes.size()){
        cout<<"ERROR en Etiquetar: El resultado de las etiquetas e imagenes no tienen el mismo tamaño o son 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    info.Tipo_Datos=0;
    info.Num_Datos=imagenes.size();
    info.DS=Mat();
    info.D_PRIME=Mat();
    info.LDA=Mat();
    info.PCA=Mat();
    info.si_dist=false;
    info.si_d_prime=false;
    info.si_lda=false;
    info.si_pca=false;
    info.Tam_Orig_X=imagenes[0].cols;
    info.Tam_Orig_Y=imagenes[0].rows;
    info.Tam_X=imagenes[0].cols;
    info.Tam_Y=imagenes[0].rows;
    if(save){
        Archivo_i<<"Tipo_Datos"<<info.Tipo_Datos;
        Archivo_i<<"Num_Datos"<<info.Num_Datos;
        Archivo_i<<"DS"<<info.DS;
        Archivo_i<<"D_PRIME"<<info.D_PRIME;
        Archivo_i<<"LDA"<<info.LDA;
        Archivo_i<<"PCA"<<info.PCA;
        Archivo_i<<"si_dist"<<info.si_dist;
        Archivo_i<<"si_d_prime"<<info.si_d_prime;
        Archivo_i<<"si_lda"<<info.si_lda;
        Archivo_i<<"si_pca"<<info.si_pca;
        Archivo_i<<"Tam_Orig_X"<<info.Tam_Orig_X;
        Archivo_i<<"Tam_Orig_Y"<<info.Tam_Orig_Y;
        Archivo_i<<"Tam_X"<<info.Tam_X;
        Archivo_i<<"Tam_Y"<<info.Tam_Y;
    }
    f.close();
    Archivo_i.release();
    Archivo_img.release();
    Archivo_recortes.release();
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Recortar_Etiquetar_imagenes(string nombre, string input_directory, bool cuadrado, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save){
    this->running=true;
    if(tam_recorte.width<1 || tam_recorte.height<1){
        cout<<"ERROR en Recortar_Etiquetar: El tamaño del recorte debe ser mayor a 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    this->Cuadrado=cuadrado;
    input_directory=input_directory+"/";
    string strPrefix;
    DIR    *dir_p = opendir (input_directory.c_str());
    struct dirent *dir_entry_p;
    std::vector<cv::Mat> img;
    std::vector<float> etiquetas;
    int cont=0;
    int contador=0;
    int contador_recortes=0;
    string output_directory="../Data/Imagenes/"+nombre+"/";
    string archivo_imagenes=output_directory+"Images.xml";
    string archivo_imagenes_recortes=output_directory+"Recortes.xml";
    string archivo_recortes=output_directory+"Recortes.txt";
    string archivo_info=output_directory+"Info.xml";
    string g=output_directory+"Config.xml";
    cv::FileStorage archivo_r(g,FileStorage::READ);
    if(archivo_r.isOpened()){
        cv::FileStorage Archivo_i(archivo_info,FileStorage::READ);
        if(!Archivo_i.isOpened()){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        int num;
        Archivo_i["Num_Datos"]>>num;
        Archivo_i.release();
#ifdef GUI
        this->total_progreso=num;
        this->progreso=0;
#endif
        Info_Datos inff;
        Cargar_Fichero(archivo_recortes,img,Labels,inff);

        string command = "cp "+archivo_imagenes+" "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "cp "+archivo_imagenes_recortes+" "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
    }
    cv::FileStorage Archivo_img(archivo_imagenes,FileStorage::WRITE);
    cv::FileStorage Archivo_recortes(archivo_imagenes_recortes,FileStorage::WRITE);
    cv::FileStorage Archivo_i(archivo_info,FileStorage::WRITE);
    ofstream f;
    if(save){
        string dir_img="../Data/Imagenes";
        DIR    *dir_p_o = opendir (dir_img.c_str());
        if(dir_p_o == NULL) {
            string command = "mkdir "+dir_img;
            int er=system(command.c_str());
            if(er==0)
                dir_p_o = opendir (dir_img.c_str());
        }
        DIR    *dir_p_i = opendir (output_directory.c_str());
        if(dir_p_i == NULL) {
            string command = "mkdir "+output_directory;
            int er=system(command.c_str());
            if(er==0)
                dir_p_i = opendir (output_directory.c_str());
        }
        Archivo_img.open(archivo_imagenes,FileStorage::WRITE);
        Archivo_recortes.open(archivo_imagenes_recortes,FileStorage::WRITE);
        Archivo_i.open(archivo_info,FileStorage::WRITE);
    }
    if(archivo_r.isOpened()){
        cv::FileStorage aux_Images(output_directory+"aux_Images.xml",FileStorage::READ);
        FileNode nodo1 = aux_Images.root();
        for (FileNodeIterator current = nodo1.begin(); current != nodo1.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Images[nom.c_str()]>>v;
                Archivo_img<<nom<<v;
        }
        cv::FileStorage aux_Recortes(output_directory+"aux_Recortes.xml",FileStorage::READ);
        FileNode nodo2 = aux_Recortes.root();
        for (FileNodeIterator current = nodo2.begin(); current != nodo2.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Recortes[nom.c_str()]>>v;
                Archivo_recortes<<nom<<v;
        }
        aux_Images.release();
        aux_Recortes.release();
        string command = "rm "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "rm "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::app);
        int num_imagen;
        archivo_r["num_imagen"]>>num_imagen;
        cont=num_imagen-1;
        for(int i=1; i<num_imagen; i++)
            dir_entry_p = readdir(dir_p);
    }
    else if(save==true){
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::trunc);
    }
    Mat imagen;
    while((dir_entry_p = readdir(dir_p)) != NULL){
        int rotacion=0;
        ostringstream datos;
        if(strcmp(dir_entry_p->d_name, ""))
            strPrefix=input_directory+dir_entry_p->d_name;
        if(strcmp(dir_entry_p->d_name, ".")!=0 && strcmp(dir_entry_p->d_name, "..")!=0 && strcmp(dir_entry_p->d_name, "Recortes")!=0 && strcmp(dir_entry_p->d_name, "Recortes_Reescalados")!=0){
            cont++;
            cout<<"Imagen= "<<dir_entry_p->d_name<<endl;
            imagen = imread(strPrefix.c_str());
            if(imagen.empty()){
                cout<<"ERROR en Recortar_Etiquetar: Imagen vacia"<<endl;
                this->running=false;
                this->error=1;
                return this->error;
            }
            Mat Imagen(imagen.rows,imagen.cols,CV_32F);
            imagen.convertTo(Imagen,CV_32F);
            imagen.copyTo(frame);
            cv::namedWindow("Recortar_Etiquetar");
            frame.copyTo(frame2);
            cv::setMouseCallback("Recortar_Etiquetar", mouseEvent, (void *)this);
            flag=false;
            bool no_num_pulsado=false;
            bool salir=false;
            while(true){
                int a;
                char z;
                tam_x=0;
                tam_y=0;
                while(true){
                    Muestra:
                    cv::imshow("Recortar_Etiquetar",frame2);
                    z=waitKey(1);
                    if(z=='i'){
                        Mat most=Mat::zeros(200,300,CV_8UC3);
                        most=most+Scalar(255,255,255);
                        string texto="ESC=Exit";
                        putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                        texto="Num=Labeling";
                        putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                        texto="Space=Next";
                        putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                        texto="e=Rotate Left";
                        putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                        texto="r=Rotate Right";
                        putText(most,texto,Point(10,250),1,1.5,Scalar(0,0,255),2);
                        imshow("Info",most);
                        waitKey(1500);
                        destroyWindow("Info");
                        goto Muestra;
                    }
                    if(z=='e'){
                        rotacion++;
                        Mat transform=cv::getRotationMatrix2D(Point(imagen.cols/2,imagen.rows/2),rotacion,1.0);
                        cv::warpAffine(imagen,frame,transform,Size(imagen.cols,imagen.rows));
                        frame.copyTo(frame2);
                        goto Muestra;
                    }
                    if(z=='r'){
                        rotacion--;
                        Mat transform=cv::getRotationMatrix2D(Point(imagen.cols/2,imagen.rows/2),rotacion,1.0);
                        cv::warpAffine(imagen,frame,transform,Size(imagen.cols,imagen.rows));
                        frame.copyTo(frame2);
                        goto Muestra;
                    }
                    a=z;
                    if(a!=-1){
                        if(a==27){
                            cv::FileStorage archivo_w(g,FileStorage::WRITE);
                            archivo_w<<"num_imagen"<<cont;
                            archivo_w.release();
                            salir=true;
                            no_num_pulsado=true;
                            break;
                        }
                        if(a!=48 && a!=49 && a!=50 && a!=51 && a!=52 && a!=53
                                 && a!=54 && a!=55 && a!=56 && a!=57 &&
                                a!=-71 && a!=-72 && a!=-73 && a!=-74 && a!=-75
                                 && a!=-76 && a!=-77 && a!=-78 && a!=-79 && a!=-80){
                            no_num_pulsado=true;
                            break;
                        }
                        if(a<0){
                            a=a+128-'0';
                            break;
                        }
                        else{
                            a=a-'0';
                            break;
                        }
                    }
                }
                if(no_num_pulsado)
                    break;
                else{
                    if(a==0)
                        a=-1.0;
                    Mat transform=cv::getRotationMatrix2D(Point(Imagen.cols/2,Imagen.rows/2),rotacion,1.0);
                    Mat Imagen2;
                    cv::warpAffine(Imagen,Imagen2,transform,Size(Imagen.cols,Imagen.rows));
                    Mat ROI=Imagen2(cv::Rect(p_x,p_y,tam_x,tam_y));
                    if(!ROI.empty()){
                        Mat modif,modificada;
                        cv::resize(ROI,modif,tam_recorte);
                        modif.convertTo(modificada,CV_32F);
                        etiquetas.push_back(a);
                        img.push_back(modificada);
                        contador_recortes++;
                        contador++;
                        if(save==true){
                            datos<<p_x;
                            datos<<" ";
                            datos<<p_y;
                            datos<<" ";
                            datos<<tam_x;
                            datos<<" ";
                            datos<<tam_y;
                            datos<<" ";
                            datos<<rotacion;
                            datos<<" ";
                            datos<<a;
                            datos<<" ";
                            stringstream nom;
                            nom<<"Imagen"<<contador_recortes;
                            Archivo_recortes<<nom.str()<<modificada;
                        }
                        cout<<"Recorte= "<<contador_recortes<<"       Etiqueta= "<<a<<endl;
                    }
                }
            }
            if(salir==true){
                if(save==true){
                    if(contador>0){
                        stringstream nom;
                        nom<<Imagen<<cont;
                        Archivo_img<<nom.str()<<imagen;
                        f<<"Imagen";
                        f<<cont;
                        f<<" ";
                        f<<contador;
                        f<<" ";
                        f<<datos.str();
                        f<<"\n";
                    }
                }
                break;
            }
        }
        if(save==true){
            if(contador>0){
                stringstream nom;
                nom<<"Imagen"<<cont;
                Archivo_img<<nom.str()<<imagen;
                f<<"Imagen";
                f<<cont;
                f<<" ";
                f<<contador;
                f<<" ";
                f<<datos.str();
                f<<"\n";
            }
        }
        contador=0;
    }
    Labels=etiquetas;
    imagenes=img;
    destroyAllWindows();
    if(Labels.size()==0 || imagenes.size()==0 || Labels.size()!=imagenes.size()){
        cout<<"ERROR en Recortar_Etiquetar: El resultado de las etiquetas e imagenes no tienen el mismo tamaño o son 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    info.Tipo_Datos=0;
    info.Num_Datos=imagenes.size();
    info.DS=Mat();
    info.D_PRIME=Mat();
    info.LDA=Mat();
    info.PCA=Mat();
    info.si_dist=false;
    info.si_d_prime=false;
    info.si_lda=false;
    info.si_pca=false;
    info.Tam_Orig_X=imagenes[0].cols;
    info.Tam_Orig_Y=imagenes[0].rows;
    info.Tam_X=imagenes[0].cols;
    info.Tam_Y=imagenes[0].rows;
    if(save){
        Archivo_i<<"Tipo_Datos"<<info.Tipo_Datos;
        Archivo_i<<"Num_Datos"<<info.Num_Datos;
        Archivo_i<<"DS"<<info.DS;
        Archivo_i<<"D_PRIME"<<info.D_PRIME;
        Archivo_i<<"LDA"<<info.LDA;
        Archivo_i<<"PCA"<<info.PCA;
        Archivo_i<<"si_dist"<<info.si_dist;
        Archivo_i<<"si_d_prime"<<info.si_d_prime;
        Archivo_i<<"si_lda"<<info.si_lda;
        Archivo_i<<"si_pca"<<info.si_pca;
        Archivo_i<<"Tam_Orig_X"<<info.Tam_Orig_X;
        Archivo_i<<"Tam_Orig_Y"<<info.Tam_Orig_Y;
        Archivo_i<<"Tam_X"<<info.Tam_X;
        Archivo_i<<"Tam_Y"<<info.Tam_Y;
    }
    f.close();
    Archivo_i.release();
    Archivo_img.release();
    Archivo_recortes.release();
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Recortar_Etiquetar_video(string nombre, VideoCapture cap, bool cuadrado, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save){
    this->running=true;
    if(tam_recorte.width<1 || tam_recorte.height<1){
        cout<<"ERROR en Recortar_Etiquetar: El tamaño del recorte debe ser mayor a 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    Cuadrado=cuadrado;
    if(!cap.isOpened()){
        cout<<"ERROR en Recortar_Etiquetar: No se ha podido cargar el video"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    std::vector<cv::Mat> img;
    std::vector<float> etiquetas;
    int cont=0;
    int contador=0;
    int contador_recortes=0;
    string output_directory="../Data/Imagenes/"+nombre+"/";
    string archivo_imagenes=output_directory+"Images.xml";
    string archivo_imagenes_recortes=output_directory+"Recortes.xml";
    string archivo_recortes=output_directory+"Recortes.txt";
    string archivo_info=output_directory+"Info.xml";
    string g=output_directory+"Config.xml";
    cv::FileStorage archivo_r(g,FileStorage::READ);
    if(archivo_r.isOpened()){
        cv::FileStorage Archivo_i(archivo_info,FileStorage::READ);
        if(!Archivo_i.isOpened()){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        int num;
        Archivo_i["Num_Datos"]>>num;
        Archivo_i.release();
#ifdef GUI
        total_progreso=num;
        progreso=0;
#endif
        Info_Datos inff;
        Cargar_Fichero(archivo_recortes,img,Labels,inff);
#ifdef GUI
//        window->v_progress_datamanaging->setValue(0);
//        window->i_progress_datamanaging->setValue(0);
#endif
        string command = "cp "+archivo_imagenes+" "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "cp "+archivo_imagenes_recortes+" "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
    }
    cv::FileStorage Archivo_img(archivo_imagenes,FileStorage::WRITE);
    cv::FileStorage Archivo_recortes(archivo_imagenes_recortes,FileStorage::WRITE);
    cv::FileStorage Archivo_i(archivo_info,FileStorage::WRITE);
    ofstream f;
    if(save){
        string dir_img="../Data/Imagenes";
        DIR    *dir_p_o = opendir (dir_img.c_str());
        if(dir_p_o == NULL) {
            string command = "mkdir "+dir_img;
            int er=system(command.c_str());
            if(er==0)
                dir_p_o = opendir (dir_img.c_str());
        }
        DIR    *dir_p_i = opendir (output_directory.c_str());
        if(dir_p_i == NULL) {
            string command = "mkdir "+output_directory;
            int er=system(command.c_str());
            if(er==0)
                dir_p_i = opendir (output_directory.c_str());
        }
        Archivo_img.open(archivo_imagenes,FileStorage::WRITE);
        Archivo_recortes.open(archivo_imagenes_recortes,FileStorage::WRITE);
        Archivo_i.open(archivo_info,FileStorage::WRITE);
    }
    Mat imagen;
    if(archivo_r.isOpened()){
        cv::FileStorage aux_Images(output_directory+"aux_Images.xml",FileStorage::READ);
        FileNode nodo1 = aux_Images.root();
        for (FileNodeIterator current = nodo1.begin(); current != nodo1.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Images[nom.c_str()]>>v;
                Archivo_img<<nom<<v;
        }
        cv::FileStorage aux_Recortes(output_directory+"aux_Recortes.xml",FileStorage::READ);
        FileNode nodo2 = aux_Recortes.root();
        for (FileNodeIterator current = nodo2.begin(); current != nodo2.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Recortes[nom.c_str()]>>v;
                Archivo_recortes<<nom<<v;
        }
        aux_Images.release();
        aux_Recortes.release();
        string command = "rm "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "rm "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::app);
        int num_imagen;
        archivo_r["num_imagen"]>>num_imagen;
        cont=num_imagen-1;
        for(int i=1; i<num_imagen; i++)
            cap>>imagen;
    }
    else if(save==true){
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::trunc);
    }
    cap>>imagen;
    int rotacion=0;
    while(!imagen.empty()){
        ostringstream datos;
        cont++;
        cout<<"Imagen= "<<cont<<endl;
        if(imagen.empty()){
            cout<<"ERROR en Recortar_Etiquetar: Imagen vacia"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        Mat Imagen(imagen.rows,imagen.cols,CV_32F);
        imagen.convertTo(Imagen,CV_32F);
        imagen.copyTo(frame);
        cv::namedWindow("Recortar_Etiquetar");
        frame.copyTo(frame2);
        cv::setMouseCallback("Recortar_Etiquetar", mouseEvent, (void *)this);
        flag=false;
        bool no_num_pulsado=false;
        bool salir=false;
        while(true){
            int a;
            char z;
            tam_x=0;
            tam_y=0;
            while(true){
                Muestra:
                cv::imshow("Recortar_Etiquetar",frame2);
                z=waitKey(1);
                if(z=='i'){
                    Mat most=Mat::zeros(200,300,CV_8UC3);
                    most=most+Scalar(255,255,255);
                    string texto="ESC=Exit";
                    putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                    texto="Num=Labeling";
                    putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                    texto="Space=Next";
                    putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                    texto="e=Rotate Left";
                    putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                    texto="r=Rotate Right";
                    putText(most,texto,Point(10,250),1,1.5,Scalar(0,0,255),2);
                    imshow("Info",most);
                    waitKey(1500);
                    destroyWindow("Info");
                    goto Muestra;
                }
                if(z=='e'){
                    rotacion++;
                    Mat transform=cv::getRotationMatrix2D(Point(imagen.cols/2,imagen.rows/2),rotacion,1.0);
                    cv::warpAffine(imagen,frame,transform,Size(imagen.cols,imagen.rows));
                    frame.copyTo(frame2);
                    goto Muestra;
                }
                if(z=='r'){
                    rotacion--;
                    Mat transform=cv::getRotationMatrix2D(Point(imagen.cols/2,imagen.rows/2),rotacion,1.0);
                    cv::warpAffine(imagen,frame,transform,Size(imagen.cols,imagen.rows));
                    frame.copyTo(frame2);
                    goto Muestra;
                }
                a=z;
                if(a!=-1){
                    if(a==27){
                        cv::FileStorage archivo_w(g,FileStorage::WRITE);
                        archivo_w<<"num_imagen"<<cont;
                        archivo_w.release();
                        salir=true;
                        no_num_pulsado=true;
                        break;
                    }
                    if(a!=48 && a!=49 && a!=50 && a!=51 && a!=52 && a!=53
                             && a!=54 && a!=55 && a!=56 && a!=57 &&
                            a!=-71 && a!=-72 && a!=-73 && a!=-74 && a!=-75
                             && a!=-76 && a!=-77 && a!=-78 && a!=-79 && a!=-80){
                        no_num_pulsado=true;
                        break;
                    }
                    if(a<0){
                        a=a+128-'0';
                        break;
                    }
                    else{
                        a=a-'0';
                        break;
                    }
                }
            }
            if(no_num_pulsado)
                break;
            else{
                if(a==0)
                    a=-1.0;
                Mat transform=cv::getRotationMatrix2D(Point(Imagen.cols/2,Imagen.rows/2),rotacion,1.0);
                Mat Imagen2;
                cv::warpAffine(Imagen,Imagen2,transform,Size(Imagen.cols,Imagen.rows));
                Mat ROI=Imagen2(cv::Rect(p_x,p_y,tam_x,tam_y));
                if(!ROI.empty()){
                    Mat modif,modificada;
                    cv::resize(ROI,modif,tam_recorte);
                    modif.convertTo(modificada,CV_32F);
                    etiquetas.push_back(a);
                    img.push_back(modificada);
                    contador_recortes++;
                    contador++;
                    if(save==true){
                        datos<<p_x;
                        datos<<" ";
                        datos<<p_y;
                        datos<<" ";
                        datos<<tam_x;
                        datos<<" ";
                        datos<<tam_y;
                        datos<<" ";
                        datos<<rotacion;
                        datos<<" ";
                        datos<<a;
                        datos<<" ";
                        stringstream nom;
                        nom<<"Imagen"<<contador_recortes;
                        Archivo_recortes<<nom.str()<<modificada;
                    }
                    cout<<"Recorte= "<<contador_recortes<<"       Etiqueta= "<<a<<endl;
                }
            }
        }
        if(salir==true){
            if(save==true){
                if(contador>0){
                    stringstream nom;
                    nom<<"Imagen"<<cont;
                    Archivo_img<<nom.str()<<imagen;
                    f<<"Imagen";
                    f<<cont;
                    f<<" ";
                    f<<contador;
                    f<<" ";
                    f<<datos.str();
                    f<<"\n";
                }
            }
            break;
        }
        if(save==true){
            if(contador>0){
                stringstream nom;
                nom<<"Imagen"<<cont;
                Archivo_img<<nom.str()<<imagen;
                f<<"Imagen";
                f<<cont;
                f<<" ";
                f<<contador;
                f<<" ";
                f<<datos.str();
                f<<"\n";
            }
        }
        contador=0;
        cap>>imagen;
    }
    Labels=etiquetas;
    imagenes=img;
    destroyAllWindows();
    if(Labels.size()==0 || imagenes.size()==0 || Labels.size()!=imagenes.size()){
        cout<<"ERROR en Recortar_Etiquetar: El resultado de las etiquetas e imagenes no tienen el mismo tamaño o son 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    info.Tipo_Datos=0;
    info.Num_Datos=imagenes.size();
    info.DS=Mat();
    info.D_PRIME=Mat();
    info.LDA=Mat();
    info.PCA=Mat();
    info.si_dist=false;
    info.si_d_prime=false;
    info.si_lda=false;
    info.si_pca=false;
    info.Tam_Orig_X=imagenes[0].cols;
    info.Tam_Orig_Y=imagenes[0].rows;
    info.Tam_X=imagenes[0].cols;
    info.Tam_Y=imagenes[0].rows;
    if(save){
        Archivo_i<<"Tipo_Datos"<<info.Tipo_Datos;
        Archivo_i<<"Num_Datos"<<info.Num_Datos;
        Archivo_i<<"DS"<<info.DS;
        Archivo_i<<"D_PRIME"<<info.D_PRIME;
        Archivo_i<<"LDA"<<info.LDA;
        Archivo_i<<"PCA"<<info.PCA;
        Archivo_i<<"si_dist"<<info.si_dist;
        Archivo_i<<"si_d_prime"<<info.si_d_prime;
        Archivo_i<<"si_lda"<<info.si_lda;
        Archivo_i<<"si_pca"<<info.si_pca;
        Archivo_i<<"Tam_Orig_X"<<info.Tam_Orig_X;
        Archivo_i<<"Tam_Orig_Y"<<info.Tam_Orig_Y;
        Archivo_i<<"Tam_X"<<info.Tam_X;
        Archivo_i<<"Tam_Y"<<info.Tam_Y;
    }
    f.close();
    Archivo_i.release();
    Archivo_img.release();
    Archivo_recortes.release();
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Random_Synthetic_Data(string nombre, int num_clases, int num_data_clase, Size tam_img, float ancho, float separacion_clases, vector<Mat> &Data, vector<float> &Labels, Info_Datos &info, bool save){
    this->running=true;
    Data.clear();
    Labels.clear();
    if(num_clases<1){
        cout<<"ERROR en Random_Synthetic_Data: num_clases es menor a uno";
        this->running=false;
        this->error=1;
        return this->error;
    }
    if(num_data_clase<1){
        cout<<"ERROR en Random_Synthetic_Data: num_data_clase es menor a uno";
        this->running=false;
        this->error=1;
        return this->error;
    }
    if(tam_img.height<1 || tam_img.width<1){
        cout<<"ERROR en Random_Synthetic_Data: tam_img tiene un tamaño menor a uno";
        this->running=false;
        this->error=1;
        return this->error;
    }
    if(ancho<0){
        cout<<"ERROR en Random_Synthetic_Data: ancho es menor a cero";
        this->running=false;
        this->error=1;
        return this->error;
    }
    if(separacion_clases<0){
        cout<<"ERROR en Random_Synthetic_Data: separacion_clases es menor a cero";
        this->running=false;
        this->error=1;
        return this->error;
    }
    Data.clear();
    Labels.clear();
    int cont=0;
    string output_directory="../Data/Imagenes/"+nombre+"/";
    string archivo_imagenes=output_directory+"Images.xml";
    string archivo_imagenes_recortes=output_directory+"Recortes.xml";
    string archivo_recortes=output_directory+"Recortes.txt";
    string archivo_info=output_directory+"Info.xml";
    cv::FileStorage Archivo_img(archivo_imagenes,FileStorage::WRITE);
    cv::FileStorage Archivo_recortes(archivo_imagenes_recortes,FileStorage::WRITE);
    cv::FileStorage Archivo_i(archivo_info,FileStorage::WRITE);
    ofstream f;
    if(save){
        string dir_img="../Data/Imagenes";
        DIR    *dir_p_o = opendir (dir_img.c_str());
        if(dir_p_o == NULL) {
            string command = "mkdir "+dir_img;
            int er=system(command.c_str());
            if(er==0)
                dir_p_o = opendir (dir_img.c_str());
        }
        DIR    *dir_p_i = opendir (output_directory.c_str());
        if(dir_p_i == NULL) {
            string command = "mkdir "+output_directory;
            int er=system(command.c_str());
            if(er==0)
                dir_p_i = opendir (output_directory.c_str());
        }
        Archivo_img.open(archivo_imagenes,FileStorage::WRITE);
        Archivo_recortes.open(archivo_imagenes_recortes,FileStorage::WRITE);
        Archivo_i.open(archivo_info,FileStorage::WRITE);
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::trunc);
    }
    float separacion=0;
#ifdef GUI
    this->total_progreso=num_clases*num_data_clase;
#endif
    for(int i=0; i<num_clases; i++){
        for(int j=0; j<num_data_clase; j++){
            cont++;
            if(i==0)
                Labels.push_back(-1.0);
            else
                Labels.push_back(i);
            Mat aux=Mat::zeros(tam_img.height,tam_img.width,CV_32F);
            randu(aux,0,ancho);
            aux=aux+separacion;
            Data.push_back(aux);
            if(save==true){
                ostringstream datos;
                datos<<"Imagen";
                datos<<cont;
                datos<<" ";
                datos<<1;
                datos<<" ";
                datos<<0;
                datos<<" ";
                datos<<0;
                datos<<" ";
                datos<<aux.cols-1;
                datos<<" ";
                datos<<aux.rows-1;
                datos<<" ";
                datos<<0;
                datos<<" ";
                datos<<Labels[Labels.size()-1];
                datos<<" ";
                f<<datos.str();
                f<<"\n";
                ostringstream nom;
                nom<<"Imagen";
                nom<<cont;
                Archivo_img<<nom.str()<<aux;
                Archivo_recortes<<nom.str()<<aux;
            }
#ifdef GUI
            this->progreso++;
//            this->window->v_progress_datamanaging->setValue(base_progreso+(max_progreso*progreso/total_progreso));
//            this->window->i_progress_datamanaging->setValue(base_progreso+(max_progreso*progreso/total_progreso));
#endif
        }
        separacion=separacion+separacion_clases;
    }
    if(Labels.size()==0 || Data.size()==0 || Labels.size()!=Data.size()){
        cout<<"ERROR en Random_Synthetic_Data: El resultado de las etiquetas e imagenes no tienen el mismo tamaño o son 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    info.Tipo_Datos=1;
    info.Num_Datos=Data.size();
    info.DS=Mat();
    info.D_PRIME=Mat();
    info.LDA=Mat();
    info.PCA=Mat();
    info.si_dist=false;
    info.si_d_prime=false;
    info.si_lda=false;
    info.si_pca=false;
    info.Tam_Orig_X=Data[0].cols;
    info.Tam_Orig_Y=Data[0].rows;
    info.Tam_X=Data[0].cols;
    info.Tam_Y=Data[0].rows;
    if(save){
        Archivo_i<<"Tipo_Datos"<<info.Tipo_Datos;
        Archivo_i<<"Num_Datos"<<info.Num_Datos;
        Archivo_i<<"DS"<<info.DS;
        Archivo_i<<"D_PRIME"<<info.D_PRIME;
        Archivo_i<<"LDA"<<info.LDA;
        Archivo_i<<"PCA"<<info.PCA;
        Archivo_i<<"si_dist"<<info.si_dist;
        Archivo_i<<"si_d_prime"<<info.si_d_prime;
        Archivo_i<<"si_lda"<<info.si_lda;
        Archivo_i<<"si_pca"<<info.si_pca;
        Archivo_i<<"Tam_Orig_X"<<info.Tam_Orig_X;
        Archivo_i<<"Tam_Orig_Y"<<info.Tam_Orig_Y;
        Archivo_i<<"Tam_X"<<info.Tam_X;
        Archivo_i<<"Tam_Y"<<info.Tam_Y;
    }
    f.close();
    Archivo_i.release();
    Archivo_img.release();
    Archivo_recortes.release();
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Random_Synthetic_Image(int num_clases, Size tam_img, float ancho, float separacion_clases,  Mat &Imagen){
    this->running=true;
    if(num_clases<1){
        cout<<"ERROR en Random_Synthetic_Image: num_clases es menor a uno";
        this->running=false;
        this->error=1;
        return this->error;
    }
    if(tam_img.height<1 || tam_img.width<1){
        cout<<"ERROR en Random_Synthetic_Image: tam_img tiene un tamaño menor a uno";
        this->running=false;
        this->error=1;
        return this->error;
    }
    if(ancho<0){
        cout<<"ERROR en Random_Synthetic_Image: ancho es menor a cero";
        this->running=false;
        this->error=1;
        return this->error;
    }
    if(separacion_clases<0){
        cout<<"ERROR en Random_Synthetic_Image: separacion_clases es menor a cero";
        this->running=false;
        this->error=1;
        return this->error;
    }
    Imagen=Mat::zeros(tam_img,CV_32F);
    int max_dim=tam_img.width;
    if(tam_img.height<max_dim)
        max_dim=tam_img.height;
    int val_pix=(max_dim/2)/(num_clases+1);
    randu(Imagen,0,ancho);
    for(int i=1; i<num_clases; i++){
        Mat ROI(Imagen,Rect(val_pix*i,val_pix*i,tam_img.width-(2*val_pix*i),tam_img.height-(2*val_pix*i)));
        ROI=ROI+separacion_clases;
    }
    if(Imagen.empty()){
        cout<<"ERROR en Random_Synthetic_Image: No se ha podido generar la imagen"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Synthethic_Data(string nombre, vector<Mat> input, vector<float> inputLabels, vector<Mat> &output, vector<float> &outputLabels, int num_by_frame, float max_noise, float max_blur, float max_rot_x, float max_rot_y, float max_rot_z, Info_Datos &info, bool save){
    this->running=true;
    if(input.size()==0){
        cout<<"ERROR en Synthethic_Data: No hay datos de entrada"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    int cont=0;
    string output_directory="../Data/Imagenes/"+nombre+"/";
    string archivo_imagenes=output_directory+"Images.xml";
    string archivo_imagenes_recortes=output_directory+"Recortes.xml";
    string archivo_recortes=output_directory+"Recortes.txt";
    string archivo_info=output_directory+"Info.xml";
    cv::FileStorage Archivo_img(archivo_imagenes,FileStorage::WRITE);
    cv::FileStorage Archivo_recortes(archivo_imagenes_recortes,FileStorage::WRITE);
    cv::FileStorage Archivo_i(archivo_info,FileStorage::WRITE);
    ofstream f_a;
    if(save){
        string dir_img="../Data/Imagenes";
        DIR    *dir_p_o = opendir (dir_img.c_str());
        if(dir_p_o == NULL) {
            string command = "mkdir "+dir_img;
            int er=system(command.c_str());
            if(er==0)
                dir_p_o = opendir (dir_img.c_str());
        }
        DIR    *dir_p_i = opendir (output_directory.c_str());
        if(dir_p_i == NULL) {
            string command = "mkdir "+output_directory;
            int er=system(command.c_str());
            if(er==0)
                dir_p_i = opendir (output_directory.c_str());
        }
        Archivo_img.open(archivo_imagenes,FileStorage::WRITE);
        Archivo_recortes.open(archivo_imagenes_recortes,FileStorage::WRITE);
        Archivo_i.open(archivo_info,FileStorage::WRITE);
        f_a.open(archivo_recortes.c_str(),ofstream::out | ofstream::trunc);
    }
    for(uint i=0; i<input.size(); i++){
        for(int j=0; j<num_by_frame; j++){
            cont++;
            Mat src;
            input[i].copyTo(src);
            double rot_z=2*(max_rot_z*(((float)rand())/(float)RAND_MAX)-max_rot_z/2);
            double rot_x=2*(max_rot_x*(((float)rand())/(float)RAND_MAX)-max_rot_x/2);
            double rot_y=2*(max_rot_y*(((float)rand())/(float)RAND_MAX)-max_rot_y/2);
            double noise=2*(max_noise*(((float)rand())/(float)RAND_MAX)-max_noise/2);
            double sigma=2*(max_blur*(((float)rand())/(float)RAND_MAX)-max_blur/2);
            if(sigma>0){
                Mat blur;
                GaussianBlur(src,blur,Size(0,0),sigma);
                blur.copyTo(src);
            }
            if(noise>0){
                Mat ruido=Mat::zeros(src.rows,src.cols,src.type());
                randn(ruido,0.5,noise);
                src=src+ruido;
            }
            double halfFovy=0.5;
            double d=hypot(src.cols,src.rows);
            double sideLength=d/cos(halfFovy*(M_PI/180));
            double st=sin(rot_z*(M_PI/180));
            double ct=cos(rot_z*(M_PI/180));
            double sp=sin(rot_x*(M_PI/180));
            double cp=cos(rot_x*(M_PI/180));
            double sg=sin(rot_y*(M_PI/180));
            double cg=cos(rot_y*(M_PI/180));
            double h=d/(2.0*sin(halfFovy*(M_PI/180)));
            double n=h-(d/2.0);
            double f=h+(d/2.0);
            Mat F=Mat(4,4,CV_64FC1);
            Mat Rtheta=Mat::eye(4,4,CV_64FC1);
            Mat Rphi=Mat::eye(4,4,CV_64FC1);
            Mat Rgamma=Mat::eye(4,4,CV_64FC1);
            Mat T=Mat::eye(4,4,CV_64FC1);
            Mat P=Mat::zeros(4,4,CV_64FC1);
            Rtheta.at<double>(0,0)=Rtheta.at<double>(1,1)=ct;
            Rtheta.at<double>(0,1)=-st;Rtheta.at<double>(1,0)=st;
            Rphi.at<double>(1,1)=Rphi.at<double>(2,2)=cp;
            Rphi.at<double>(1,2)=-sp;Rphi.at<double>(2,1)=sp;
            Rgamma.at<double>(0,0)=Rgamma.at<double>(2,2)=cg;
            Rgamma.at<double>(0,2)=sg;Rgamma.at<double>(2,0)=sg;
            T.at<double>(2,3)=-h;
            P.at<double>(0,0)=P.at<double>(1,1)=1.0/tan(halfFovy*(M_PI/180));
            P.at<double>(2,2)=-(f+n)/(f-n);
            P.at<double>(2,3)=-(2.0*f*n)/(f-n);
            P.at<double>(3,2)=-1.0;
            F=P*T*Rphi*Rtheta*Rgamma;
            double ptsIn [4*3];
            double ptsOut[4*3];
            double halfW=src.cols/2, halfH=src.rows/2;
            ptsIn[0]=-halfW;ptsIn[ 1]= halfH;
            ptsIn[3]= halfW;ptsIn[ 4]= halfH;
            ptsIn[6]= halfW;ptsIn[ 7]=-halfH;
            ptsIn[9]=-halfW;ptsIn[10]=-halfH;
            ptsIn[2]=ptsIn[5]=ptsIn[8]=ptsIn[11]=0;
            Mat ptsInMat(1,4,CV_64FC3,ptsIn);
            Mat ptsOutMat(1,4,CV_64FC3,ptsOut);
            perspectiveTransform(ptsInMat,ptsOutMat,F);
            Point2f ptsInPt2f[4];
            Point2f ptsOutPt2f[4];
            for(int k=0;k<4;k++){
                Point2f ptIn (ptsIn [k*3+0], ptsIn [k*3+1]);
                Point2f ptOut(ptsOut[k*3+0], ptsOut[k*3+1]);
                ptsInPt2f[k]  = ptIn+Point2f(halfW,halfH);
                ptsOutPt2f[k] = (ptOut+Point2f(1,1))*(sideLength*0.5);
            }
            Mat persp,dst;
            Mat M=getPerspectiveTransform(ptsInPt2f,ptsOutPt2f);
            warpPerspective(src,persp,M,Size(sideLength,sideLength));
            Mat tras=Mat::eye(2,3,CV_32F);
            tras.at<float>(0,2)=-(sideLength-src.cols)/2;
            tras.at<float>(1,2)=-(sideLength-src.rows)/2;
            warpAffine(persp,dst,tras,Size(src.cols,src.rows));
            output.push_back(dst);
            outputLabels.push_back(inputLabels[i]);
            if(save==true){
                ostringstream datos;
                datos<<"Imagen";
                datos<<cont;
                datos<<" ";
                datos<<1;
                datos<<" ";
                datos<<0;
                datos<<" ";
                datos<<0;
                datos<<" ";
                datos<<dst.cols-1;
                datos<<" ";
                datos<<dst.rows-1;
                datos<<" ";
                datos<<0;
                datos<<" ";
                datos<<inputLabels[i];
                datos<<" ";
                f_a<<datos.str();
                f_a<<"\n";
                ostringstream nom;
                nom<<"Imagen";
                nom<<cont;
                Archivo_img<<nom.str()<<dst;
                Archivo_recortes<<nom.str()<<dst;
            }
        }
#ifdef GUI
        progreso++;
#endif
    }
    info.Tipo_Datos=0;
    info.Num_Datos=output.size();
    info.DS=Mat();
    info.D_PRIME=Mat();
    info.LDA=Mat();
    info.PCA=Mat();
    info.si_dist=false;
    info.si_d_prime=false;
    info.si_lda=false;
    info.si_pca=false;
    info.Tam_Orig_X=output[0].cols;
    info.Tam_Orig_Y=output[0].rows;
    info.Tam_X=output[0].cols;
    info.Tam_Y=output[0].rows;
    if(save){
        Archivo_i<<"Tipo_Datos"<<info.Tipo_Datos;
        Archivo_i<<"Num_Datos"<<info.Num_Datos;
        Archivo_i<<"DS"<<info.DS;
        Archivo_i<<"D_PRIME"<<info.D_PRIME;
        Archivo_i<<"LDA"<<info.LDA;
        Archivo_i<<"PCA"<<info.PCA;
        Archivo_i<<"si_dist"<<info.si_dist;
        Archivo_i<<"si_d_prime"<<info.si_d_prime;
        Archivo_i<<"si_lda"<<info.si_lda;
        Archivo_i<<"si_pca"<<info.si_pca;
        Archivo_i<<"Tam_Orig_X"<<info.Tam_Orig_X;
        Archivo_i<<"Tam_Orig_Y"<<info.Tam_Orig_Y;
        Archivo_i<<"Tam_X"<<info.Tam_X;
        Archivo_i<<"Tam_Y"<<info.Tam_Y;
    }
    f_a.close();
    Archivo_i.release();
    Archivo_img.release();
    Archivo_recortes.release();
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Autopositivos(string nombre, VideoCapture cap, bool cuadrado, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save){
    this->running=true;
    Cuadrado=cuadrado;
    if(!cap.isOpened()){
        cout<<"ERROR en Autopositivos: No se ha podido cargar el video"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    if(tam_recorte.height<1 || tam_recorte.width<1){
        cout<<"ERROR en Autopositivos: tamaño de recorte negativo o igual a 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    string output_directory="../Data/Imagenes/"+nombre+"/";
    string archivo_imagenes=output_directory+"Images.xml";
    string archivo_imagenes_recortes=output_directory+"Recortes.xml";
    string archivo_recortes=output_directory+"Recortes.txt";
    string archivo_info=output_directory+"Info.xml";
    string g=output_directory+"Config.xml";
    cv::FileStorage archivo_r(g,FileStorage::READ);
    if(archivo_r.isOpened()){
        cv::FileStorage Archivo_i(archivo_info,FileStorage::READ);
        if(!Archivo_i.isOpened()){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        int num;
        Archivo_i["Num_Datos"]>>num;
        Archivo_i.release();
#ifdef GUI
        total_progreso=num;
        progreso=0;
#endif
        Info_Datos inff;
        Cargar_Fichero(archivo_recortes,imagenes,Labels,inff);
#ifdef GUI
//        window->v_progress_datamanaging->setValue(0);
//        window->i_progress_datamanaging->setValue(0);
#endif
        string command = "cp "+archivo_imagenes+" "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "cp "+archivo_imagenes_recortes+" "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
    }
    cv::FileStorage Archivo_img(archivo_imagenes,FileStorage::WRITE);
    cv::FileStorage Archivo_recortes(archivo_imagenes_recortes,FileStorage::WRITE);
    cv::FileStorage Archivo_i(archivo_info,FileStorage::WRITE);
    ofstream f;
    if(save){
        string dir_img="../Data/Imagenes";
        DIR    *dir_p_o = opendir (dir_img.c_str());
        if(dir_p_o == NULL) {
            string command = "mkdir "+dir_img;
            int er=system(command.c_str());
            if(er==0)
                dir_p_o = opendir (dir_img.c_str());
        }
        DIR    *dir_p_i = opendir (output_directory.c_str());
        if(dir_p_i == NULL) {
            string command = "mkdir "+output_directory;
            int er=system(command.c_str());
            if(er==0)
                dir_p_i = opendir (output_directory.c_str());
        }
        Archivo_img.open(archivo_imagenes,FileStorage::WRITE);
        Archivo_recortes.open(archivo_imagenes_recortes,FileStorage::WRITE);
        Archivo_i.open(archivo_info,FileStorage::WRITE);
    }
    vector<Ptr<Tracker> > trackers;
    Mat imagen;
    bool cambios=false;
    vector<float> Labels_trackers;
    cv::namedWindow("Autopositivos");
    vector<Rect> Posiciones;
    Rect posicion_inicial;
    posicion_inicial.x=0;
    posicion_inicial.y=0;
    posicion_inicial.height=0;
    posicion_inicial.width=0;
    cout<<"MODO SEGUIMIENTO"<<endl;
    cout<<"e=Modo edicion   p=Pasar sin guardar ESC=Salir   resto=Pasar guardando"<<endl;
    int conta=0;
    int cont=0;
    if(archivo_r.isOpened()){
        cv::FileStorage aux_Images(output_directory+"aux_Images.xml",FileStorage::READ);
        FileNode nodo1 = aux_Images.root();
        for (FileNodeIterator current = nodo1.begin(); current != nodo1.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Images[nom.c_str()]>>v;
                Archivo_img<<nom<<v;
        }
        cv::FileStorage aux_Recortes(output_directory+"aux_Recortes.xml",FileStorage::READ);
        FileNode nodo2 = aux_Recortes.root();
        for (FileNodeIterator current = nodo2.begin(); current != nodo2.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Recortes[nom.c_str()]>>v;
                Archivo_recortes<<nom<<v;
        }
        aux_Images.release();
        aux_Recortes.release();
        string command = "rm "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "rm "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::app);
        int num_imagen;
        archivo_r["num_imagen"]>>num_imagen;
        conta=num_imagen-1;
        for(int i=0; i<num_imagen; i++)
            cap>>imagen;
    }
    else if(save==true){
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::trunc);
    }
    cap>>imagen;
    imagen.copyTo(frame);
    frame.copyTo(frame2);
    while(!imagen.empty()){
        imagen.copyTo(frame);
        imagen.copyTo(frame2);
        conta++;
        Posiciones.clear();
        vector<Ptr<Tracker> > trackers_aux;
        vector<float> Labels_trackers_aux;
        for(uint i=0; i<trackers.size(); i++){
            Rect2d posicion;
            bool encontrado=trackers[i]->update(frame,posicion);
            if(encontrado){
                Posiciones.push_back(posicion);
                trackers_aux.push_back(trackers[i]);
                Labels_trackers_aux.push_back(Labels_trackers[i]);
                cv::rectangle(frame2,posicion,cv::Scalar(0,255,0));
            }
            else
                cout<<"WARNING:Tracker perdido. Se eliminara de la lista"<<endl;
        }
        trackers=trackers_aux;
        Labels_trackers_aux=Labels_trackers;
        Mat copia;
        frame2.copyTo(copia);
        bool copiar=false;
        Muestra:
        if(copiar)
            copia.copyTo(frame2);
        imshow("Autopositivos",frame2);
        char z=waitKey(0);
        if(z=='i'){
            copiar=true;
            Mat most=Mat::zeros(250,400,CV_8UC3);
            most=most+Scalar(255,255,255);
            string texto="e=Edit Mode";
            putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
            texto="p=Not Save and Next";
            putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
            texto="ESC=Exit";
            putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
            texto="resto=Save and Next";
            putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
            imshow("Info",most);
            waitKey(1500);
            destroyWindow("Info");
            goto Muestra;
        }
        int a=z;
        if(a==27){
            cv::FileStorage archivo_w(g,FileStorage::WRITE);
            archivo_w<<"num_imagen"<<conta;
            archivo_w.release();
            f.close();
            break;
        }
        else if(z=='e'){
            cambios=true;
        }
        else if(z=='p'){
        }
        else{
            if(trackers.size()>0){
                if(save==true){
                    ostringstream nom;
                    nom << conta;
                    string salida="Imagen"+nom.str();
                    Archivo_img<<salida<<imagen;
                    f<<"Imagen";
                    f<<conta;
                    f<<" ";
                    f<<trackers.size();
                    f<<" ";
                }
                for(uint i=0; i<trackers.size(); i++){
                    cont++;
                    Mat ROI(frame,Posiciones[i]);
                    Mat modificada;
                    cv::resize(ROI,modificada,tam_recorte);
                    imagenes.push_back(modificada);
                    Labels.push_back(Labels_trackers[i]);
                    if(save==true){
                        ostringstream mon;
                        mon<<"Imagen";
                        mon<<cont;
                        f<<Posiciones[i].x;
                        f<<" ";
                        f<<Posiciones[i].y;
                        f<<" ";
                        f<<Posiciones[i].width;
                        f<<" ";
                        f<<Posiciones[i].height;
                        f<<" ";
                        f<<"0";
                        f<<" ";
                        f<<Labels_trackers[i];
                        f<<" ";
                        Archivo_recortes<<mon.str()<<modificada;
                    }
                }
                if(save==true)
                    f<<"\n";
            }
        }
        if(cambios){
            cout<<"MODO EDICION"<<endl;
            cout<<"n=Nuevo tracker  r=Reentrenar tracker    b=Borrar tracker    ESC=Salir de modo edicion"<<endl;
            while(true){
                frame.copyTo(frame2);
                for(uint i=0; i<Posiciones.size(); i++)
                        cv::rectangle(frame2,Posiciones[i],cv::Scalar(0,255,0));
                frame2.copyTo(copia);
                copiar=false;
                Muestra2:
                if(copiar)
                    copia.copyTo(frame2);
                imshow("Autopositivos",frame2);
                int tip=-1;
                char tipo=waitKey(0);
                if(tipo=='i'){
                    copiar=true;
                    Mat most=Mat::zeros(250,400,CV_8UC3);
                    most=most+Scalar(255,255,255);
                    string texto="n=New Tracker";
                    putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                    texto="r=Retrain tracker";
                    putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                    texto="b=Delete Tracker";
                    putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                    texto="ESC=Exit from Mode";
                    putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                    imshow("Info",most);
                    waitKey(1500);
                    destroyWindow("Info");
                    goto Muestra2;
                }
                tip=tipo;
                if(tip==27){
                    if(trackers.size()>0){
                        if(save==true){
                            ostringstream num;
                            num << conta;
                            string salida="Imagen"+num.str();
                            Archivo_img<<salida<<imagen;
                            f<<"Imagen";
                            f<<conta;
                            f<<" ";
                            f<<trackers.size();
                            f<<" ";
                        }
                        for(uint i=0; i<trackers.size(); i++){
                            cont++;
                            Mat ROI(frame,Posiciones[i]);
                            Mat modificada;
                            cv::resize(ROI,modificada,tam_recorte);
                            imagenes.push_back(modificada);
                            Labels.push_back(Labels_trackers[i]);
                            if(save==true){
                                ostringstream num;
                                num<<"Imagen";
                                num<<cont;
                                f<<Posiciones[i].x;
                                f<<" ";
                                f<<Posiciones[i].y;
                                f<<" ";
                                f<<Posiciones[i].width;
                                f<<" ";
                                f<<Posiciones[i].height;
                                f<<" ";
                                f<<"0";
                                f<<" ";
                                f<<Labels_trackers[i];
                                f<<" ";
                                Archivo_recortes<<num.str()<<modificada;
                            }
                        }
                        if(save==true)
                            f<<"\n";
                    }
                    cout<<"MODO SEGUIMIENTO"<<endl;
                    cout<<"e=Modo edicion   p=Pasar sin guardar esc=Salir   resto=Pasar guardando"<<endl;
                    break;
                }
                else if(tipo=='n'){
                    cout<<"AGREGAR NUEVOS TRACKERS"<<endl;
                    cout<<"Recuadra y etiqueta el nuevo tracker"<<endl;
                    tam_x=0;
                    tam_y=0;
                    cv::setMouseCallback("Autopositivos", mouseEvent, (void *)this);
                    flag=false;
                    while(true){
                        for(uint i=0; i<Posiciones.size(); i++)
                            cv::rectangle(frame2,Posiciones[i],cv::Scalar(0,255,0));
                        Muestra3:
                        imshow("Autopositivos",frame2);
                        char z=waitKey(1);
                        if(z=='i'){
                            Mat most=Mat::zeros(250,250,CV_8UC3);
                            most=most+Scalar(255,255,255);
                            string texto="Square and ";
                            putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                            texto="Labeling the ";
                            putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                            texto="New Tracker";
                            putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                            texto="ESC=Back";
                            putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                            imshow("Info",most);
                            waitKey(1500);
                            destroyWindow("Info");
                            goto Muestra3;
                        }
                        int a=z;
                        if(a!=-1){
                            if(a==27){
                                cout<<"MODO EDICION"<<endl;
                                cout<<"n=Nuevo tracker  r=Reentrenar tracker    b=Borrar tracker    esc=Salir de modo edicion"<<endl;
                                break;
                            }
                            if(a==48 || a==49 || a==50 || a==51 || a==52 || a==53
                                     || a==54 || a==55 || a==56 || a==57 ||
                                    a==-71 || a==-72 || a==-73 || a==-74 || a==-75
                                     || a==76 || a==-77 || a==-78 || a==-79 || a==-80){
                                if(a<0)
                                    a=a+128-'0';
                                else
                                    a=a-'0';
                                if(a==0)
                                    a=-1.0;
                                Ptr<TrackerTLD> tracker = TrackerTLD::create();
                                Rect bb;
                                bb.x=p_x;
                                bb.y=p_y;
                                bb.width=tam_x;
                                bb.height=tam_y;
                                tracker->init(frame,bb);
                                trackers.push_back(tracker);
                                Posiciones.push_back(bb);
                                Labels_trackers.push_back(a);
                            }
                        }
                    }
                }
                else if(tipo=='r'){
                    cout<<"REENTRENAR UN TRACKER"<<endl;
                    cout<<"Pincha un tracker y despues haz el nuevo recorte"<<endl;
                    flag2=false;
                    while(true){
                        frame.copyTo(frame2);
                        for(uint i=0; i<Posiciones.size(); i++)
                            cv::rectangle(frame2,Posiciones[i],cv::Scalar(0,255,0));
                        setMouseCallback("Autopositivos",mouseEvent2, (void *)this);
                        Muestra4:
                        imshow("Autopositivos",frame2);
                        char z=waitKey(1);
                        if(z=='i'){
                            Mat most=Mat::zeros(250,250,CV_8UC3);
                            most=most+Scalar(255,255,255);
                            string texto="Click on a Tracker";
                            putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                            texto="and then square";
                            putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                            texto="a new box";
                            putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                            texto="ESC=Back";
                            putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                            imshow("Info",most);
                            waitKey(1500);
                            destroyWindow("Info");
                            goto Muestra4;
                        }
                        int a=z;
                        if(a==27){
                            cout<<"MODO EDICION"<<endl;
                            cout<<"n=Nuevo tracker  r=Reentrenar tracker    b=Borrar tracker    esc=Salir de modo edicion"<<endl;
                            break;
                        }
                        if(flag2==true){
                            int numero_cambiar=0;
                            for(uint i=0; i<Posiciones.size(); i++) {
                                if(pos_x2>Posiciones[i].x && pos_y2> Posiciones[i].y
                                        && pos_x2<Posiciones[i].x+Posiciones[i].width
                                        && pos_y2<Posiciones[i].y+Posiciones[i].height){
                                    numero_cambiar++;
                                }
                            }
                            if(numero_cambiar==1){
                                int pos=-1;
                                for(uint i=0; i<Posiciones.size(); i++) {
                                    if(pos_x2>Posiciones[i].x && pos_y2> Posiciones[i].y
                                            && pos_x2<Posiciones[i].x+Posiciones[i].width
                                            && pos_y2<Posiciones[i].y+Posiciones[i].height){
                                        Posiciones[i]=posicion_inicial;
                                        pos=i;
                                    }
                                }
                                flag=false;
                                tam_x=0;
                                tam_y=0;
                                cv::setMouseCallback("Autopositivos", mouseEvent, (void *)this);
                                while(true){
                                    for(uint i=0; i<Posiciones.size(); i++)
                                        cv::rectangle(frame2,Posiciones[i],cv::Scalar(0,255,0));
                                    Muestra7:
                                    imshow("Autopositivos",frame2);
                                    char z=waitKey(1);
                                    if(z=='i'){
                                        Mat most=Mat::zeros(250,250,CV_8UC3);
                                        most=most+Scalar(255,255,255);
                                        String texto="Click on a Tracker";
                                        putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                                        texto="and then Square";
                                        putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                                        texto="a new box";
                                        putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                                        texto="ESC=Back";
                                        putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                                        imshow("Info",most);
                                        waitKey(1500);
                                        destroyWindow("Info");
                                        goto Muestra7;
                                    }
                                    int a=z;
                                    if(a!=-1){
                                        Rect bb;
                                        bb.x=p_x;
                                        bb.y=p_y;
                                        bb.width=tam_x;
                                        bb.height=tam_y;
                                        trackers[pos].release();
                                        Ptr<TrackerTLD> tracker = TrackerTLD::create();
                                        tracker->init(frame,bb);
                                        trackers[pos]=tracker;
                                        Posiciones[pos]=bb;
                                        break;
                                    }
                                }
                            }
                            flag2=false;
                        }
                    }
                }
                else if(tipo=='b'){
                    cout<<"BORRAR UN TRACKER"<<endl;
                    cout<<"Pincha el tracker que se quiera borrar"<<endl;
                    setMouseCallback("Autopositivos",mouseEvent2, (void *)this);
                    flag2=false;
                    while(true){
                        frame.copyTo(frame2);
                        for(uint i=0; i<Posiciones.size(); i++)
                                cv::rectangle(frame2,Posiciones[i],cv::Scalar(0,255,0));
                        Muestra5:
                        imshow("Autopositivos",frame2);
                        char z=waitKey(1);
                        if(z=='i'){
                            Mat most=Mat::zeros(250,250,CV_8UC3);
                            most=most+Scalar(255,255,255);
                            string texto="Click on the Tracker";
                            putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                            texto="that you want";
                            putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                            texto="to Delete";
                            putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                            texto="ESC=BAck";
                            putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                            imshow("Info",most);
                            waitKey(1500);
                            destroyWindow("Info");
                            goto Muestra5;
                        }
                        int a=z;
                        if(a==27){
                            cout<<"MODO EDICION"<<endl;
                            cout<<"n=Nuevo tracker  r=Reentrenar tracker    b=Borrar tracker    esc=Salir de modo edicion"<<endl;
                            break;
                        }
                        if(flag2==true){
                            vector<Ptr<Tracker> > trackers_aux;
                            vector<float> labels_aux;
                            vector<Rect> posiciones_aux;
                            for(uint i=0; i<Posiciones.size(); i++) {
                                if(pos_x2>Posiciones[i].x && pos_y2> Posiciones[i].y
                                        && pos_x2<Posiciones[i].x+Posiciones[i].width
                                        && pos_y2<Posiciones[i].y+Posiciones[i].height){
                                }
                                else{
                                    trackers_aux.push_back(trackers[i]);
                                    posiciones_aux.push_back(Posiciones[i]);
                                    labels_aux.push_back(Labels_trackers[i]);
                                }
                            }
                            trackers.clear();
                            trackers=trackers_aux;
                            Posiciones.clear();
                            Posiciones=posiciones_aux;
                            Labels_trackers.clear();
                            Labels_trackers=labels_aux;
                            flag2=false;
                        }
                    }
                }
            }
            cambios=false;
        }
        cap>>imagen;
    }
    cv::destroyWindow("Autopositivos");
    info.Tipo_Datos=0;
    info.Num_Datos=imagenes.size();
    info.DS=Mat();
    info.D_PRIME=Mat();
    info.LDA=Mat();
    info.PCA=Mat();
    info.si_dist=false;
    info.si_d_prime=false;
    info.si_lda=false;
    info.si_pca=false;
    info.Tam_Orig_X=imagenes[0].cols;
    info.Tam_Orig_Y=imagenes[0].rows;
    info.Tam_X=imagenes[0].cols;
    info.Tam_Y=imagenes[0].rows;
    if(save){
        Archivo_i<<"Tipo_Datos"<<info.Tipo_Datos;
        Archivo_i<<"Num_Datos"<<info.Num_Datos;
        Archivo_i<<"DS"<<info.DS;
        Archivo_i<<"D_PRIME"<<info.D_PRIME;
        Archivo_i<<"LDA"<<info.LDA;
        Archivo_i<<"PCA"<<info.PCA;
        Archivo_i<<"si_dist"<<info.si_dist;
        Archivo_i<<"si_d_prime"<<info.si_d_prime;
        Archivo_i<<"si_lda"<<info.si_lda;
        Archivo_i<<"si_pca"<<info.si_pca;
        Archivo_i<<"Tam_Orig_X"<<info.Tam_Orig_X;
        Archivo_i<<"Tam_Orig_Y"<<info.Tam_Orig_Y;
        Archivo_i<<"Tam_X"<<info.Tam_X;
        Archivo_i<<"Tam_Y"<<info.Tam_Y;
    }
    f.close();
    Archivo_i.release();
    Archivo_img.release();
    Archivo_recortes.release();
    this->running=false;
    this->error=0;
    return this->error;
}


int MLT::Generacion::Autonegativos(string nombre, string Archivo, Size2i reescalado, int num_recortes_imagen, vector<Mat> &Negativos, vector<float> &Labels, Info_Datos &info, bool save){
    this->running=true;
    if(num_recortes_imagen<1){
        cout<<"ERROR en Autonegativos: num_recortes_imagen es menor a 0";
        this->running=false;
        this->error=1;
        return this->error;
    }
    int posicion=0;
    for(uint i=0; i<Archivo.size();i++){
        if(Archivo[i]=='/')
            posicion=i;
    }
    string nombre_archivo;
    for(uint i=posicion+1; i<Archivo.size(); i++){
        nombre_archivo=nombre_archivo+Archivo[i];
    }
    string input_directory;
    for(int i=0; i<posicion+1; i++){
        input_directory=input_directory+Archivo[i];
    }
    if(nombre_archivo!="Recortes.txt"){
        cout<<"ERROR en Autonegativos: El archivo de entrada no es del tipo esperado"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    string output_directory="../Data/Imagenes/"+nombre+"/";
    string archivo_imagenes=output_directory+"Images.xml";
    string archivo_imagenes_recortes=output_directory+"Recortes.xml";
    string archivo_recortes=output_directory+"Recortes.txt";
    string archivo_info=output_directory+"Info.xml";
    string archivo_imagenes_in=input_directory+"Images.xml";
    string g=output_directory+"Config.xml";
    cv::FileStorage archivo_r(g,FileStorage::READ);
    if(archivo_r.isOpened()){
        cv::FileStorage Archivo_i(archivo_info,FileStorage::READ);
        if(!Archivo_i.isOpened()){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        int num;
        Archivo_i["Num_Datos"]>>num;
        Archivo_i.release();
#ifdef GUI
        total_progreso=num;
        progreso=0;
#endif
        Info_Datos inff;
        Cargar_Fichero(archivo_recortes,Negativos,Labels,inff);

        string command = "cp "+archivo_imagenes+" "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "cp "+archivo_imagenes_recortes+" "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
    }
    cv::FileStorage Archivo_img(archivo_imagenes,FileStorage::WRITE);
    cv::FileStorage Archivo_img_in(archivo_imagenes_in,FileStorage::READ);
    cv::FileStorage Archivo_recortes(archivo_imagenes_recortes,FileStorage::WRITE);
    cv::FileStorage Archivo_i(archivo_info,FileStorage::WRITE);
    ofstream f;
    if(save){
        string dir_img="../Data/Imagenes";
        DIR    *dir_p_o = opendir (dir_img.c_str());
        if(dir_p_o == NULL) {
            string command = "mkdir "+dir_img;
            int er=system(command.c_str());
            if(er==0)
                dir_p_o = opendir (dir_img.c_str());
        }
        DIR    *dir_p_i = opendir (output_directory.c_str());
        if(dir_p_i == NULL) {
            string command = "mkdir "+output_directory;
            int er=system(command.c_str());
            if(er==0)
                dir_p_i = opendir (output_directory.c_str());
        }
        Archivo_img.open(archivo_imagenes,FileStorage::WRITE);
        Archivo_recortes.open(archivo_imagenes_recortes,FileStorage::WRITE);
        Archivo_i.open(archivo_info,FileStorage::WRITE);
    }
    Negativos.clear();
    Labels.clear();
    Scalar Rojo;
    Rojo[0]=0;
    Rojo[1]=0;
    Rojo[2]=255;
    Scalar Verde;
    Verde[0]=0;
    Verde[1]=255;
    Verde[2]=0;
    Scalar Azul;
    Azul[0]=255;
    Azul[1]=0;
    Azul[2]=0;
    std::ifstream input_rec(Archivo.c_str());
    std::string line;
    string path_imagen;
    string num_recortes;
    vector<Rect> recortes;
    int conta=0;
    int cuenta_recortes=0;
    if(archivo_r.isOpened()){
        cv::FileStorage aux_Images(output_directory+"aux_Images.xml",FileStorage::READ);
        FileNode nodo1 = aux_Images.root();
        for (FileNodeIterator current = nodo1.begin(); current != nodo1.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Images[nom.c_str()]>>v;
                Archivo_img<<nom<<v;
        }
        cv::FileStorage aux_Recortes(output_directory+"aux_Recortes.xml",FileStorage::READ);
        FileNode nodo2 = aux_Recortes.root();
        for (FileNodeIterator current = nodo2.begin(); current != nodo2.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Recortes[nom.c_str()]>>v;
                Archivo_recortes<<nom<<v;
        }
        aux_Images.release();
        aux_Recortes.release();
        string command = "rm "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "rm "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Etiquetar: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::app);
        int num_imagen;
        archivo_r["num_imagen"]>>num_imagen;
        conta=num_imagen-1;
        for(int i=1; i<num_imagen; i++){
            std::getline(input_rec, line);
            line.clear();
        }
    }
    else if(save==true){
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::trunc);
    }
    cout<<"Negativos: r=Nuevos negativos    BORRAR= seleccioinar + b     p=Pasar sin guardar     esc=Salir   resto=Guardar y pasar"<<endl;
    while(std::getline(input_rec, line)){
        conta++;
        vector<int> espacios;
        int cont=0;
        Rect aux;
        vector<int> rotacion;
        for(uint i=0; i<line.size(); i++){
            if(line[i]==' '){
                espacios.push_back(i);
                if(espacios.size()==1){
                    for(uint j=0; j<i; j++)
                        path_imagen=path_imagen+line[j];
//                    if(save==true){
//                        f<<path_imagen;
//                        f<<" ";
//                    }
                }
                else if(espacios.size()==2){
                    for(uint j=espacios[0]+1; j<i; j++)
                        num_recortes=num_recortes+line[j];
//                    if(save==true){
//                        f<<atoi(num_recortes.c_str())+num_recortes_imagen;
//                        f<<" ";
//                    }
                    aux.x=0;
                    aux.y=0;
                    aux.height=0;
                    aux.width=0;
                }
                else if(espacios.size()>2 && cont==0){
                    string numero;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        numero=numero+line[j];
                    aux.x=atoi(numero.c_str());
//                    if(save==true){
//                        f<<aux.x;
//                        f<<" ";
//                    }
                    cont++;
                }
                else if(espacios.size()>2 && cont==1){
                    string numero;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        numero=numero+line[j];
                    aux.y=atoi(numero.c_str());
//                    if(save==true){
//                        f<<aux.y;
//                        f<<" ";
//                    }
                    cont++;
                }
                else if(espacios.size()>2 && cont==2){
                    string numero;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        numero=numero+line[j];
                    aux.width=atoi(numero.c_str());
//                    if(save==true){
//                        f<<aux.width;
//                        f<<" ";
//                    }
                    cont++;
                }
                else if(espacios.size()>2 && cont==3){
                    string numero;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        numero=numero+line[j];
                    aux.height=atoi(numero.c_str());
//                    if(save==true){
//                        f<<aux.height;
//                        f<<" ";
//                    }
                    recortes.push_back(aux);
                    cont++;
                }
                else if(espacios.size()>2 && cont==4){
                    string rota;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        rota=rota+line[j];
                    rotacion.push_back(atoi(rota.c_str()));
//                    if(save==true){
//                        f<<atoi(rota.c_str());
//                        f<<" ";
//                    }
                    cont++;
                }
                else if(espacios.size()>2 && cont==5){
                    string numero;
                    for(uint j=espacios[espacios.size()-2]+1; j<i; j++)
                        numero=numero+line[j];
//                    if(save==true){
//                        f<<numero;
//                        f<<" ";
//                    }
                    cont=0;
                }
            }
        }
        Mat imagen;
        Archivo_img_in[path_imagen.c_str()]>>imagen;
        if(imagen.empty()){
            cout<<"ERROR en Autonegativos: Imagen vecia"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        Mat copy;
        imagen.copyTo(copy);
        vector<Rect> rects;
        for(uint i=0; i<recortes.size(); i++){
            RotatedRect rec_rot;
            rec_rot.angle=-rotacion[i];
            rec_rot.center.x=(2*recortes[i].x+recortes[i].width)/2;
            rec_rot.center.y=(2*recortes[i].y+recortes[i].height)/2;
            rec_rot.size.height=recortes[i].height;
            rec_rot.size.width=recortes[i].width;
            rects.push_back(rec_rot.boundingRect());
        }
        recortes.clear();
        for(uint i=0; i<rects.size(); i++){
            Rect rect=rects[i];
            rectangle(copy,rect,Azul);
            recortes.push_back(rect);
        }
        vector<Rect> Posiciones;
        Rect posicion;
        for(int i=0; i<num_recortes_imagen; i++){
            bool mal=true;
            int cuenta_males=0;
            while(mal){
                cuenta_males++;
                if(cuenta_males>100){
                    cout<<"ERROR en Autonegativos: No se ha podido generar negativo"<<endl;
                    this->running=false;
                    this->error=1;
                    return this->error;
                }
                posicion.x=rand()%(imagen.cols-1-recortes[0].width);
                posicion.y=rand()%(imagen.rows-1-recortes[0].height);
                posicion.height=recortes[0].height;
                posicion.width=recortes[0].width;
                mal=false;
                for(uint j=0; j<recortes.size(); j++){
                    if(!(posicion.x+posicion.width<recortes[j].x) && !(posicion.x>recortes[j].x+recortes[j].width)
                            && !(posicion.y+posicion.height<recortes[j].y) && !(posicion.y>recortes[j].y+recortes[j].height))
                        mal=true;
                }
            }
            Posiciones.push_back(posicion);
            rectangle(copy,posicion,Verde);
        }
        char z=-1;
        int a=z;
        cv::namedWindow("Recortes Negativos");
        cv::setMouseCallback("Recortes Negativos", mouseEvent2, (void *)this);
        flag2=false;
        vector<bool> para_borrar(Posiciones.size());
        pos_x2=0;
        pos_y2=0;
        while(true){
            Muestra:
            imshow("Recortes Negativos",copy);
            z=waitKey(1);
            if(z=='i'){
                Mat most=Mat::zeros(300,350,CV_8UC3);
                most=most+Scalar(255,255,255);
                string texto="r=New Negatives";
                putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                texto="DELETE= Choose + b";
                putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                texto="p=Not Save and Next";
                putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                texto="ESC=Exit";
                putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                texto="Other=Save and Next";
                putText(most,texto,Point(10,250),1,1.5,Scalar(0,0,255),2);
                imshow("Info",most);
                waitKey(1500);
                destroyWindow("Info");
                goto Muestra;
            }
            a=z;
            if(a==27)
                break;
            if(flag2==true){
                for(uint i=0; i<Posiciones.size(); i++) {
                    if(pos_x2>Posiciones[i].x && pos_y2> Posiciones[i].y
                            && pos_x2<Posiciones[i].x+Posiciones[i].width
                            && pos_y2<Posiciones[i].y+Posiciones[i].height){
                        if(para_borrar[i]==false){
                            para_borrar[i]=true;
                            rectangle(copy,Posiciones[i],Rojo);
                        }
                        else{
                            para_borrar[i]=false;
                            rectangle(copy,Posiciones[i],Verde);
                        }
                    }
                }
                flag2=false;
            }
            if(z!=-1){
                if(z=='r'){
                    imagen.copyTo(copy);
                    for(uint i=0; i<recortes.size(); i++)
                        rectangle(copy,recortes[i],Azul);
                    Posiciones.clear();
                    for(int i=0; i<num_recortes_imagen; i++){
                        Rect posicion;
                        bool mal=true;
                        int cuenta_males=0;
                        while(mal){
                            cuenta_males++;
                            if(cuenta_males>100){
                                cout<<"ERROR en Autonegativos: No se ha podido generar negativo"<<endl;
                                this->running=false;
                                this->error=1;
                                return this->error;
                            }
                            posicion.x=rand()%(imagen.cols-1-recortes[0].width);
                            posicion.y=rand()%(imagen.rows-1-recortes[0].height);
                            posicion.height=recortes[0].height;
                            posicion.width=recortes[0].width;
                            mal=false;
                            for(uint j=0; j<recortes.size(); j++){
                                if(!(posicion.x+posicion.width<recortes[j].x) && !(posicion.x>recortes[j].x+recortes[j].width)
                                        && !(posicion.y+posicion.height<recortes[j].y) && !(posicion.y>recortes[j].y+recortes[j].height))
                                    mal=true;
                            }
                        }
                        Posiciones.push_back(posicion);
                        rectangle(copy,posicion,Verde);
                    }
                }
                else if(z=='b'){
                    imagen.copyTo(copy);
                    for(uint i=0; i<recortes.size(); i++)
                        rectangle(copy,recortes[i],Azul);
                    int contador=0;
                    for(uint i=0; i<para_borrar.size(); i++){
                        if(para_borrar[i]==true){
                            Posiciones.erase(Posiciones.begin()+i);
                            contador++;
                        }
                    }
                    for(int i=0; i<contador; i++){
                        Rect posicion;
                        bool mal=true;
                        int cuenta_males=0;
                        while(mal){
                            cuenta_males++;
                            if(cuenta_males>100){
                                cout<<"ERROR en Autonegativos: No se ha podido generar negativo"<<endl;
                                this->running=false;
                                this->error=1;
                                return this->error;
                            }
                            posicion.x=rand()%(imagen.cols-1-recortes[0].width);
                            posicion.y=rand()%(imagen.rows-1-recortes[0].height);
                            posicion.height=recortes[0].height;
                            posicion.width=recortes[0].width;
                            mal=false;
                            for(uint j=0; j<recortes.size(); j++){
                                rectangle(copy,recortes[j],Azul);
                                if(!(posicion.x+posicion.width<recortes[j].x) && !(posicion.x>recortes[j].x+recortes[j].width)
                                        && !(posicion.y+posicion.height<recortes[j].y) && !(posicion.y>recortes[j].y+recortes[j].height))
                                    mal=true;
                            }
                        }
                        Posiciones.push_back(posicion);
                    }
                    for(uint i=0; i<Posiciones.size(); i++)
                        rectangle(copy,Posiciones[i],Verde);
                    for(uint i=0; i<para_borrar.size(); i++)
                        para_borrar[i]=false;
                }
                else if(z=='p')
                    break;
                else{
                    if(save==true){
                        ostringstream num;
                        num << conta;
                        string salida="Imagen"+num.str();
                        Archivo_img<<salida<<imagen;
                        f<<"Imagen";
                        f<<conta;
                        f<<" ";
                        f<<Posiciones.size();
                        f<<" ";
                    }
                    for(uint pos=0; pos<Posiciones.size(); pos++){
                        cuenta_recortes++;
                        Mat ROI(imagen,Posiciones[pos]);
                        Mat modificada;
                        cv::resize(ROI,modificada,reescalado);
                        Negativos.push_back(modificada);
                        Labels.push_back(-1.0);
                        if(save==true){
                            ostringstream nombr;
                            nombr <<"Imagen";
                            nombr<<cuenta_recortes;
                            Archivo_recortes<<nombr.str()<<modificada;
                            f<<Posiciones[pos].x;
                            f<<" ";
                            f<<Posiciones[pos].y;
                            f<<" ";
                            f<<Posiciones[pos].width;
                            f<<" ";
                            f<<Posiciones[pos].height;
                            f<<" ";
                            f<<"0";
                            f<<" ";
                            f<<"-1";
                            f<<" ";
                        }
                    }
                    if(save==true)
                        f<<"\n";
                }
            }
        }
        if(a==27){
            cv::FileStorage archivo_w(g,FileStorage::WRITE);
            conta++;
            archivo_w<<"num_imagen"<<conta;
            archivo_w.release();
            f.close();
            break;
        }
        line.clear();
        path_imagen.clear();
        num_recortes.clear();
        recortes.clear();
        rotacion.clear();
        Posiciones.clear();
    }
    cv::destroyWindow("Autonegativos");
    info.Tipo_Datos=0;
    info.Num_Datos=Negativos.size();
    info.DS=Mat();
    info.D_PRIME=Mat();
    info.LDA=Mat();
    info.PCA=Mat();
    info.si_dist=false;
    info.si_d_prime=false;
    info.si_lda=false;
    info.si_pca=false;
    info.Tam_Orig_X=Negativos[0].cols;
    info.Tam_Orig_Y=Negativos[0].rows;
    info.Tam_X=Negativos[0].cols;
    info.Tam_Y=Negativos[0].rows;
    if(save){
        Archivo_i<<"Tipo_Datos"<<info.Tipo_Datos;
        Archivo_i<<"Num_Datos"<<info.Num_Datos;
        Archivo_i<<"DS"<<info.DS;
        Archivo_i<<"D_PRIME"<<info.D_PRIME;
        Archivo_i<<"LDA"<<info.LDA;
        Archivo_i<<"PCA"<<info.PCA;
        Archivo_i<<"si_dist"<<info.si_dist;
        Archivo_i<<"si_d_prime"<<info.si_d_prime;
        Archivo_i<<"si_lda"<<info.si_lda;
        Archivo_i<<"si_pca"<<info.si_pca;
        Archivo_i<<"Tam_Orig_X"<<info.Tam_Orig_X;
        Archivo_i<<"Tam_Orig_Y"<<info.Tam_Orig_Y;
        Archivo_i<<"Tam_X"<<info.Tam_X;
        Archivo_i<<"Tam_Y"<<info.Tam_Y;
    }
    f.close();
    Archivo_i.release();
    Archivo_img.release();
    Archivo_recortes.release();
    this->running=false;
    this->error=0;
    return this->error;
}

int MLT::Generacion::Autogeneracion(string nombre, VideoCapture cap, int num_negativos_imagen, bool cuadrado, cv::Size2i tam_recorte, vector<float> &Labels, vector<Mat> &imagenes, Info_Datos &info, bool save){
    this->running=true;
    Cuadrado=cuadrado;
    if(!cap.isOpened()){
        cout<<"ERROR en Autogeneracion: No se ha podido cargar el video"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    if(tam_recorte.height<1 || tam_recorte.width<1){
        cout<<"ERROR en Autogeneracion: tamaño de recorte negativo o igual a 0"<<endl;
        this->running=false;
        this->error=1;
        return this->error;
    }
    string output_directory="../Data/Imagenes/"+nombre+"/";
    string archivo_imagenes=output_directory+"Images.xml";
    string archivo_imagenes_recortes=output_directory+"Recortes.xml";
    string archivo_recortes=output_directory+"Recortes.txt";
    string archivo_info=output_directory+"Info.xml";
    string g=output_directory+"Config.xml";
    cv::FileStorage archivo_r(g,FileStorage::READ);
    if(archivo_r.isOpened()){
        cv::FileStorage Archivo_i(archivo_info,FileStorage::READ);
        if(!Archivo_i.isOpened()){
            cout<<"ERROR en Autogeneracion: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        int num;
        Archivo_i["Num_Datos"]>>num;
        Archivo_i.release();
#ifdef GUI
        total_progreso=num;
        progreso=0;
#endif
        Info_Datos inff;
        Cargar_Fichero(archivo_recortes,imagenes,Labels,inff);
#ifdef GUI
//        window->v_progress_datamanaging->setValue(0);
//        window->i_progress_datamanaging->setValue(0);
#endif
        string command = "cp "+archivo_imagenes+" "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "cp "+archivo_imagenes_recortes+" "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Autogeneracion: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
    }
    cv::FileStorage Archivo_img(archivo_imagenes,FileStorage::WRITE);
    cv::FileStorage Archivo_recortes(archivo_imagenes_recortes,FileStorage::WRITE);
    cv::FileStorage Archivo_i(archivo_info,FileStorage::WRITE);
    ofstream f;
    if(save){
        string dir_img="../Data/Imagenes";
        DIR    *dir_p_o = opendir (dir_img.c_str());
        if(dir_p_o == NULL) {
            string command = "mkdir "+dir_img;
            int er=system(command.c_str());
            if(er==0)
                dir_p_o = opendir (dir_img.c_str());
        }
        DIR    *dir_p_i = opendir (output_directory.c_str());
        if(dir_p_i == NULL) {
            string command = "mkdir "+output_directory;
            int er=system(command.c_str());
            if(er==0)
                dir_p_i = opendir (output_directory.c_str());
        }
        Archivo_img.open(archivo_imagenes,FileStorage::WRITE);
        Archivo_recortes.open(archivo_imagenes_recortes,FileStorage::WRITE);
        Archivo_i.open(archivo_info,FileStorage::WRITE);
    }
    vector<Ptr<Tracker> > trackers;
    Scalar Rojo;
    Rojo[0]=0;
    Rojo[1]=0;
    Rojo[2]=255;
    Scalar Verde;
    Verde[0]=0;
    Verde[1]=255;
    Verde[2]=0;
    Scalar Azul;
    Azul[0]=255;
    Azul[1]=0;
    Azul[2]=0;
    Mat imagen;
    vector<float> Labels_trackers;
    cv::namedWindow("Autogeneracion");
    vector<Rect> Posiciones;
    Rect posicion_inicial;
    posicion_inicial.x=0;
    posicion_inicial.y=0;
    posicion_inicial.height=0;
    posicion_inicial.width=0;
    cout<<"MODO SEGUIMIENTO"<<endl;
    cout<<"Positivos: e=Modo edicion   p=Pasar sin guardar esc=Salir   resto=Guardar y pasar a negativos"<<endl;
    cout<<"Negativos: r=Nuevos negativos    BORRAR= seleccioinar + b     p=Pasar sin guardar     esc=Salir   resto=Guardar y pasar"<<endl;
    int conta=0;
    int cuenta_recortes=0;
    if(archivo_r.isOpened()){
        cv::FileStorage aux_Images(output_directory+"aux_Images.xml",FileStorage::READ);
        FileNode nodo1 = aux_Images.root();
        for (FileNodeIterator current = nodo1.begin(); current != nodo1.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Images[nom.c_str()]>>v;
                Archivo_img<<nom<<v;
        }
        cv::FileStorage aux_Recortes(output_directory+"aux_Recortes.xml",FileStorage::READ);
        FileNode nodo2 = aux_Recortes.root();
        for (FileNodeIterator current = nodo2.begin(); current != nodo2.end(); current++) {
                FileNode item = *current;
                string nom=item.name();
                Mat v;
                aux_Recortes[nom.c_str()]>>v;
                Archivo_recortes<<nom<<v;
        }
        aux_Images.release();
        aux_Recortes.release();
        string command = "rm "+output_directory+"aux_Images.xml";
        int er=system(command.c_str());
        if(er==0){
            command = "rm "+output_directory+"aux_Recortes.xml";
            er=system(command.c_str());
        }
        if(er==1){
            cout<<"ERROR en Autogeneracion: No se han podido recuperar los datos generados con anterioridad"<<endl;
            this->running=false;
            this->error=1;
            return this->error;
        }
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::app);
        int num_imagen;
        archivo_r["num_imagen"]>>num_imagen;
        conta=num_imagen-1;
        for(int i=0; i<num_imagen; i++)
            cap>>imagen;
    }
    else if(save==true){
        f.open(archivo_recortes.c_str(),ofstream::out | ofstream::trunc);
    }
    cap>>imagen;
    imagen.copyTo(frame);
    frame.copyTo(frame2);
    bool cambios=false;
    while(!imagen.empty()){
        imagen.copyTo(frame);
        imagen.copyTo(frame2);
        conta++;
        Posiciones.clear();
        vector<Ptr<Tracker> > trackers_aux;
        vector<float> Labels_trackers_aux;
        for(uint i=0; i<trackers.size(); i++){
            Rect2d posicion;
            bool encontrado=trackers[i]->update(frame,posicion);
            if(encontrado){
                Posiciones.push_back(posicion);
                trackers_aux.push_back(trackers[i]);
                Labels_trackers_aux.push_back(Labels_trackers[i]);
                cv::rectangle(frame2,posicion,cv::Scalar(0,255,0));
            }
            else
                cout<<"WARNING:Tracker perdido. Se eliminara de la lista"<<endl;
        }
        trackers=trackers_aux;
        Labels_trackers_aux=Labels_trackers;
        Mat copia;
        frame2.copyTo(copia);
        bool copiar=false;
        Muestra:
        if(copiar)
            copia.copyTo(frame2);
        imshow("Autogeneracion",frame2);
        char z=waitKey(0);
        if(z=='i'){
            copiar=true;
            Mat most=Mat::zeros(250,400,CV_8UC3);
            most=most+Scalar(255,255,255);
            String texto="e=Edition Mode";
            putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
            texto="p=Not Save and Next";
            putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
            texto="ESC=Exit";
            putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
            texto="resto=Save and Next";
            putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
            imshow("Info",most);
            waitKey(1500);
            destroyWindow("Info");
            goto Muestra;
        }
        int a=z;
        if(a==27){
            cv::FileStorage archivo_w(g,FileStorage::WRITE);
            archivo_w<<"num_imagen"<<conta;
            archivo_w.release();
            f.close();
            break;
        }
        else if(z=='e'){
            cambios=true;
        }
        else if(z=='p'){
        }
        else{
            int tam_recorte_x=0;
            int tam_recorte_y=0;
            if(trackers.size()>0){
                if(save==true){
                    ostringstream nom;
                    nom << conta;
                    string salida="Imagen"+nom.str();
                    Archivo_img<<salida<<imagen;
                    f<<"Imagen";
                    f<<conta;
                    f<<" ";
                    f<<trackers.size()+num_negativos_imagen;
                    f<<" ";
                }
                for(uint i=0; i<trackers.size(); i++){
                    cuenta_recortes++;
                    Mat ROI(frame,Posiciones[i]);
                    tam_recorte_x=ROI.cols;
                    tam_recorte_y=ROI.rows;
                    Mat modificada;
                    cv::resize(ROI,modificada,tam_recorte);
                    imagenes.push_back(modificada);
                    Labels.push_back(Labels_trackers[i]);
                    if(save==true){
                        ostringstream mon;
                        mon<<"Imagen";
                        mon<<cuenta_recortes;
                        f<<Posiciones[i].x;
                        f<<" ";
                        f<<Posiciones[i].y;
                        f<<" ";
                        f<<Posiciones[i].width;
                        f<<" ";
                        f<<Posiciones[i].height;
                        f<<" ";
                        f<<"0";
                        f<<" ";
                        f<<Labels_trackers[i];
                        f<<" ";
                        Archivo_recortes<<mon.str()<<modificada;
                    }
                }
                Mat copy;
                frame.copyTo(copy);
                vector<Rect> Posiciones_n;
                Rect posicion_n;
                for(int i=0; i<num_negativos_imagen; i++){
                    bool mal=true;
                    int cuenta_males=0;
                    while(mal){
                        cuenta_males++;
                        if(cuenta_males>500){
                            cout<<"ERROR en Autogeneracion: No se ha podido generar negativo"<<endl;
                            this->running=false;
                            this->error=1;
                            return this->error;
                        }
                        posicion_n.x=rand()%(imagen.cols-1-tam_recorte_x);
                        posicion_n.y=rand()%(imagen.rows-1-tam_recorte_y);
                        posicion_n.height=tam_recorte_y;
                        posicion_n.width=tam_recorte_x;
                        mal=false;
                        for(uint j=0; j<trackers.size(); j++){
                            if(!(posicion_n.x+posicion_n.width<Posiciones[j].x) && !(posicion_n.x>Posiciones[j].x+Posiciones[j].width)
                                    && !(posicion_n.y+posicion_n.height<Posiciones[j].y) && !(posicion_n.y>Posiciones[j].y+Posiciones[j].height))
                                mal=true;
                        }
                    }
                    Posiciones_n.push_back(posicion_n);
                    rectangle(copy,posicion_n,Rojo);
                }
                for(uint i=0; i<Posiciones.size(); i++){
                    rectangle(copy,Posiciones[i],Verde);
                }
                cv::setMouseCallback("Autogeneracion", mouseEvent2, (void *)this);
                flag2=false;
                vector<bool> para_borrar(Posiciones_n.size());
                pos_x2=0;
                pos_y2=0;
                char Z=-1;
                int A=Z;
                while(true){
                    Muestra2:
                    imshow("Autogeneracion",copy);
                    Z=waitKey(1);
                    if(Z=='i'){
                        Mat most=Mat::zeros(300,350,CV_8UC3);
                        most=most+Scalar(255,255,255);
                        String texto="r=New Negatives";
                        putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                        texto="DELETE= Choose + b";
                        putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                        texto="p=Not Save and Next";
                        putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                        texto="ESC=Exit";
                        putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                        texto="resto=Save and Next";
                        putText(most,texto,Point(10,250),1,1.5,Scalar(0,0,255),2);
                        imshow("Info",most);
                        waitKey(1500);
                        destroyWindow("Info");
                        goto Muestra2;
                    }
                    A=Z;
                    if(A==27)
                        break;
                    if(flag2==true){
                        for(uint i=0; i<Posiciones_n.size(); i++) {
                            if(pos_x2>Posiciones_n[i].x && pos_y2> Posiciones_n[i].y
                                    && pos_x2<Posiciones_n[i].x+Posiciones_n[i].width
                                    && pos_y2<Posiciones_n[i].y+Posiciones_n[i].height){
                                if(para_borrar[i]==false){
                                    para_borrar[i]=true;
                                    rectangle(copy,Posiciones_n[i],Azul);
                                }
                                else{
                                    para_borrar[i]=false;
                                    rectangle(copy,Posiciones_n[i],Rojo);
                                }
                            }
                        }
                        flag2=false;
                    }
                    if(Z!=-1){
                        if(Z=='r'){
                            frame2.copyTo(copy);
                            for(uint i=0; i<Posiciones.size(); i++)
                                rectangle(copy,Posiciones[i],Verde);
                            Posiciones_n.clear();
                            for(int i=0; i<num_negativos_imagen; i++){
                                Rect posicion_n;
                                bool mal=true;
                                int cuenta_males=0;
                                while(mal){
                                    cuenta_males++;
                                    if(cuenta_males>500){
                                        cout<<"ERROR en Autonegativos: No se ha podido generar negativo"<<endl;
                                        this->running=false;
                                        this->error=1;
                                        return this->error;
                                    }
                                    posicion_n.x=rand()%(imagen.cols-1-tam_recorte_x);
                                    posicion_n.y=rand()%(imagen.rows-1-tam_recorte_y);
                                    posicion_n.height=tam_recorte_y;
                                    posicion_n.width=tam_recorte_x;
                                    mal=false;
                                    for(uint j=0; j<trackers.size(); j++){
                                        if(!(posicion_n.x+posicion_n.width<Posiciones[j].x) && !(posicion_n.x>Posiciones[j].x+Posiciones[j].width)
                                                && !(posicion_n.y+posicion_n.height<Posiciones[j].y) && !(posicion_n.y>Posiciones[j].y+Posiciones[j].height))
                                            mal=true;
                                    }
                                }
                                Posiciones_n.push_back(posicion_n);
                                rectangle(copy,posicion_n,Rojo);
                            }
                        }
                        else if(Z=='b'){
                            frame2.copyTo(copy);
                            for(uint i=0; i<Posiciones.size(); i++)
                                rectangle(copy,Posiciones[i],Verde);
                            int contador=0;
                            for(uint i=0; i<para_borrar.size(); i++){
                                if(para_borrar[i]==true){
                                    Posiciones_n.erase(Posiciones_n.begin()+i);
                                    contador++;
                                }
                            }
                            for(int i=0; i<contador; i++){
                                Rect posicion_n;
                                bool mal=true;
                                while(mal){
                                    posicion_n.x=rand()%(imagen.cols-1-tam_recorte_x);
                                    posicion_n.y=rand()%(imagen.rows-1-tam_recorte_y);
                                    posicion_n.height=tam_recorte_y;
                                    posicion_n.width=tam_recorte_x;
                                    mal=false;
                                    for(uint j=0; j<trackers.size(); j++){
                                        rectangle(frame,Posiciones[j],Azul);
                                        if(!(posicion_n.x+posicion_n.width<Posiciones[j].x) && !(posicion_n.x>Posiciones[j].x+Posiciones[j].width)
                                                && !(posicion_n.y+posicion_n.height<Posiciones[j].y) && !(posicion_n.y>Posiciones[j].y+Posiciones[j].height))
                                            mal=true;
                                    }
                                }
                                Posiciones_n.push_back(posicion_n);
                            }
                            for(uint i=0; i<Posiciones_n.size(); i++)
                                rectangle(copy,Posiciones_n[i],Rojo);
                            for(uint i=0; i<para_borrar.size(); i++)
                                para_borrar[i]=false;
                        }
                        else if(Z=='p')
                            break;
                        else{
                            for(uint pos=0; pos<Posiciones_n.size(); pos++){
                                cuenta_recortes++;
                                Mat ROI(imagen,Posiciones_n[pos]);
                                Mat modificada;
                                cv::resize(ROI,modificada,tam_recorte);
                                imagenes.push_back(modificada);
                                Labels.push_back(-1.0);
                                if(save==true){
                                    ostringstream nombr;
                                    nombr <<"Imagen";
                                    nombr<<cuenta_recortes;
                                    Archivo_recortes<<nombr.str()<<modificada;
                                    f<<Posiciones_n[pos].x;
                                    f<<" ";
                                    f<<Posiciones_n[pos].y;
                                    f<<" ";
                                    f<<Posiciones_n[pos].width;
                                    f<<" ";
                                    f<<Posiciones_n[pos].height;
                                    f<<" ";
                                    f<<"0";
                                    f<<" ";
                                    f<<"-1";
                                    f<<" ";
                                }
                            }
                            if(save==true)
                                f<<"\n";
                            break;
                        }
                    }
                }
                if(A==27){
                    cv::FileStorage archivo_w(g,FileStorage::WRITE);
                    archivo_w<<"num_imagen"<<conta+1;
                    archivo_w.release();
                    f.close();
                    break;
                }
            }
        }
        if(cambios){
            cout<<"MODO EDICION"<<endl;
            cout<<"n=Nuevo tracker  r=Reentrenar tracker    b=Borrar tracker    esc=Salir de modo edicion"<<endl;
            while(true){
                frame.copyTo(frame2);
                for(uint i=0; i<Posiciones.size(); i++)
                        cv::rectangle(frame2,Posiciones[i],cv::Scalar(0,255,0));
                frame2.copyTo(copia);
                copiar=false;
                Muestra3:
                if(copiar)
                    copia.copyTo(frame2);
                imshow("Autogeneracion",frame2);
                int tip=-1;
                char tipo=waitKey(0);
                if(tipo=='i'){
                    copiar=true;
                    Mat most=Mat::zeros(250,400,CV_8UC3);
                    most=most+Scalar(255,255,255);
                    String texto="n=New Tracker";
                    putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                    texto="r=Retrain Tracker";
                    putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                    texto="b=Detele Tracker";
                    putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                    texto="ESC=Exit from Mode";
                    putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                    imshow("Info",most);
                    waitKey(1500);
                    destroyWindow("Info");
                    goto Muestra3;
                }
                tip=tipo;
                if(tip==27){
                    if(trackers.size()>0){
                        if(save==true){
                            ostringstream num;
                            num << conta;
                            string salida="Imagen"+num.str();
                            Archivo_img<<salida<<imagen;
                            f<<"Imagen";
                            f<<conta;
                            f<<" ";
                            f<<trackers.size();
                            f<<" ";
                        }
                        for(uint i=0; i<trackers.size(); i++){
                            cuenta_recortes++;
                            Mat ROI(frame,Posiciones[i]);
                            Mat modificada;
                            cv::resize(ROI,modificada,tam_recorte);
                            imagenes.push_back(modificada);
                            Labels.push_back(Labels_trackers[i]);
                            if(save==true){
                                ostringstream num;
                                num<<"Imagen";
                                num<<cuenta_recortes;
                                f<<Posiciones[i].x;
                                f<<" ";
                                f<<Posiciones[i].y;
                                f<<" ";
                                f<<Posiciones[i].width;
                                f<<" ";
                                f<<Posiciones[i].height;
                                f<<" ";
                                f<<"0";
                                f<<" ";
                                f<<Labels_trackers[i];
                                f<<" ";
                                Archivo_recortes<<num.str()<<modificada;
                            }
                        }
                        if(save==true)
                            f<<"\n";
                    }
                    cout<<"MODO SEGUIMIENTO"<<endl;
                    cout<<"Positivos: e=Modo edicion   p=Pasar sin guardar esc=Salir   resto=Guardar y pasar a negativos"<<endl;
                    cout<<"Negativos: r=Nuevos negativos    BORRAR= seleccioinar + b     p=Pasar sin guardar     esc=Salir   resto=Guardar y pasar"<<endl;
                    break;
                }
                else if(tipo=='n'){
                    cout<<"AGREGAR NUEVOS TRACKERS"<<endl;
                    cout<<"Recuadra y etiqueta el nuevo tracker"<<endl;
                    tam_x=0;
                    tam_y=0;
                    cv::setMouseCallback("Autogeneracion", mouseEvent, (void *)this);
                    flag=false;
                    while(true){
                        for(uint i=0; i<Posiciones.size(); i++)
                            cv::rectangle(frame2,Posiciones[i],cv::Scalar(0,255,0));
                        Muestra4:
                        imshow("Autogeneracion",frame2);
                        char z=waitKey(1);
                        if(z=='i'){
                            Mat most=Mat::zeros(250,250,CV_8UC3);
                            most=most+Scalar(255,255,255);
                            String texto="Square and ";
                            putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                            texto="Label the ";
                            putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                            texto="New Tracker";
                            putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                            texto="ESC=Back";
                            putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                            imshow("Info",most);
                            waitKey(1500);
                            destroyWindow("Info");
                            goto Muestra4;
                        }
                        int a=z;
                        if(a!=-1){
                            if(a==27){
                                cout<<"MODO EDICION"<<endl;
                                cout<<"n=Nuevo tracker  r=Reentrenar tracker    b=Borrar tracker    esc=Salir de modo edicion"<<endl;
                                break;
                            }
                            if(a==48 || a==49 || a==50 || a==51 || a==52 || a==53
                                     || a==54 || a==55 || a==56 || a==57 ||
                                    a==-71 || a==-72 || a==-73 || a==-74 || a==-75
                                     || a==76 || a==-77 || a==-78 || a==-79 || a==-80){
                                if(a<0)
                                    a=a+128-'0';
                                else
                                    a=a-'0';
                                if(a==0)
                                    a=-1.0;
                                Ptr<TrackerTLD> tracker = TrackerTLD::create();
                                Rect bb;
                                bb.x=p_x;
                                bb.y=p_y;
                                bb.width=tam_x;
                                bb.height=tam_y;
                                tracker->init(frame,bb);
                                trackers.push_back(tracker);
                                Posiciones.push_back(bb);
                                Labels_trackers.push_back(a);
                            }
                        }
                    }
                }
                else if(tipo=='r'){
                    cout<<"REENTRENAR UN TRACKER"<<endl;
                    cout<<"Pincha un tracker y despues haz el nuevo recorte"<<endl;
                    flag2=false;
                    while(true){
                        frame.copyTo(frame2);
                        for(uint i=0; i<Posiciones.size(); i++)
                            cv::rectangle(frame2,Posiciones[i],cv::Scalar(0,255,0));
                        setMouseCallback("Autogeneracion",mouseEvent2, (void *)this);
                        Muestra5:
                        imshow("Autogeneracion",frame2);
                        char z=waitKey(1);
                        if(z=='i'){
                            Mat most=Mat::zeros(250,250,CV_8UC3);
                            most=most+Scalar(255,255,255);
                            String texto="Click on a Tracker";
                            putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                            texto="and then Square";
                            putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                            texto="the new box";
                            putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                            texto="ESC=Back";
                            putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                            imshow("Info",most);
                            waitKey(1500);
                            destroyWindow("Info");
                            goto Muestra5;
                        }
                        int a=z;
                        if(a==27){
                            cout<<"MODO EDICION"<<endl;
                            cout<<"n=Nuevo tracker  r=Reentrenar tracker    b=Borrar tracker    esc=Salir de modo edicion"<<endl;
                            break;
                        }
                        if(flag2==true){
                            int numero_cambiar=0;
                            for(uint i=0; i<Posiciones.size(); i++) {
                                if(pos_x2>Posiciones[i].x && pos_y2> Posiciones[i].y
                                        && pos_x2<Posiciones[i].x+Posiciones[i].width
                                        && pos_y2<Posiciones[i].y+Posiciones[i].height){
                                    numero_cambiar++;
                                }
                            }
                            if(numero_cambiar==1){
                                int pos=-1;
                                for(uint i=0; i<Posiciones.size(); i++) {
                                    if(pos_x2>Posiciones[i].x && pos_y2> Posiciones[i].y
                                            && pos_x2<Posiciones[i].x+Posiciones[i].width
                                            && pos_y2<Posiciones[i].y+Posiciones[i].height){
                                        Posiciones[i]=posicion_inicial;
                                        pos=i;
                                    }
                                }
                                flag=false;
                                tam_x=0;
                                tam_y=0;
                                cv::setMouseCallback("Autogeneracion", mouseEvent, (void *)this);
                                while(true){
                                    for(uint i=0; i<Posiciones.size(); i++)
                                        cv::rectangle(frame2,Posiciones[i],cv::Scalar(0,255,0));
                                    Muestra7:
                                    imshow("Autogeneracion",frame2);
                                    char z=waitKey(1);
                                    if(z=='i'){
                                        Mat most=Mat::zeros(250,250,CV_8UC3);
                                        most=most+Scalar(255,255,255);
                                        String texto="Click on a Tracker";
                                        putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                                        texto="and then Square";
                                        putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                                        texto="the new Box";
                                        putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                                        texto="ESC=Back";
                                        putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                                        imshow("Info",most);
                                        waitKey(1500);
                                        destroyWindow("Info");
                                        goto Muestra7;
                                    }
                                    int a=z;
                                    if(a!=-1){
                                        Rect bb;
                                        bb.x=p_x;
                                        bb.y=p_y;
                                        bb.width=tam_x;
                                        bb.height=tam_y;
                                        trackers[pos].release();
                                        Ptr<TrackerTLD> tracker = TrackerTLD::create();
                                        tracker->init(frame,bb);
                                        trackers[pos]=tracker;
                                        Posiciones[pos]=bb;
                                        break;
                                    }
                                }
                            }
                            flag2=false;
                        }
                    }
                }
                else if(tipo=='b'){
                    cout<<"BORRAR UN TRACKER"<<endl;
                    cout<<"Pincha el tracker que se quiera borrar"<<endl;
                    setMouseCallback("Autogeneracion",mouseEvent2, (void *)this);
                    flag2=false;
                    while(true){
                        frame.copyTo(frame2);
                        for(uint i=0; i<Posiciones.size(); i++)
                                cv::rectangle(frame2,Posiciones[i],cv::Scalar(0,255,0));
                        Muestra6:
                        imshow("Autogeneracion",frame2);
                        char z=waitKey(1);
                        if(z=='i'){
                            Mat most=Mat::zeros(250,250,CV_8UC3);
                            most=most+Scalar(255,255,255);
                            String texto="Click on a Tracker";
                            putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
                            texto="that you want";
                            putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
                            texto="to Delete";
                            putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
                            texto="ESC=Back";
                            putText(most,texto,Point(10,200),1,1.5,Scalar(0,0,255),2);
                            imshow("Info",most);
                            waitKey(1500);
                            destroyWindow("Info");
                            goto Muestra6;
                        }
                        int a=z;
                        if(a==27){
                            cout<<"MODO EDICION"<<endl;
                            cout<<"n=Nuevo tracker  r=Reentrenar tracker    b=Borrar tracker    esc=Salir de modo edicion"<<endl;
                            break;
                        }
                        if(flag2==true){
                            vector<Ptr<Tracker> > trackers_aux;
                            vector<float> labels_aux;
                            vector<Rect> posiciones_aux;
                            for(uint i=0; i<Posiciones.size(); i++) {
                                if(pos_x2>Posiciones[i].x && pos_y2> Posiciones[i].y
                                        && pos_x2<Posiciones[i].x+Posiciones[i].width
                                        && pos_y2<Posiciones[i].y+Posiciones[i].height){
                                }
                                else{
                                    trackers_aux.push_back(trackers[i]);
                                    posiciones_aux.push_back(Posiciones[i]);
                                    labels_aux.push_back(Labels_trackers[i]);
                                }
                            }
                            trackers.clear();
                            trackers=trackers_aux;
                            Posiciones.clear();
                            Posiciones=posiciones_aux;
                            Labels_trackers.clear();
                            Labels_trackers=labels_aux;
                            flag2=false;
                        }
                    }
                }
            }
            cambios=false;
        }
        cap>>imagen;
    }
    cv::destroyWindow("Autogeneracion");
    info.Tipo_Datos=0;
    info.Num_Datos=imagenes.size();
    info.DS=Mat();
    info.D_PRIME=Mat();
    info.LDA=Mat();
    info.PCA=Mat();
    info.si_dist=false;
    info.si_d_prime=false;
    info.si_lda=false;
    info.si_pca=false;
    info.Tam_Orig_X=imagenes[0].cols;
    info.Tam_Orig_Y=imagenes[0].rows;
    info.Tam_X=imagenes[0].cols;
    info.Tam_Y=imagenes[0].rows;
    if(save){
        Archivo_i<<"Tipo_Datos"<<info.Tipo_Datos;
        Archivo_i<<"Num_Datos"<<info.Num_Datos;
        Archivo_i<<"DS"<<info.DS;
        Archivo_i<<"D_PRIME"<<info.D_PRIME;
        Archivo_i<<"LDA"<<info.LDA;
        Archivo_i<<"PCA"<<info.PCA;
        Archivo_i<<"si_dist"<<info.si_dist;
        Archivo_i<<"si_d_prime"<<info.si_d_prime;
        Archivo_i<<"si_lda"<<info.si_lda;
        Archivo_i<<"si_pca"<<info.si_pca;
        Archivo_i<<"Tam_Orig_X"<<info.Tam_Orig_X;
        Archivo_i<<"Tam_Orig_Y"<<info.Tam_Orig_Y;
        Archivo_i<<"Tam_X"<<info.Tam_X;
        Archivo_i<<"Tam_Y"<<info.Tam_Y;
    }
    f.close();
    Archivo_i.release();
    Archivo_img.release();
    Archivo_recortes.release();
    this->running=false;
    this->error=0;
    return this->error;
}















void mouseEvent(int evt, int x, int y, int flags, void* param){
    MLT::Generacion* C=(MLT::Generacion*) param;
    C->frame.copyTo(C->frame2);
    if(evt==EVENT_LBUTTONDOWN){
        C->pos_x=x;
        C->pos_y=y;
        C->tam_x=0;
        C->tam_y=0;
        C->flag=true;
    }
    if(evt==EVENT_LBUTTONUP){
        C->flag=false;
    }
    if(evt==EVENT_MOUSEMOVE && C->flag==true){
        if(C->pos_x>x){
            C->p_x=x;
            if(C->p_x<0)
                C->p_x=0;
            C->tam_x=C->pos_x-C->p_x;
        }
        else{
            C->p_x=C->pos_x;
            C->tam_x=x-C->pos_x;
        }
        if(C->pos_y>y){
           C->p_y=y;
           if(C->p_y<0)
               C->p_y=0;
           C->tam_y=C->pos_y-C->p_y;
        }
        else{
            C->p_y=C->pos_y;
            C->tam_y=y-C->pos_y;
        }
        if(C->Cuadrado){
            C->tam_x=C->tam_y;
        }
        if(C->p_x+C->tam_x>C->frame2.cols)
            C->tam_x=C->frame2.cols-C->p_x;
        if(C->p_y+C->tam_y>C->frame2.rows)
            C->tam_y=C->frame2.rows-C->p_y;
    }
    cv::rectangle(C->frame2,cv::Rect(C->p_x,C->p_y,C->tam_x,C->tam_y),cv::Scalar(255,255,255));
    flags=flags;
}

void mouseEvent2(int evt, int x, int y, int flags, void* param){
    MLT::Generacion* C=(MLT::Generacion*) param;
    if(evt==EVENT_LBUTTONDOWN){
        C->pos_x2=x;
        C->pos_y2=y;
        C->flag2=true;
    }
    flags=flags;
}

