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

#include "representacion.h"

bool ord_x (Point A, Point B) {
    return A.x<B.x;
}

MLT::Representacion::Representacion(){}

int MLT::Representacion::Color(Mat Result_Etiq, vector<cv::Scalar> Colores, Mat &Colored, bool Show){
    if(Result_Etiq.empty()){
        cout<<"ERROR en Color: Result_Etiq esta vacia"<<endl;
        return 1;
    }
    bool negativa=false;
    int num_etiq=0;
    for(int i=0; i<Result_Etiq.cols; i++){
        for(int j=0; j<Result_Etiq.rows; j++){
            if(Result_Etiq.at<float>(j,i)<0.0)
                negativa=true;
            if(num_etiq<(int)Result_Etiq.at<float>(j,i))
                num_etiq=(int)Result_Etiq.at<float>(j,i);
        }
    }
    if(negativa)
        num_etiq=num_etiq+1;
    if((uint)num_etiq>Colores.size()){
        cout<<"ERROR en Color: El numero de colores definidos es menor que el numero de etiquetas detectadas"<<endl;
        return 1;
    }
    Colored=Mat::zeros(Result_Etiq.rows,Result_Etiq.cols,CV_8UC3);
    Scalar negro;
    negro[0]=0;
    negro[1]=0;
    negro[2]=0;
    for(int i=0; i<Result_Etiq.cols; i++){
        for(int j=0; j<Result_Etiq.rows; j++){
            float pix=Result_Etiq.at<float>(j,i);
            if(pix==0.f)
                Colored.row(j).col(i)=negro;
            else if(negativa && pix==-1)
                Colored.row(j).col(i)=Colores[(int)pix+1];
            else if(negativa && pix !=-1)
                Colored.row(j).col(i)=Colores[(int)pix];
            else
                Colored.row(j).col(i)=Colores[(int)pix-1];
        }
    }
    if(Show){
        Mat most=Mat::zeros(50+(50*num_etiq),200,CV_8UC3);
        most=most+Scalar(255,255,255);
        for(int i=0; i<num_etiq; i++){
            if(i==0 && negativa){
                String texto;
                texto="Etiqueta -1";
                putText(most,texto,Point(10,50*(i+1)),1,1.5,Colores[i],2);
            }
            else if(negativa){
                stringstream tex;
                tex<<"Etiqueta "<<i;
                putText(most,tex.str(),Point(10,50*(i+1)),1,1.5,Colores[i],2);
            }
            else{
                stringstream tex;
                tex<<"Etiqueta "<<i+1;
                putText(most,tex.str(),Point(10,50*(i+1)),1,1.5,Colores[i],2);
            }
        }
        Colored.convertTo(Colored,CV_32FC3);
        double minval,maxval;
        cv::minMaxLoc(Colored,&minval,&maxval);
        Colored=(Colored-minval)/(maxval-minval);
        imshow("Leyenda",most);
        imshow("Coloreada",Colored);
        waitKey(0);
    }
    return 0;
}

int MLT::Representacion::Recuadros(Mat imagen, vector<RotatedRect> recuadros, vector<float> labels_recuadros, vector<Scalar> Colores, Mat &salida, bool show){
    if(recuadros.empty()){
        cout<<"ERROR en Recuadros: Recuadros vacio"<<endl;
        return 1;
    }
    if(labels_recuadros.empty()){
        cout<<"ERROR en Recuadros: Etiquetas vacias"<<endl;
        return 1;
    }
    if(labels_recuadros.size()!=recuadros.size()){
        cout<<"ERROR en Recuadros: El numeor de labels y recuadros no coincide"<<endl;
        return 1;
    }
    bool negativa;
    Auxiliares aux;
    int num_etiq=aux.numero_etiquetas(labels_recuadros,negativa);
    if((uint)num_etiq>Colores.size()){
        cout<<"ERROR en Recuadros: El numero de colores definidos es menor que el numero de etiquetas detectadas"<<endl;
        return 1;
    }
    Mat mostrar;
    if(imagen.channels()==1){
        Mat im;
        imagen.convertTo(im,CV_32FC1);
        double minv,maxv;
        cv::minMaxLoc(im,&minv,&maxv);
        im=255*(im-minv)/(maxv-minv);
        vector<Mat> chanels;
        chanels.push_back(im);
        chanels.push_back(im);
        chanels.push_back(im);
        merge(chanels,mostrar);
    }
    else{
        Mat im;
        imagen.convertTo(im,CV_32FC1);
        double minv,maxv;
        cv::minMaxLoc(im,&minv,&maxv);
        im=255*(im-minv)/(maxv-minv);
        im.copyTo(mostrar);
    }
    for(uint i=0; i<recuadros.size(); i++){
        if(negativa && labels_recuadros[i]==-1.0){
            Point2f vertices[4];
            recuadros[i].points(vertices);
            for (int j = 0; j < 4; j++)
                line(mostrar, vertices[j], vertices[(j+1)%4], Colores[0]);
        }
        else if(negativa && labels_recuadros[i]>0){
            Point2f vertices[4];
            recuadros[i].points(vertices);
            for (int j = 0; j < 4; j++)
                line(mostrar, vertices[j], vertices[(j+1)%4], Colores[labels_recuadros[i]]);
        }

        else{
            Point2f vertices[4];
            recuadros[i].points(vertices);
            for (int j = 0; j < 4; j++)
                line(mostrar, vertices[j], vertices[(j+1)%4], Colores[labels_recuadros[i]-1]);
        }
    }
    if(show){
        Mat most=Mat::zeros(50+(50*num_etiq),200,CV_8UC3);
        most=most+Scalar(255,255,255);
        for(int i=0; i<num_etiq; i++){
            if(i==0 && negativa){
                String texto;
                texto="Etiqueta -1";
                putText(most,texto,Point(10,50*(i+1)),1,1.5,Colores[i],2);
            }
            else if(negativa){
                stringstream tex;
                tex<<"Etiqueta "<<i;
                putText(most,tex.str(),Point(10,50*(i+1)),1,1.5,Colores[i],2);
            }
            else{
                stringstream tex;
                tex<<"Etiqueta "<<i+1;
                putText(most,tex.str(),Point(10,50*(i+1)),1,1.5,Colores[i],2);
            }
        }
        mostrar.convertTo(mostrar,CV_32FC3);
        double minval,maxval;
        cv::minMaxLoc(mostrar,&minval,&maxval);
        mostrar=(mostrar-minval)/(maxval-minval);
        imshow("Leyenda",most);
        cv::imshow("Imagen Clasificada",mostrar);
        cv::waitKey(0);
    }
    mostrar.copyTo(salida);
    return 0;
}

int MLT::Representacion::Data_represent(string nombre, vector<Mat> Data, vector<float> labels, vector<int> dimensions, vector<cv::Scalar> Colores){
    int e=0;
    if(dimensions.size()!=2 && dimensions.size()!=1){
        cout<<"ERROR en Data_represent: El número de dimensiones a representar es distintos de 1 o 2->No se puede representar"<<endl;
        return 1;
    }
#ifdef WARNINGS
    for(uint i=0; i<labels.size(); i++){
        if(labels[i]==0)
            cout<<"WARNING en Data_represent: Etiquetas con valor 0"<<endl;
    }
#endif
    Auxiliares ax;
    Mat Datos;
    e=ax.Image2Lexic(Data,Datos);
    if(e==1){
        cout<<"ERROR en Data_represent: Error en Image2lexic"<<endl;
        return 1;
    }
    bool negativa;
    int num_etiq=ax.numero_etiquetas(labels,negativa);
    if((uint)num_etiq>Colores.size()){
        cout<<"ERROR en Data_represent: El número de etiquetas es mayor que el numero de colores"<<endl;
        return 1;
    }
    Mat represent=Mat::zeros(600,650,CV_8UC3);
    represent=represent+Scalar(255,255,255);
    line(represent,Point(100,0),Point(100,560),Scalar(0,0,0),3);
    line(represent,Point(150,0),Point(150,550),Scalar(0,0,0));
    line(represent,Point(200,0),Point(200,560),Scalar(0,0,0));
    line(represent,Point(250,0),Point(250,550),Scalar(0,0,0));
    line(represent,Point(300,0),Point(300,560),Scalar(0,0,0));
    line(represent,Point(350,0),Point(350,550),Scalar(0,0,0));
    line(represent,Point(400,0),Point(400,560),Scalar(0,0,0));
    line(represent,Point(450,0),Point(450,550),Scalar(0,0,0));
    line(represent,Point(500,0),Point(500,560),Scalar(0,0,0));
    line(represent,Point(550,0),Point(550,550),Scalar(0,0,0));
    line(represent,Point(600,0),Point(600,560),Scalar(0,0,0));
    line(represent,Point(90,50),Point(650,50),Scalar(0,0,0));
    line(represent,Point(100,100),Point(650,100),Scalar(0,0,0));
    line(represent,Point(90,150),Point(650,150),Scalar(0,0,0));
    line(represent,Point(100,200),Point(650,200),Scalar(0,0,0));
    line(represent,Point(90,250),Point(650,250),Scalar(0,0,0));
    line(represent,Point(100,300),Point(650,300),Scalar(0,0,0));
    line(represent,Point(90,350),Point(650,350),Scalar(0,0,0));
    line(represent,Point(100,400),Point(650,400),Scalar(0,0,0));
    line(represent,Point(90,450),Point(650,450),Scalar(0,0,0));
    line(represent,Point(100,500),Point(650,500),Scalar(0,0,0));
    line(represent,Point(90,550),Point(650,550),Scalar(0,0,0),3);
    if(dimensions.size()==2 && (Data[0].cols*Data[0].rows)>1){
        Mat dim1=Datos.col(dimensions[0]-1);
        Mat dim2=Datos.col(dimensions[1]-1);
        float max_x=0,max_y=0;
        float min_x=999999999,min_y=999999999;
        for(int i=0; i<dim1.rows; i++){
            if(dim1.at<float>(i,0)>max_x)
                max_x=dim1.at<float>(i,0);
            if(dim1.at<float>(i,0)<min_x)
                min_x=dim1.at<float>(i,0);
            if(dim2.at<float>(i,0)>max_y)
                max_y=dim2.at<float>(i,0);
            if(dim2.at<float>(i,0)<min_y)
                min_y=dim2.at<float>(i,0);
        }
        min_y=min_y-((max_y-min_y)/20);
        min_x=min_x-((max_x-min_x)/20);
        if(min_y==max_y){
            min_y=min_y-1;
            max_y=max_y+1;
        }
        if(min_x==max_x){
            min_x=min_x-1;
            max_x=max_x+1;
        }
        if(min_x<0 && max_x>0){
            int x_o=round(100+(500*(0-min_x)/(max_x-min_x)));
            line(represent,Point(x_o,0),Point(x_o,560),Scalar(0,0,0),2);
            putText(represent,"0",Point(x_o+2,570),1,1.2,Scalar(0,0,0),2);
        }
        if(min_y<0 && max_y>0){
            int y_o=600-round(50+(500*(0.0-min_y)/(max_y-min_y)));
            line(represent,Point(90,y_o),Point(650,y_o),Scalar(0,0,0),2);
            putText(represent,"0",Point(80,y_o-2),1,1.2,Scalar(0,0,0),2);
        }
        int precision=1;
        if(abs(max_y)<0.001 || abs(max_y-min_y)<0.001)
            precision=5;
        else if(abs(max_y)<0.01 || abs(max_y-min_y)<0.01)
            precision=4;
        else if(abs(max_y)<0.1 || abs(max_y-min_y)<0.1)
            precision=3;
        else if(abs(max_y)<1 || abs(max_y-min_y)<1)
            precision=2;
        stringstream ss;
        ss.precision(precision);
        ss<<fixed<<max_y;
        String valor=ss.str();
        putText(represent,valor,Point(10,50),1,1.2,Scalar(0,0,0),2);
        stringstream ss2;
        ss2.precision(precision);
        ss2<<fixed<<min_y+(8*(max_y-min_y)/10);
        String valor2=ss2.str();
        putText(represent,valor2,Point(10,150),1,1.2,Scalar(0,0,0),2);
        stringstream ss3;
        ss3.precision(precision);
        ss3<<fixed<<min_y+(6*(max_y-min_y)/10);
        String valor3=ss3.str();
        putText(represent,valor3,Point(10,250),1,1.2,Scalar(0,0,0),2);
        stringstream ss4;
        ss4.precision(precision);
        ss4<<fixed<<min_y+(4*(max_y-min_y)/10);
        String valor4=ss4.str();
        putText(represent,valor4,Point(10,350),1,1.2,Scalar(0,0,0),2);
        stringstream ss5;
        ss5.precision(precision);
        ss5<<fixed<<min_y+(2*(max_y-min_y)/10);
        String valor5=ss5.str();
        putText(represent,valor5,Point(10,450),1,1.2,Scalar(0,0,0),2);
        stringstream ss6;
        ss6.precision(precision);
        ss6<<fixed<<min_y;
        String valor6=ss6.str();
        putText(represent,valor6,Point(10,550),1,1.2,Scalar(0,0,0),2);
        precision=1;
        if(abs(max_x)<0.001 || abs(max_x-min_x)<0.001)
            precision=5;
        else if(abs(max_x)<0.01 || abs(max_x-min_x)<0.01)
            precision=4;
        else if(abs(max_x)<0.1 || abs(max_x-min_x)<0.1)
            precision=3;
        else if(abs(max_x)<1 || abs(max_x-min_x)<1)
            precision=2;
        stringstream ss7;
        ss7.precision(precision);
        ss7<<fixed<<min_x;
        String valor7=ss7.str();
        putText(represent,valor7,Point(70,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss8;
        ss8.precision(precision);
        ss8<<fixed<<min_x+(2*(max_x-min_x)/10);
        String valor8=ss8.str();
        putText(represent,valor8,Point(170,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss9;
        ss9.precision(precision);
        ss9<<fixed<<min_x+(4*(max_x-min_x)/10);
        String valor9=ss9.str();
        putText(represent,valor9,Point(270,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss10;
        ss10.precision(precision);
        ss10<<fixed<<min_x+(6*(max_x-min_x)/10);
        String valor10=ss10.str();
        putText(represent,valor10,Point(370,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss11;
        ss11.precision(precision);
        ss11<<fixed<<min_x+(8*(max_x-min_x)/10);
        String valor11=ss11.str();
        putText(represent,valor11,Point(470,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss12;
        ss12.precision(precision);
        ss12<<fixed<<max_x;
        String valor12=ss12.str();
        putText(represent,valor12,Point(570,595),1,1.2,Scalar(0,0,0),2);
        for(int i=0; i<dim1.rows; i++){
            if(labels[i]!=0){
                Point punto;
                punto.x=round(100+(500*(dim1.at<float>(i,0)-min_x)/(max_x-min_x)));
                punto.y=600-round(50+(500*(dim2.at<float>(i,0)-min_y)/(max_y-min_y)));
                if(punto.x>100 && punto.y<550){
                    Scalar color;
                    if(negativa){
                        if(labels[i]==-1)
                            color=Colores[0];
                        else
                            color=Colores[(int)labels[i]];
                    }
                    else
                        color=Colores[(int)labels[i]-1];
                    circle(represent, punto,5,color,1);
                }
            }
        }
    }
    else if(dimensions.size()==1 || (Data[0].cols*Data[0].rows)==1){
        Mat dim1=Datos.col(dimensions[0]-1);
        float max_x=0;
        float min_x=999999999;
        for(int i=0; i<dim1.rows; i++){
            if(dim1.at<float>(i,0)>max_x)
                max_x=dim1.at<float>(i,0);
            if(dim1.at<float>(i,0)<min_x)
                min_x=dim1.at<float>(i,0);
        }
        min_x=min_x-((max_x-min_x)/20);
        if(min_x==max_x){
            min_x=min_x-1;
            max_x=max_x+1;
        }
        if(min_x<0 && max_x>0){
            int x_o=round(100+(500*(0-min_x)/(max_x-min_x)));
            line(represent,Point(x_o,0),Point(x_o,560),Scalar(0,0,0),2);
            putText(represent,"0",Point(x_o+2,570),1,1.2,Scalar(0,0,0),2);
        }
        int precision=1;
        if(abs(max_x)<0.001 || abs(max_x-min_x)<0.001)
            precision=5;
        else if(abs(max_x)<0.01 || abs(max_x-min_x)<0.01)
            precision=4;
        else if(abs(max_x)<0.1 || abs(max_x-min_x)<0.1)
            precision=3;
        else if(abs(max_x)<1 || abs(max_x-min_x)<1)
            precision=2;
        stringstream ss7;
        ss7.precision(precision);
        ss7<<fixed<<min_x;
        String valor7=ss7.str();
        putText(represent,valor7,Point(70,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss8;
        ss8.precision(precision);
        ss8<<fixed<<min_x+(2*(max_x-min_x)/10);
        String valor8=ss8.str();
        putText(represent,valor8,Point(170,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss9;
        ss9.precision(precision);
        ss9<<fixed<<min_x+(4*(max_x-min_x)/10);
        String valor9=ss9.str();
        putText(represent,valor9,Point(270,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss10;
        ss10.precision(precision);
        ss10<<fixed<<min_x+(6*(max_x-min_x)/10);
        String valor10=ss10.str();
        putText(represent,valor10,Point(370,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss11;
        ss11.precision(precision);
        ss11<<fixed<<min_x+(8*(max_x-min_x)/10);
        String valor11=ss11.str();
        putText(represent,valor11,Point(470,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss12;
        ss12.precision(precision);
        ss12<<fixed<<max_x;
        String valor12=ss12.str();
        putText(represent,valor12,Point(570,595),1,1.2,Scalar(0,0,0),2);
        for(int i=0; i<dim1.rows; i++){
            if(labels[i]!=0){
                Point punto;
                punto.x=round(100+(500*(dim1.at<float>(i,0)-min_x)/(max_x-min_x)));
                punto.y=500;
                if(punto.x>100){
                    Scalar color;
                    if(negativa){
                        if(labels[i]==-1)
                            color=Colores[0];
                        else
                            color=Colores[(int)labels[i]];
                    }
                    else
                        color=Colores[(int)labels[i]-1];
                    circle(represent, punto,5,color,1);
                }
            }
        }
    }
#ifdef GUI
    QApplication::restoreOverrideCursor();
#endif
    Mat most=Mat::zeros(50+(50*num_etiq),200,CV_8UC3);
    most=most+Scalar(255,255,255);
    for(int i=0; i<num_etiq; i++){
        if(i==0 && negativa){
            String texto;
            texto="Etiqueta -1";
            putText(most,texto,Point(10,50*(i+1)),1,1.5,Colores[i],2);
        }
        else if(negativa){
            stringstream tex;
            tex<<"Etiqueta "<<i;
            putText(most,tex.str(),Point(10,50*(i+1)),1,1.5,Colores[i],2);
        }
        else{
            stringstream tex;
            tex<<"Etiqueta "<<i+1;
            putText(most,tex.str(),Point(10,50*(i+1)),1,1.5,Colores[i],2);
        }
    }
    imshow("Leyenda",most);
    imshow(nombre,represent);
    waitKey(0);
    return 0;
}

//ELIPSES PARA EL 99 (c=3.035*500), 95(c=2.447*500) Y 50%(c=1.177*500) DE PROBABILIDAD SI SIGUIESEN UNA DISTRIBUCION CHI CUADRADO
int MLT::Representacion::Ellipse_represent(string nombre, vector<Mat> Data, vector<float> labels, vector<int> dimensions, vector<cv::Scalar> Colores){
    int e=0;
    if(dimensions.size()!=2){
        cout<<"ERROR en Ellipse_represent: El número de dimensiones a representar es distintos de 2->No se puede representar"<<endl;
        return 1;
    }
#ifdef WARNINGS
    for(uint i=0; i<labels.size(); i++){
        if(labels[i]==0)
            cout<<"ERROR en Ellipse_represent: Etiquetas con valor 0"<<endl;
    }
#endif
    Analisis comp;
    vector<Analisis::Ellipse_data> Elipses;
    e=comp.Ellipse_Error(Data,labels,dimensions,Elipses);
    if(e==1){
        cout<<"ERROR en Ellipse_represent: Error en Ellipse_Error"<<endl;
        return 1;
    }
    Auxiliares ax;
    Mat Datos;
    e=ax.Image2Lexic(Data,Datos);
    if(e==1){
        cout<<"ERROR en Ellipse_represent: Error en Image2lexic"<<endl;
        return 1;
    }
    bool negativa;
    int num_etiq=ax.numero_etiquetas(labels,negativa);
    if((uint)num_etiq>Colores.size()){
        cout<<"ERROR en Ellipse_represent: El número de etiquetas es mayor que el numero de colores"<<endl;
        return 1;
    }
    Mat dim1=Datos.col(dimensions[0]-1);
    Mat dim2=Datos.col(dimensions[1]-1);
    float max_x=0,max_y=0;
    float min_x=999999999,min_y=999999999;
    for(int i=0; i<dim1.rows; i++){
        if(dim1.at<float>(i,0)>max_x)
            max_x=dim1.at<float>(i,0);
        if(dim1.at<float>(i,0)<min_x)
            min_x=dim1.at<float>(i,0);
        if(dim2.at<float>(i,0)>max_y)
            max_y=dim2.at<float>(i,0);
        if(dim2.at<float>(i,0)<min_y)
            min_y=dim2.at<float>(i,0);
    }
    min_y=min_y-((max_y-min_y)/20);
    min_x=min_x-((max_x-min_x)/20);
    float max_total,min_total;
    if(max_x>max_y)
        max_total=max_x;
    else
        max_total=max_y;
    if(min_x<min_y)
        min_total=min_x;
    else
        min_total=min_y;
    Mat represent=Mat::zeros(600,650,CV_8UC3);
    represent=represent+Scalar(255,255,255);
    line(represent,Point(100,0),Point(100,560),Scalar(0,0,0),3);
    line(represent,Point(150,0),Point(150,550),Scalar(0,0,0));
    line(represent,Point(200,0),Point(200,560),Scalar(0,0,0));
    line(represent,Point(250,0),Point(250,550),Scalar(0,0,0));
    line(represent,Point(300,0),Point(300,560),Scalar(0,0,0));
    line(represent,Point(350,0),Point(350,550),Scalar(0,0,0));
    line(represent,Point(400,0),Point(400,560),Scalar(0,0,0));
    line(represent,Point(450,0),Point(450,550),Scalar(0,0,0));
    line(represent,Point(500,0),Point(500,560),Scalar(0,0,0));
    line(represent,Point(550,0),Point(550,550),Scalar(0,0,0));
    line(represent,Point(600,0),Point(600,560),Scalar(0,0,0));
    line(represent,Point(90,50),Point(650,50),Scalar(0,0,0));
    line(represent,Point(100,100),Point(650,100),Scalar(0,0,0));
    line(represent,Point(90,150),Point(650,150),Scalar(0,0,0));
    line(represent,Point(100,200),Point(650,200),Scalar(0,0,0));
    line(represent,Point(90,250),Point(650,250),Scalar(0,0,0));
    line(represent,Point(100,300),Point(650,300),Scalar(0,0,0));
    line(represent,Point(90,350),Point(650,350),Scalar(0,0,0));
    line(represent,Point(100,400),Point(650,400),Scalar(0,0,0));
    line(represent,Point(90,450),Point(650,450),Scalar(0,0,0));
    line(represent,Point(100,500),Point(650,500),Scalar(0,0,0));
    line(represent,Point(90,550),Point(650,550),Scalar(0,0,0),3);
    if(min_y==max_y){
        min_y=min_y-1;
        max_y=max_y+1;
    }
    if(min_x==max_x){
        min_x=min_x-1;
        max_x=max_x+1;
    }
    if(min_x<0 && max_x>0){
        int x_o=round(100+(500*(0-min_x)/(max_x-min_x)));
        line(represent,Point(x_o,0),Point(x_o,560),Scalar(0,0,0),2);
        putText(represent,"0",Point(x_o+2,570),1,1.2,Scalar(0,0,0),2);
    }
    if(min_y<0 && max_y>0){
        int y_o=600-round(50+(500*(0.0-min_y)/(max_y-min_y)));
        line(represent,Point(90,y_o),Point(650,y_o),Scalar(0,0,0),2);
        putText(represent,"0",Point(80,y_o-2),1,1.2,Scalar(0,0,0),2);
    }
    int precision=1;
    if(abs(max_y)<0.001 || abs(max_y-min_y)<0.001)
        precision=5;
    else if(abs(max_y)<0.01 || abs(max_y-min_y)<0.01)
        precision=4;
    else if(abs(max_y)<0.1 || abs(max_y-min_y)<0.1)
        precision=3;
    else if(abs(max_y)<1 || abs(max_y-min_y)<1)
        precision=2;
    stringstream ss;
    ss.precision(precision);
    ss<<fixed<<max_y;
    String valor=ss.str();
    putText(represent,valor,Point(10,50),1,1.2,Scalar(0,0,0),2);
    stringstream ss2;
    ss2.precision(precision);
    ss2<<fixed<<min_y+(8*(max_y-min_y)/10);
    String valor2=ss2.str();
    putText(represent,valor2,Point(10,150),1,1.2,Scalar(0,0,0),2);
    stringstream ss3;
    ss3.precision(precision);
    ss3<<fixed<<min_y+(6*(max_y-min_y)/10);
    String valor3=ss3.str();
    putText(represent,valor3,Point(10,250),1,1.2,Scalar(0,0,0),2);
    stringstream ss4;
    ss4.precision(precision);
    ss4<<fixed<<min_y+(4*(max_y-min_y)/10);
    String valor4=ss4.str();
    putText(represent,valor4,Point(10,350),1,1.2,Scalar(0,0,0),2);
    stringstream ss5;
    ss5.precision(precision);
    ss5<<fixed<<min_y+(2*(max_y-min_y)/10);
    String valor5=ss5.str();
    putText(represent,valor5,Point(10,450),1,1.2,Scalar(0,0,0),2);
    stringstream ss6;
    ss6.precision(precision);
    ss6<<fixed<<min_y;
    String valor6=ss6.str();
    putText(represent,valor6,Point(10,550),1,1.2,Scalar(0,0,0),2);
    precision=1;
    if(abs(max_x)<0.001 || abs(max_x-min_x)<0.001)
        precision=5;
    else if(abs(max_x)<0.01 || abs(max_x-min_x)<0.01)
        precision=4;
    else if(abs(max_x)<0.1 || abs(max_x-min_x)<0.1)
        precision=3;
    else if(abs(max_x)<1 || abs(max_x-min_x)<1)
        precision=2;
    stringstream ss7;
    ss7.precision(precision);
    ss7<<fixed<<min_x;
    String valor7=ss7.str();
    putText(represent,valor7,Point(70,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss8;
    ss8.precision(precision);
    ss8<<fixed<<min_x+(2*(max_x-min_x)/10);
    String valor8=ss8.str();
    putText(represent,valor8,Point(170,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss9;
    ss9.precision(precision);
    ss9<<fixed<<min_x+(4*(max_x-min_x)/10);
    String valor9=ss9.str();
    putText(represent,valor9,Point(270,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss10;
    ss10.precision(precision);
    ss10<<fixed<<min_x+(6*(max_x-min_x)/10);
    String valor10=ss10.str();
    putText(represent,valor10,Point(370,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss11;
    ss11.precision(precision);
    ss11<<fixed<<min_x+(8*(max_x-min_x)/10);
    String valor11=ss11.str();
    putText(represent,valor11,Point(470,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss12;
    ss12.precision(precision);
    ss12<<fixed<<max_x;
    String valor12=ss12.str();
    putText(represent,valor12,Point(570,595),1,1.2,Scalar(0,0,0),2);
    for(uint i=0; i<Elipses.size(); i++){
        Point punto;
        punto.x=round(100+(500*(Elipses[i].media_x-min_x)/(max_x-min_x)));
        punto.y=600-round(50+(500*(Elipses[i].media_y-min_y)/(max_y-min_y)));
        if(punto.x>100 && punto.y<550){
            Scalar color;
            color=Colores[i];
            circle(represent, punto,5,color,-1);
            float grande,pequeno;
            grande=((Elipses[i].big)/(max_total-min_total))*1518;
            pequeno=((Elipses[i].small)/(max_total-min_total))*1518;
            cv::ellipse(represent,punto,Size(grande,pequeno),Elipses[i].angle-90,0,360,color,2);
            grande=((Elipses[i].big)/(max_total-min_total))*1224;
            pequeno=((Elipses[i].small)/(max_total-min_total))*1224;
            cv::ellipse(represent,punto,Size(grande,pequeno),Elipses[i].angle-90,0,360,color,2);
            grande=((Elipses[i].big)/(max_total-min_total))*589;
            pequeno=((Elipses[i].small)/(max_total-min_total))*589;
            cv::ellipse(represent,punto,Size(grande,pequeno),Elipses[i].angle-90,0,360,color,2);
        }
    }
#ifdef GUI
    QApplication::restoreOverrideCursor();
#endif
    Mat most=Mat::zeros(50+(50*num_etiq),200,CV_8UC3);
    most=most+Scalar(255,255,255);
    for(int i=0; i<num_etiq; i++){
        if(i==0 && negativa){
            String texto;
            texto="Etiqueta -1";
            putText(most,texto,Point(10,50*(i+1)),1,1.5,Colores[i],2);
        }
        else if(negativa){
            stringstream tex;
            tex<<"Etiqueta "<<i;
            putText(most,tex.str(),Point(10,50*(i+1)),1,1.5,Colores[i],2);
        }
        else{
            stringstream tex;
            tex<<"Etiqueta "<<i+1;
            putText(most,tex.str(),Point(10,50*(i+1)),1,1.5,Colores[i],2);
        }
    }
    imshow("Leyenda",most);
    imshow(nombre,represent);
    waitKey(0);
    return 0;
}

//ELIPSES PARA EL 99,95 Y 50% DE PROBABILIDAD CON DISTRIBUCIÓN CHI-CUADRADO
int MLT::Representacion::Data_Ellipse_represent(string nombre, vector<Mat> Data, vector<float> labels, vector<int> dimensions, vector<cv::Scalar> Colores){
    int e=0;
    if(dimensions.size()!=2){
        cout<<"ERROR en Data_Ellipse_represent: El número de dimensiones a representar es distintos de 2->No se puede representar"<<endl;
        return 1;
    }
#ifdef WARNINGS
    for(uint i=0; i<labels.size(); i++){
        if(labels[i]==0)
            cout<<"ERROR en Data_Ellipse_represent: Etiquetas con valor 0"<<endl;
    }
#endif
    Analisis comp;
    vector<Analisis::Ellipse_data> Elipses;
    e=comp.Ellipse_Error(Data,labels,dimensions,Elipses);
    if(e==1){
        cout<<"ERROR en Data_Ellipse_represent: Error en Ellipse_Error"<<endl;
        return 1;
    }
    Auxiliares ax;
    Mat Datos;
    e=ax.Image2Lexic(Data,Datos);
    if(e==1){
        cout<<"ERROR en Data_Ellipse_represent: Error en Image2lexic"<<endl;
        return 1;
    }
    bool negativa;
    int num_etiq=ax.numero_etiquetas(labels,negativa);
    if((uint)num_etiq>Colores.size()){
        cout<<"ERROR en Data_Ellipse_represent: El número de etiquetas es mayor que el numero de colores"<<endl;
        return 1;
    }
    Mat dim1=Datos.col(dimensions[0]-1);
    Mat dim2=Datos.col(dimensions[1]-1);
    float max_x=0,max_y=0;
    float min_x=999999999,min_y=999999999;
    for(int i=0; i<dim1.rows; i++){
        if(dim1.at<float>(i,0)>max_x)
            max_x=dim1.at<float>(i,0);
        if(dim1.at<float>(i,0)<min_x)
            min_x=dim1.at<float>(i,0);
        if(dim2.at<float>(i,0)>max_y)
            max_y=dim2.at<float>(i,0);
        if(dim2.at<float>(i,0)<min_y)
            min_y=dim2.at<float>(i,0);
    }
    min_y=min_y-((max_y-min_y)/20);
    min_x=min_x-((max_x-min_x)/20);
    float max_total,min_total;
    if(max_x>max_y)
        max_total=max_x;
    else
        max_total=max_y;
    if(min_x<min_y)
        min_total=min_x;
    else
        min_total=min_y;
    Mat represent=Mat::zeros(600,650,CV_8UC3);
    represent=represent+Scalar(255,255,255);
    line(represent,Point(100,0),Point(100,560),Scalar(0,0,0),3);
    line(represent,Point(150,0),Point(150,550),Scalar(0,0,0));
    line(represent,Point(200,0),Point(200,560),Scalar(0,0,0));
    line(represent,Point(250,0),Point(250,550),Scalar(0,0,0));
    line(represent,Point(300,0),Point(300,560),Scalar(0,0,0));
    line(represent,Point(350,0),Point(350,550),Scalar(0,0,0));
    line(represent,Point(400,0),Point(400,560),Scalar(0,0,0));
    line(represent,Point(450,0),Point(450,550),Scalar(0,0,0));
    line(represent,Point(500,0),Point(500,560),Scalar(0,0,0));
    line(represent,Point(550,0),Point(550,550),Scalar(0,0,0));
    line(represent,Point(600,0),Point(600,560),Scalar(0,0,0));
    line(represent,Point(90,50),Point(650,50),Scalar(0,0,0));
    line(represent,Point(100,100),Point(650,100),Scalar(0,0,0));
    line(represent,Point(90,150),Point(650,150),Scalar(0,0,0));
    line(represent,Point(100,200),Point(650,200),Scalar(0,0,0));
    line(represent,Point(90,250),Point(650,250),Scalar(0,0,0));
    line(represent,Point(100,300),Point(650,300),Scalar(0,0,0));
    line(represent,Point(90,350),Point(650,350),Scalar(0,0,0));
    line(represent,Point(100,400),Point(650,400),Scalar(0,0,0));
    line(represent,Point(90,450),Point(650,450),Scalar(0,0,0));
    line(represent,Point(100,500),Point(650,500),Scalar(0,0,0));
    line(represent,Point(90,550),Point(650,550),Scalar(0,0,0),3);
    if(min_y==max_y){
        min_y=min_y-1;
        max_y=max_y+1;
    }
    if(min_x==max_x){
        min_x=min_x-1;
        max_x=max_x+1;
    }
    if(min_x<0 && max_x>0){
        int x_o=round(100+(500*(0-min_x)/(max_x-min_x)));
        line(represent,Point(x_o,0),Point(x_o,560),Scalar(0,0,0),2);
        putText(represent,"0",Point(x_o+2,570),1,1.2,Scalar(0,0,0),2);
    }
    if(min_y<0 && max_y>0){
        int y_o=600-round(50+(500*(0.0-min_y)/(max_y-min_y)));
        line(represent,Point(90,y_o),Point(650,y_o),Scalar(0,0,0),2);
        putText(represent,"0",Point(80,y_o-2),1,1.2,Scalar(0,0,0),2);
    }
    int precision=1;
    if(abs(max_y)<0.001 || abs(max_y-min_y)<0.001)
        precision=5;
    else if(abs(max_y)<0.01 || abs(max_y-min_y)<0.01)
        precision=4;
    else if(abs(max_y)<0.1 || abs(max_y-min_y)<0.1)
        precision=3;
    else if(abs(max_y)<1 || abs(max_y-min_y)<1)
        precision=2;
    stringstream ss;
    ss.precision(precision);
    ss<<fixed<<max_y;
    String valor=ss.str();
    putText(represent,valor,Point(10,50),1,1.2,Scalar(0,0,0),2);
    stringstream ss2;
    ss2.precision(precision);
    ss2<<fixed<<min_y+(8*(max_y-min_y)/10);
    String valor2=ss2.str();
    putText(represent,valor2,Point(10,150),1,1.2,Scalar(0,0,0),2);
    stringstream ss3;
    ss3.precision(precision);
    ss3<<fixed<<min_y+(6*(max_y-min_y)/10);
    String valor3=ss3.str();
    putText(represent,valor3,Point(10,250),1,1.2,Scalar(0,0,0),2);
    stringstream ss4;
    ss4.precision(precision);
    ss4<<fixed<<min_y+(4*(max_y-min_y)/10);
    String valor4=ss4.str();
    putText(represent,valor4,Point(10,350),1,1.2,Scalar(0,0,0),2);
    stringstream ss5;
    ss5.precision(precision);
    ss5<<fixed<<min_y+(2*(max_y-min_y)/10);
    String valor5=ss5.str();
    putText(represent,valor5,Point(10,450),1,1.2,Scalar(0,0,0),2);
    stringstream ss6;
    ss6.precision(precision);
    ss6<<fixed<<min_y;
    String valor6=ss6.str();
    putText(represent,valor6,Point(10,550),1,1.2,Scalar(0,0,0),2);
    precision=1;
    precision=1;
    if(abs(max_x)<0.001 || abs(max_x-min_x)<0.001)
        precision=5;
    else if(abs(max_x)<0.01 || abs(max_x-min_x)<0.01)
        precision=4;
    else if(abs(max_x)<0.1 || abs(max_x-min_x)<0.1)
        precision=3;
    else if(abs(max_x)<1 || abs(max_x-min_x)<1)
        precision=2;
    stringstream ss7;
    ss7.precision(precision);
    ss7<<fixed<<min_x;
    String valor7=ss7.str();
    putText(represent,valor7,Point(70,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss8;
    ss8.precision(precision);
    ss8<<fixed<<min_x+(2*(max_x-min_x)/10);
    String valor8=ss8.str();
    putText(represent,valor8,Point(170,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss9;
    ss9.precision(precision);
    ss9<<fixed<<min_x+(4*(max_x-min_x)/10);
    String valor9=ss9.str();
    putText(represent,valor9,Point(270,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss10;
    ss10.precision(precision);
    ss10<<fixed<<min_x+(6*(max_x-min_x)/10);
    String valor10=ss10.str();
    putText(represent,valor10,Point(370,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss11;
    ss11.precision(precision);
    ss11<<fixed<<min_x+(8*(max_x-min_x)/10);
    String valor11=ss11.str();
    putText(represent,valor11,Point(470,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss12;
    ss12.precision(precision);
    ss12<<fixed<<max_x;
    String valor12=ss12.str();
    putText(represent,valor12,Point(570,595),1,1.2,Scalar(0,0,0),2);
    for(int i=0; i<dim1.rows; i++){
        if(labels[i]!=0){
            Point punto;
            punto.x=round(100+(500*(dim1.at<float>(i,0)-min_x)/(max_x-min_x)));
            punto.y=600-round(50+(500*(dim2.at<float>(i,0)-min_y)/(max_y-min_y)));
            if(punto.x>100 && punto.y<550){
                Scalar color;
                if(negativa){
                    if(labels[i]==-1)
                        color=Colores[0];
                    else
                        color=Colores[(int)labels[i]];
                }
                else
                    color=Colores[(int)labels[i]-1];
                circle(represent, punto,5,color,1);
            }
        }
    }
    for(uint i=0; i<Elipses.size(); i++){
        Point punto;
        punto.x=round(100+(500*(Elipses[i].media_x-min_x)/(max_x-min_x)));
        punto.y=600-round(50+(500*(Elipses[i].media_y-min_y)/(max_y-min_y)));
        if(punto.x>100 && punto.y<550){
            Scalar color;
            color=Colores[i];
            circle(represent, punto,5,color,-1);
            float grande,pequeno;
            grande=((Elipses[i].big)/(max_total-min_total))*1518;
            pequeno=((Elipses[i].small)/(max_total-min_total))*1518;
            cv::ellipse(represent,punto,Size(grande,pequeno),Elipses[i].angle-90,0,360,color,2);
            grande=((Elipses[i].big)/(max_total-min_total))*1224;
            pequeno=((Elipses[i].small)/(max_total-min_total))*1224;
            cv::ellipse(represent,punto,Size(grande,pequeno),Elipses[i].angle-90,0,360,color,2);
            grande=((Elipses[i].big)/(max_total-min_total))*589;
            pequeno=((Elipses[i].small)/(max_total-min_total))*589;
            cv::ellipse(represent,punto,Size(grande,pequeno),Elipses[i].angle-90,0,360,color,2);
        }
    }
#ifdef GUI
    QApplication::restoreOverrideCursor();
#endif
    Mat most=Mat::zeros(50+(50*num_etiq),200,CV_8UC3);
    most=most+Scalar(255,255,255);
    for(int i=0; i<num_etiq; i++){
        if(i==0 && negativa){
            String texto;
            texto="Etiqueta -1";
            putText(most,texto,Point(10,50*(i+1)),1,1.5,Colores[i],2);
        }
        else if(negativa){
            stringstream tex;
            tex<<"Etiqueta "<<i;
            putText(most,tex.str(),Point(10,50*(i+1)),1,1.5,Colores[i],2);
        }
        else{
            stringstream tex;
            tex<<"Etiqueta "<<i+1;
            putText(most,tex.str(),Point(10,50*(i+1)),1,1.5,Colores[i],2);
        }
    }
    imshow("Leyenda",most);
    imshow(nombre,represent);
    waitKey(0);
    return 0;
}

int MLT::Representacion::Continuous_data_represent(string nombre, Mat Data, vector<float> labels, vector<cv::Scalar> Colores){
    for(uint i=0; i<labels.size(); i++){
        if(labels[i]==0){
            cout<<"ERROR en Continuous_data_represent: Etiquetas con valor 0"<<endl;
            return 1;
        }
    }
    if(labels.size()<2){
        cout<<"ERROR en Data_represent: No hay datos o sólo hay uno"<<endl;
        return 1;
    }
    Mat Datos;
    Data.copyTo(Datos);
    bool negativa;
    Auxiliares ax;
    int num_etiq=ax.numero_etiquetas(labels,negativa);
    if((uint)num_etiq>Colores.size()){
        cout<<"ERROR en Continuous_data_represent: El número de etiquetas es mayor que el numero de colores"<<endl;
        return 1;
    }
    Mat represent=Mat::zeros(600,650,CV_8UC3);
    represent=represent+Scalar(255,255,255);
    line(represent,Point(100,0),Point(100,560),Scalar(0,0,0),3);
    line(represent,Point(150,0),Point(150,550),Scalar(0,0,0));
    line(represent,Point(200,0),Point(200,560),Scalar(0,0,0));
    line(represent,Point(250,0),Point(250,550),Scalar(0,0,0));
    line(represent,Point(300,0),Point(300,560),Scalar(0,0,0));
    line(represent,Point(350,0),Point(350,550),Scalar(0,0,0));
    line(represent,Point(400,0),Point(400,560),Scalar(0,0,0));
    line(represent,Point(450,0),Point(450,550),Scalar(0,0,0));
    line(represent,Point(500,0),Point(500,560),Scalar(0,0,0));
    line(represent,Point(550,0),Point(550,550),Scalar(0,0,0));
    line(represent,Point(600,0),Point(600,560),Scalar(0,0,0));
    line(represent,Point(90,50),Point(650,50),Scalar(0,0,0));
    line(represent,Point(100,100),Point(650,100),Scalar(0,0,0));
    line(represent,Point(90,150),Point(650,150),Scalar(0,0,0));
    line(represent,Point(100,200),Point(650,200),Scalar(0,0,0));
    line(represent,Point(90,250),Point(650,250),Scalar(0,0,0));
    line(represent,Point(100,300),Point(650,300),Scalar(0,0,0));
    line(represent,Point(90,350),Point(650,350),Scalar(0,0,0));
    line(represent,Point(100,400),Point(650,400),Scalar(0,0,0));
    line(represent,Point(90,450),Point(650,450),Scalar(0,0,0));
    line(represent,Point(100,500),Point(650,500),Scalar(0,0,0));
    line(represent,Point(90,550),Point(650,550),Scalar(0,0,0),3);
    if(num_etiq==1){
    vector<Point2f> puntos;
        for(int i=0; i<Datos.rows; i++){
            Point2f punto;
            punto.x=Datos.at<float>(i,0);
            punto.y=Datos.at<float>(i,1);
            puntos.push_back(punto);
        }
        sort(puntos.begin(), puntos.end(), ord_x);
        for(uint i=0; i<puntos.size(); i++){
            Datos.at<float>(i,0)=puntos[i].x;
            Datos.at<float>(i,1)=puntos[i].y;
        }
    }
    if(Datos.cols==2){
        Mat dim1=Datos.col(0);
        Mat dim2=Datos.col(1);
        float max_x=0,max_y=0;
        float min_x=999999999,min_y=999999999;
        for(int i=0; i<dim1.rows; i++){
            if(dim1.at<float>(i,0)>max_x)
                max_x=dim1.at<float>(i,0);
            if(dim1.at<float>(i,0)<min_x)
                min_x=dim1.at<float>(i,0);
            if(dim2.at<float>(i,0)>max_y)
                max_y=dim2.at<float>(i,0);
            if(dim2.at<float>(i,0)<min_y)
                min_y=dim2.at<float>(i,0);
        }
        min_y=min_y-((max_y-min_y)/20);
        min_x=min_x-((max_x-min_x)/20);
        if(min_y==max_y){
            min_y=min_y-1;
            max_y=max_y+1;
        }
        if(min_x==max_x){
            min_x=min_x-1;
            max_x=max_x+1;
        }
        if(min_x<0 && max_x>0){
            int x_o=round(100+(500*(0-min_x)/(max_x-min_x)));
            line(represent,Point(x_o,0),Point(x_o,560),Scalar(0,0,0),2);
            putText(represent,"0",Point(x_o+2,570),1,1.2,Scalar(0,0,0),2);
        }
        if(min_y<0 && max_y>0){
            int y_o=600-round(50+(500*(0.0-min_y)/(max_y-min_y)));
            line(represent,Point(90,y_o),Point(650,y_o),Scalar(0,0,0),2);
            putText(represent,"0",Point(80,y_o-2),1,1.2,Scalar(0,0,0),2);
        }
        int precision=1;
        if(abs(max_y)<0.001 || abs(max_y-min_y)<0.001)
            precision=5;
        else if(abs(max_y)<0.01 || abs(max_y-min_y)<0.01)
            precision=4;
        else if(abs(max_y)<0.1 || abs(max_y-min_y)<0.1)
            precision=3;
        else if(abs(max_y)<1 || abs(max_y-min_y)<1)
            precision=2;
        stringstream ss;
        ss.precision(precision);
        ss<<fixed<<max_y;
        String valor=ss.str();
        putText(represent,valor,Point(10,50),1,1.2,Scalar(0,0,0),2);
        stringstream ss2;
        ss2.precision(precision);
        ss2<<fixed<<min_y+(8*(max_y-min_y)/10);
        String valor2=ss2.str();
        putText(represent,valor2,Point(10,150),1,1.2,Scalar(0,0,0),2);
        stringstream ss3;
        ss3.precision(precision);
        ss3<<fixed<<min_y+(6*(max_y-min_y)/10);
        String valor3=ss3.str();
        putText(represent,valor3,Point(10,250),1,1.2,Scalar(0,0,0),2);
        stringstream ss4;
        ss4.precision(precision);
        ss4<<fixed<<min_y+(4*(max_y-min_y)/10);
        String valor4=ss4.str();
        putText(represent,valor4,Point(10,350),1,1.2,Scalar(0,0,0),2);
        stringstream ss5;
        ss5.precision(precision);
        ss5<<fixed<<min_y+(2*(max_y-min_y)/10);
        String valor5=ss5.str();
        putText(represent,valor5,Point(10,450),1,1.2,Scalar(0,0,0),2);
        stringstream ss6;
        ss6.precision(precision);
        ss6<<fixed<<min_y;
        String valor6=ss6.str();
        putText(represent,valor6,Point(10,550),1,1.2,Scalar(0,0,0),2);
        precision=1;
        if(abs(max_x)<0.001 || abs(max_x-min_x)<0.001)
            precision=5;
        else if(abs(max_x)<0.01 || abs(max_x-min_x)<0.01)
            precision=4;
        else if(abs(max_x)<0.1 || abs(max_x-min_x)<0.1)
            precision=3;
        else if(abs(max_x)<1 || abs(max_x-min_x)<1)
            precision=2;
        stringstream ss7;
        ss7.precision(precision);
        ss7<<fixed<<min_x;
        String valor7=ss7.str();
        putText(represent,valor7,Point(70,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss8;
        ss8.precision(precision);
        ss8<<fixed<<min_x+(2*(max_x-min_x)/10);
        String valor8=ss8.str();
        putText(represent,valor8,Point(170,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss9;
        ss9.precision(precision);
        ss9<<fixed<<min_x+(4*(max_x-min_x)/10);
        String valor9=ss9.str();
        putText(represent,valor9,Point(270,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss10;
        ss10.precision(precision);
        ss10<<fixed<<min_x+(6*(max_x-min_x)/10);
        String valor10=ss10.str();
        putText(represent,valor10,Point(370,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss11;
        ss11.precision(precision);
        ss11<<fixed<<min_x+(8*(max_x-min_x)/10);
        String valor11=ss11.str();
        putText(represent,valor11,Point(470,595),1,1.2,Scalar(0,0,0),2);
        stringstream ss12;
        ss12.precision(precision);
        ss12<<fixed<<max_x;
        String valor12=ss12.str();
        putText(represent,valor12,Point(570,595),1,1.2,Scalar(0,0,0),2);
        vector<vector<Point> > Puntos(num_etiq);
        for(int i=0; i<dim1.rows; i++){
            if(labels[i]!=0){
                Point punto;
                punto.x=round(100+(500*(dim1.at<float>(i,0)-min_x)/(max_x-min_x)));
                punto.y=600-round(50+(500*(dim2.at<float>(i,0)-min_y)/(max_y-min_y)));
                if(punto.x>100 && punto.y<550){
                    Scalar color;
                    if(negativa){
                        if(labels[i]==-1){
                            color=Colores[0];
                            Puntos[0].push_back(punto);
                        }
                        else{
                            color=Colores[(int)labels[i]];
                            Puntos[(int)labels[i]].push_back(punto);
                        }
                    }
                    else{
                        color=Colores[(int)labels[i]-1];
                        Puntos[(int)labels[i]-1].push_back(punto);
                    }
                    circle(represent, punto,5,color,1);
                }
            }
        }
        for(uint i=0; i< Puntos.size(); i++){
            for(uint j=1; j<Puntos[i].size(); j++){
                line(represent,Puntos[i][j-1],Puntos[i][j],Colores[i],3);
            }
        }
    }
    else{
        cout<<"ERROR en Continuous_data_represent: Numero de dimensiones distinto de dos"<<endl;
        return 1;
    }
#ifdef GUI
    QApplication::restoreOverrideCursor();
#endif
    imshow(nombre,represent);
    waitKey(0);
    return 0;
}

int MLT::Representacion::Histogram_represent(string nombre, vector<vector<Mat> > Histograma, vector<cv::Scalar> Colores, int dimension){
    if(Histograma.size()==0){
        cout<<"ERROR en Histogram_represent: Histograma vacio"<<endl;
        return 1;
    }
    if(Histograma[0].size()==0){
        cout<<"ERROR en Histogram_represent: El numero de dimensiones del histograma es cero"<<endl;
        return 1;
    }
    if(Histograma.size()>Colores.size()){
        cout<<"ERROR en Histogram_represent: El número de etiquetas es mayor que el numero de colores"<<endl;
        return 1;
    }
    if((uint)dimension>Histograma[0].size()){
        cout<<"ERROR en Histogram_represent: Numero de dimension mayor al del histograma"<<endl;
        return 1;
    }
    if(dimension<1){
        cout<<"ERROR en Histogram_represent: Numero de dimension debe ser mayor a cero"<<endl;
        return 1;
    }
    Mat represent=Mat::zeros(600,650,CV_8UC3);
    represent=represent+Scalar(255,255,255);
    line(represent,Point(100,0),Point(100,560),Scalar(0,0,0),3);
    line(represent,Point(150,0),Point(150,550),Scalar(0,0,0));
    line(represent,Point(200,0),Point(200,560),Scalar(0,0,0));
    line(represent,Point(250,0),Point(250,550),Scalar(0,0,0));
    line(represent,Point(300,0),Point(300,560),Scalar(0,0,0));
    line(represent,Point(350,0),Point(350,550),Scalar(0,0,0));
    line(represent,Point(400,0),Point(400,560),Scalar(0,0,0));
    line(represent,Point(450,0),Point(450,550),Scalar(0,0,0));
    line(represent,Point(500,0),Point(500,560),Scalar(0,0,0));
    line(represent,Point(550,0),Point(550,550),Scalar(0,0,0));
    line(represent,Point(600,0),Point(600,560),Scalar(0,0,0));
    line(represent,Point(90,50),Point(650,50),Scalar(0,0,0));
    line(represent,Point(100,100),Point(650,100),Scalar(0,0,0));
    line(represent,Point(90,150),Point(650,150),Scalar(0,0,0));
    line(represent,Point(100,200),Point(650,200),Scalar(0,0,0));
    line(represent,Point(90,250),Point(650,250),Scalar(0,0,0));
    line(represent,Point(100,300),Point(650,300),Scalar(0,0,0));
    line(represent,Point(90,350),Point(650,350),Scalar(0,0,0));
    line(represent,Point(100,400),Point(650,400),Scalar(0,0,0));
    line(represent,Point(90,450),Point(650,450),Scalar(0,0,0));
    line(represent,Point(100,500),Point(650,500),Scalar(0,0,0));
    line(represent,Point(90,550),Point(650,550),Scalar(0,0,0),3);
    float max_x=0,max_y=0;
    float min_x=999999999,min_y=999999999;
    for(uint i=0; i<Histograma.size(); i++){
        for(int j=0; j<Histograma[i][dimension-1].cols; j++){
            if(Histograma[i][dimension-1].at<float>(0,j)>max_x)
                max_x=Histograma[i][dimension-1].at<float>(0,j);
            if(Histograma[i][dimension-1].at<float>(0,j)<min_x)
                min_x=Histograma[i][dimension-1].at<float>(0,j);
            if(Histograma[i][dimension-1].at<float>(1,j)>max_y)
                max_y=Histograma[i][dimension-1].at<float>(1,j);
            if(Histograma[i][dimension-1].at<float>(1,j)<min_y)
                min_y=Histograma[i][dimension-1].at<float>(1,j);
        }
    }
    min_y=min_y-((max_y-min_y)/20);
    min_x=min_x-((max_x-min_x)/20);
    if(min_y==max_y){
        min_y=min_y-1;
        max_y=max_y+1;
    }
    if(min_x==max_x){
        min_x=min_x-1;
        max_x=max_x+1;
    }
    if(min_x<0 && max_x>0){
        int x_o=round(100+(500*(0-min_x)/(max_x-min_x)));
        line(represent,Point(x_o,0),Point(x_o,560),Scalar(0,0,0),2);
        putText(represent,"0",Point(x_o+2,570),1,1.2,Scalar(0,0,0),2);
    }
    if(min_y<0 && max_y>0){
        int y_o=600-round(50+(500*(0.0-min_y)/(max_y-min_y)));
        line(represent,Point(90,y_o),Point(650,y_o),Scalar(0,0,0),2);
        putText(represent,"0",Point(80,y_o-2),1,1.2,Scalar(0,0,0),2);
    }
    int precision=1;
    if(abs(max_y)<0.001 || abs(max_y-min_y)<0.001)
        precision=5;
    else if(abs(max_y)<0.01 || abs(max_y-min_y)<0.01)
        precision=4;
    else if(abs(max_y)<0.1 || abs(max_y-min_y)<0.1)
        precision=3;
    else if(abs(max_y)<1 || abs(max_y-min_y)<1)
        precision=2;
    stringstream ss;
    ss.precision(precision);
    ss<<fixed<<max_y;
    String valor=ss.str();
    putText(represent,valor,Point(10,50),1,1.2,Scalar(0,0,0),2);
    stringstream ss2;
    ss2.precision(precision);
    ss2<<fixed<<min_y+(8*(max_y-min_y)/10);
    String valor2=ss2.str();
    putText(represent,valor2,Point(10,150),1,1.2,Scalar(0,0,0),2);
    stringstream ss3;
    ss3.precision(precision);
    ss3<<fixed<<min_y+(6*(max_y-min_y)/10);
    String valor3=ss3.str();
    putText(represent,valor3,Point(10,250),1,1.2,Scalar(0,0,0),2);
    stringstream ss4;
    ss4.precision(precision);
    ss4<<fixed<<min_y+(4*(max_y-min_y)/10);
    String valor4=ss4.str();
    putText(represent,valor4,Point(10,350),1,1.2,Scalar(0,0,0),2);
    stringstream ss5;
    ss5.precision(precision);
    ss5<<fixed<<min_y+(2*(max_y-min_y)/10);
    String valor5=ss5.str();
    putText(represent,valor5,Point(10,450),1,1.2,Scalar(0,0,0),2);
    stringstream ss6;
    ss6.precision(precision);
    ss6<<fixed<<min_y;
    String valor6=ss6.str();
    putText(represent,valor6,Point(10,550),1,1.2,Scalar(0,0,0),2);
    precision=1;
    if(abs(max_x)<0.001 || abs(max_x-min_x)<0.001)
        precision=5;
    else if(abs(max_x)<0.01 || abs(max_x-min_x)<0.01)
        precision=4;
    else if(abs(max_x)<0.1 || abs(max_x-min_x)<0.1)
        precision=3;
    else if(abs(max_x)<1 || abs(max_x-min_x)<1)
        precision=2;
    stringstream ss7;
    ss7.precision(precision);
    ss7<<fixed<<min_x;
    String valor7=ss7.str();
    putText(represent,valor7,Point(70,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss8;
    ss8.precision(precision);
    ss8<<fixed<<min_x+(2*(max_x-min_x)/10);
    String valor8=ss8.str();
    putText(represent,valor8,Point(170,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss9;
    ss9.precision(precision);
    ss9<<fixed<<min_x+(4*(max_x-min_x)/10);
    String valor9=ss9.str();
    putText(represent,valor9,Point(270,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss10;
    ss10.precision(precision);
    ss10<<fixed<<min_x+(6*(max_x-min_x)/10);
    String valor10=ss10.str();
    putText(represent,valor10,Point(370,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss11;
    ss11.precision(precision);
    ss11<<fixed<<min_x+(8*(max_x-min_x)/10);
    String valor11=ss11.str();
    putText(represent,valor11,Point(470,595),1,1.2,Scalar(0,0,0),2);
    stringstream ss12;
    ss12.precision(precision);
    ss12<<fixed<<max_x;
    String valor12=ss12.str();
    putText(represent,valor12,Point(570,595),1,1.2,Scalar(0,0,0),2);
    for(uint i=0; i<Histograma.size(); i++){
//        Point punto_anterior;
        for(int j=0; j<Histograma[i][dimension-1].cols; j++){
            int tam_barra=(500-(5*Histograma[i][dimension-1].cols)*Histograma.size())/(Histograma[i][dimension-1].cols*Histograma.size());
            if(tam_barra<1){
                cout<<"ERROR en Histogram_represent: El tamaño de las barras es muy pequeño debido a un numero elevado de barras. No se puede representar"<<endl;
                return 1;
            }
            Point punto;
            punto.x=round(100+(500*(Histograma[i][dimension-1].at<float>(0,j)-min_x)/(max_x-min_x)));
            punto.y=600-round(50+(500*(Histograma[i][dimension-1].at<float>(1,j)-min_y)/(max_y-min_y)));
            if(punto.x>100 && punto.y<550){
                Rect rectangulo=Rect(punto.x-floor(tam_barra/2),punto.y,tam_barra,550-punto.y);
                rectangle(represent,rectangulo,Colores[i],-1);
            }
//            circle(represent,punto,floor(tam_barra/2),Colores[i],-1);
//            if(j>0){
//                line(represent,punto_anterior,punto,Colores[i],2,-1);
//            }
//            punto_anterior=punto;
        }
    }
#ifdef GUI
    QApplication::restoreOverrideCursor();
#endif
    Mat most=Mat::zeros(50+(50*Histograma.size()),200,CV_8UC3);
    most=most+Scalar(255,255,255);
    for(uint i=0; i<Histograma.size(); i++){
        stringstream tex;
        tex<<"Etiqueta "<<i+1;
        putText(most,tex.str(),Point(10,50*(i+1)),1,1.5,Colores[i],2);
    }
    imshow("Leyenda",most);
    imshow(nombre,represent);
    waitKey(0);
    return 0;
}

int MLT::Representacion::Imagen(vector<Mat> Imagenes, int numero){
    string nombre="IMAGEN";
    double minimo=999999999,maximo=0;
    double minval, maxval;
    for(uint i=0; i<Imagenes.size(); i++){
        cv::minMaxLoc(Imagenes[i],&minval,&maxval);
        if(minval<minimo)
            minimo=minval;
        if(maxval>maximo)
            maximo=maxval;
    }
    if(numero>(int)Imagenes.size()-1 || numero<0){
        cout<<"ERROR en Imagen: El numero esta fuera del rango de las Imagenes"<<endl;
        return 1;
    }
    while(true){
        if(numero>(int)Imagenes.size()-1)
            numero=(int)Imagenes.size()-1;
        if(numero<0)
            numero=0;
        Mat Mostrar=Mat::zeros(Imagenes[numero].rows,Imagenes[numero].cols,CV_32FC1);
        Mostrar=((Imagenes[numero]-minimo)/(maximo-minimo));
        Muestra:
        imshow(nombre,Mostrar);
        char z=waitKey(1);
        int a=z;
        if(z=='i'){
            Mat most=Mat::zeros(200,200,CV_8UC3);
            most=most+Scalar(255,255,255);
            String texto="s=Siguiente";
            putText(most,texto,Point(10,50),1,1.5,Scalar(0,0,255),2);
            texto="a=Anterior";
            putText(most,texto,Point(10,100),1,1.5,Scalar(0,0,255),2);
            texto="ESC=Salir";
            putText(most,texto,Point(10,150),1,1.5,Scalar(0,0,255),2);
            imshow("Info",most);
            waitKey(1500);
            destroyWindow("Info");
            goto Muestra;
        }
        else if(z=='s')
            numero++;
        else if(z=='a')
            numero--;
        else if(a==27){
            cv::destroyWindow(nombre);
            break;
        }
    }
    return 0;
}
