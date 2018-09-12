#include "busqueda.h"

MLT::Busqueda::Busqueda(Clasificador *clasificador, int Tipo_Descriptor, Descriptor *descriptor, MultiClasificador::Multi_type *multitipo){

    this->clasificador=clasificador;
    numero_etiquetas=clasificador->numero_etiquetas;
    ventana_x=clasificador->ventana_o_x;
    ventana_y=clasificador->ventana_o_y;
    tipo_dato=Tipo_Descriptor;
    tipo=clasificador->tipo_clasificador;
    descrip=descriptor;
}

MLT::Busqueda::Busqueda(MultiClasificador *clasificador, int Tipo_Descriptor, Descriptor *descriptor, MultiClasificador::Multi_type *multitipo){

    Multi=clasificador;
    numero_etiquetas=Multi->numero_etiquetas;
    ventana_x=Multi->ventana_x;
    ventana_y=Multi->ventana_y;
    Tipo_Multi.tipo=multitipo->tipo;
    Tipo_Multi.tipo_regla=multitipo->tipo_regla;
    Tipo_Multi.label_ref=multitipo->label_ref;
    Tipo_Multi.w_clasif=multitipo->w_clasif;

    tipo=MULTICLASIFICADOR;
    tipo_dato=Tipo_Descriptor;
    descrip=descriptor;
}

int MLT::Busqueda::Textura(Mat src, Size tam_base, int escalas, int salto, int rotate, bool relleno, Mat &OUT){
    if(src.empty()){
        cout<<"ERROR en Textura: Imagen vacia"<<endl;
        this->error=1;
        return this->error;
    }
    if(tam_base.height<2 || tam_base.width<2){
        cout<<"ERROR en Textura: tam_base debe ser por lo menos de 2x2"<<endl;
        this->error=1;
        return this->error;
    }
    if(salto<1){
        cout<<"ERROR en Textura: salto debe ser como mínimo 1"<<endl;
        this->error=1;
        return this->error;
    }
    if(rotate<0){
        cout<<"ERROR en Textura: rotate debe ser como mínimo 0"<<endl;
        this->error=1;
        return this->error;
    }
    if(tipo==MULTICLASIFICADOR){
        if(tipo_dato!=Multi->tipo_dato){
            cout<<"ERROR en Textura: El clasificador no se entrenó con este tipo de dato"<<endl;
            this->error=1;
            return this->error;
        }
    }
    else{
        if(tipo_dato!=clasificador->tipo_dato){
            cout<<"ERROR en Textura: El clasificador no se entrenó con este tipo de dato"<<endl;
            this->error=1;
            return this->error;
        }
    }
    int e=0;
    OUT=Mat::zeros(src.rows,src.cols,CV_32FC1);
    int escalas_down_up=floor(escalas/2);
    int vent_med_x=floor(tam_base.width/2);
    int vent_med_y=floor(tam_base.height/2);
    int tam_x=src.cols;
    int tam_y=src.rows;
    Mat imagen;
    src.copyTo(imagen);
    bool negativa=false;
    bool rot=true;
    Mat Rotada;
    int rotacion=0;
    while(rotacion<360 && rot){
        if(rotate==0)
            rot=false;
        Mat transform=cv::getRotationMatrix2D(Point(imagen.cols/2,imagen.rows/2),rotacion,1.0);
        Mat transform_inv;
        cv::invertAffineTransform(transform,transform_inv);
        transform_inv.convertTo(transform_inv,CV_32F);
        cv::warpAffine(imagen,Rotada,transform,Size(imagen.cols,imagen.rows));
        Mat up,down;
        Rotada.copyTo(up);
        Rotada.copyTo(down);
        for (int x=vent_med_x; x<tam_x-vent_med_x; x=x+salto){
            for(int y=vent_med_y; y<tam_y-vent_med_y; y=y+salto){
                Mat ROI_g(Rotada,cv::Rect(x-vent_med_x,y-vent_med_y,tam_base.width,tam_base.height));
                Mat ROI;
                cv::resize(ROI_g,ROI,Size(ventana_x,ventana_y));
                if(ROI.empty()){
                    cout<<"ERROR en Textura: ROI con tamaño 0"<<endl;
                    this->error=1;
                    return this->error;
                }
                std::vector<cv::Mat> Imagen;
                if(tipo_dato==RGB){
                    Imagen.push_back(ROI);
                }
                else {
                    vector<Mat> img,des;
                    img.push_back(ROI);
                    descrip->Extract(img,des);
                    Imagen.push_back(des[0]);
                }
                vector<float> Label;
                if(tipo==MULTICLASIFICADOR){
                    if(Tipo_Multi.tipo==CASCADA)
                        e=Multi->Cascada(Imagen,Tipo_Multi.tipo_regla,Tipo_Multi.label_ref,Label);
                    else if(Tipo_Multi.tipo==VOTACION)
                        e=Multi->Votacion(Imagen,Tipo_Multi.w_clasif,Label);
                    else{
                        cout<<"ERROR en Textura: Tipo_Multi con valor erroneo debe ser CASCADA O VOTACION"<<endl;
                        this->error=1;
                        return this->error;
                    }
                }
                else
                    e=clasificador->Autoclasificacion(Imagen,Label,true,false);
                if(e==1){
                    cout<<"ERROR en Textura: Error en Autoclasificacion"<<endl;
                    this->error=1;
                    return this->error;
                }
                if(Label[0]==-1.0)
                    negativa=true;
                Mat punto_rotado(3,1,CV_32FC1);
                punto_rotado.at<float>(0,0)=x;
                punto_rotado.at<float>(1,0)=y;
                punto_rotado.at<float>(2,0)=1;
                Mat punto_desrotado=transform_inv*punto_rotado;
                int x2=punto_desrotado.at<float>(0,0);
                int y2=punto_desrotado.at<float>(1,0);
                if(x2>0 && y2>0 && x2<OUT.cols && y2<OUT.rows)
                    OUT.at<float>(y2,x2)=Label[0];
            }
        }
        for(int k=1; k<=escalas_down_up; k++){
            Mat imagen_up;
            pyrUp( up, imagen_up, Size( up.cols*2, up.rows*2));
            int tam_x_up=imagen_up.cols;
            int tam_y_up=imagen_up.rows;
            for (int x=vent_med_x; x<tam_x_up-vent_med_x; x=x+salto){
                for(int y=vent_med_y; y<tam_y_up-vent_med_y; y=y+salto){
                    Mat ROI_up_g(imagen_up,cv::Rect(x-vent_med_x,y-vent_med_y,tam_base.width,tam_base.height));
                    Mat ROI_up;
                    cv::resize(ROI_up_g,ROI_up,Size(ventana_x,ventana_y));
                    if(ROI_up.empty()){
                        cout<<"ERROR en Textura: ROI con tamaño 0"<<endl;
                        this->error=1;
                        return this->error;
                    }
                    std::vector<cv::Mat> Imagen;
                    if(tipo_dato==RGB){
                        Imagen.push_back(ROI_up);
                    }
                    else {
                        vector<Mat> img,des;
                        img.push_back(ROI_up);
                        descrip->Extract(img,des);
                        Imagen.push_back(des[0]);
                    }
                    vector<float> Label;
                    if(tipo==MULTICLASIFICADOR){
                        if(Tipo_Multi.tipo==CASCADA)
                            e=Multi->Cascada(Imagen,Tipo_Multi.tipo_regla,Tipo_Multi.label_ref,Label);
                        else if(Tipo_Multi.tipo==VOTACION)
                            e=Multi->Votacion(Imagen,Tipo_Multi.w_clasif,Label);
                        else{
                            cout<<"ERROR en Textura: Tipo_Multi con valor erroneo debe ser CASCADA O VOTACION"<<endl;
                            this->error=1;
                            return this->error;
                        }
                    }
                    else
                        e=clasificador->Autoclasificacion(Imagen,Label,true,false);
                    if(e==1){
                        cout<<"ERROR en Textura: Error en Autoclasificacion"<<endl;
                        this->error=1;
                        return this->error;
                    }
                    if(Label[0]==-1.0)
                        negativa=true;
                    Mat punto_rotado(3,1,CV_32FC1);
                    punto_rotado.at<float>(0,0)=x/(2*k);
                    punto_rotado.at<float>(1,0)=y/(2*k);
                    punto_rotado.at<float>(2,0)=1;
                    Mat punto_desrotado=transform_inv*punto_rotado;
                    int x2=punto_desrotado.at<float>(0,0);
                    int y2=punto_desrotado.at<float>(1,0);
                    if(x2>0 && y2>0 && x2<OUT.cols && y2<OUT.rows)
                        if(OUT.at<float>(y2,x2)==0)
                            OUT.at<float>(y2,x2)=Label[0];
                }
            }
            imagen_up.copyTo(up);
            Mat imagen_down;
            if(down.cols/2>tam_base.width && down.rows/2>tam_base.height){
                pyrDown( down, imagen_down, Size( down.cols/2, down.rows/2));
                int tam_x_down=imagen_down.cols;
                int tam_y_down=imagen_down.rows;
                for (int x=vent_med_x; x<tam_x_down-vent_med_x; x=x+salto){
                    for(int y=vent_med_y; y<tam_y_down-vent_med_y; y=y+salto){
                        if(tam_base.width<imagen_down.rows && tam_base.height<imagen_down.cols){
                            Mat ROI_down_g(imagen_down,cv::Rect(x-vent_med_x,y-vent_med_y,tam_base.width,tam_base.height));
                            Mat ROI_down;
                            cv::resize(ROI_down_g,ROI_down,Size(ventana_x,ventana_y));
                            if(ROI_down.empty()){
                                cout<<"ERROR en Textura: ROI con tamaño 0"<<endl;
                                this->error=1;
                                return this->error;
                            }
                            std::vector<cv::Mat> Imagen;
                            if(tipo_dato==RGB){
                                Imagen.push_back(ROI_down);
                            }
                            else {
                                vector<Mat> img,des;
                                img.push_back(ROI_down);
                                descrip->Extract(img,des);
                                Imagen.push_back(des[0]);
                            }
                            vector<float> Label;
                            if(tipo==MULTICLASIFICADOR){
                                if(Tipo_Multi.tipo==CASCADA)
                                    e=Multi->Cascada(Imagen,Tipo_Multi.tipo_regla,Tipo_Multi.label_ref,Label);
                                else if(Tipo_Multi.tipo==VOTACION)
                                    e=Multi->Votacion(Imagen,Tipo_Multi.w_clasif,Label);
                                else{
                                    cout<<"ERROR en Textura: Tipo_Multi con valor erroneo debe ser CASCADA O VOTACION"<<endl;
                                    this->error=1;
                                    return this->error;
                                }
                            }
                            else
                                e=clasificador->Autoclasificacion(Imagen,Label,true,false);
                            if(e==1){
                                cout<<"ERROR en Textura: Error en Autoclasificacion"<<endl;
                                this->error=1;
                                return this->error;
                            }
                            if(Label[0]==-1.0)
                                negativa=true;
                            Mat punto_rotado(3,1,CV_32FC1);
                            punto_rotado.at<float>(0,0)=x*(2*k);
                            punto_rotado.at<float>(1,0)=y*(2*k);
                            punto_rotado.at<float>(2,0)=1;
                            Mat punto_desrotado=transform_inv*punto_rotado;
                            int x2=punto_desrotado.at<float>(0,0);
                            int y2=punto_desrotado.at<float>(1,0);
                            if(x2>0 && y2>0 && x2<OUT.cols && y2<OUT.rows)
                                if(OUT.at<float>(y2,x2)==0)
                                    OUT.at<float>(y2,x2)=Label[0];
                        }
                    }
                }
                imagen_down.copyTo(down);
            }
        }
        rotacion=rotacion+rotate;
    }
    if(relleno){
#ifdef WARNINGS
        if(tam_base.width<salto || tam_base.height<salto)
            cout<<"Warning en Buscar Textura: El tamaño de la ventana es menor que el salto, el objeto no saldrá cerrado"<<endl;
#endif
        Mat OUT_s;
        OUT_s=Mat::zeros(OUT.rows,OUT.cols,CV_32FC1);
        Mat Kernel_dilate = Mat::ones(tam_base.height/(escalas_down_up+1),tam_base.width/(escalas_down_up+1),CV_32FC1);
        Mat Kernel_erode = Mat::ones(1+(tam_base.height/(escalas_down_up+1)),1+(tam_base.width/(escalas_down_up+1)),CV_32FC1);
        if(negativa==true){
            for(int k=-1; k<numero_etiquetas; k++){
                Mat OUT_label=Mat::zeros(OUT.rows,OUT.cols,CV_32FC1);
                if(k!=0){
                    for(int x=0; x<OUT.cols; x++){
                        for(int y=0; y<OUT.rows; y++){
                            if(OUT.at<float>(y,x)==k)
                                OUT_label.at<float>(y,x)=1.0;
                            else
                                OUT_label.at<float>(y,x)=0.0;
                        }
                    }
                    cv::dilate(OUT_label,OUT_label,Kernel_dilate);
                    cv::erode(OUT_label,OUT_label,Kernel_erode);
                    OUT_s=OUT_s+(k*OUT_label);
                }
            }
        }
        else{
            for(int k=1; k<=numero_etiquetas; k++){
                Mat OUT_label=Mat::zeros(OUT.rows,OUT.cols,CV_32FC1);
                for(int x=0; x<OUT.cols; x++){
                    for(int y=0; y<OUT.rows; y++){
                        if(OUT.at<float>(y,x)==k)
                            OUT_label.at<float>(y,x)=1;
                        else
                            OUT_label.at<float>(y,x)=0;
                    }
                }
                cv::dilate(OUT_label,OUT_label,Kernel_dilate);
                cv::erode(OUT_label,OUT_label,Kernel_erode);
                OUT_s=OUT_s+(k*OUT_label);
            }
        }
        OUT_s.copyTo(OUT);
    }
    this->error=0;
    return this->error;
}

int MLT::Busqueda::Posicion(Mat src, Size tam_base, int escalas, int salto, int rotate, bool juntar_recuadros, bool solapamiento, bool aislamiento, float distancia_recuadros, int rotacion_recuadros, vector<RotatedRect> &recuadros, vector<float> &Labels){
    if(src.empty()){
        cout<<"ERROR en Posicion: Imagen vacia"<<endl;
        this->error=1;
        return this->error;
    }
    if(tam_base.height<2 || tam_base.width<2){
        cout<<"ERROR en Posicion: tam_base debe ser por lo menos de 2x2"<<endl;
        this->error=1;
        return this->error;
    }
    if(salto<1){
        cout<<"ERROR en Posicion: salto debe ser como mínimo 1"<<endl;
        this->error=1;
        return this->error;
    }
    if(rotate<0){
        cout<<"ERROR en Posicion: rotate debe ser como mínimo 0"<<endl;
        this->error=1;
        return this->error;
    }
    if(tipo==MULTICLASIFICADOR){
        if(tipo_dato!=Multi->tipo_dato){
            cout<<"ERROR en Posicion: El clasificador no se entrenó con este tipo de dato"<<endl;
            this->error=1;
            return this->error;
        }
    }
    else{
        if(tipo_dato!=clasificador->tipo_dato){
            cout<<"ERROR en Posicion: El clasificador no se entrenó con este tipo de dato"<<endl;
            this->error=1;
            return this->error;
        }
    }

    int e=0;
    recuadros.clear();
    Labels.clear();
    int escalas_down_up=floor(escalas/2);
    int vent_med_x=floor(tam_base.width/2);
    int vent_med_y=floor(tam_base.height/2);
    int tam_x=src.cols;
    int tam_y=src.rows;
    Mat imagen;
    src.copyTo(imagen);
    bool rot=true;
    Mat Rotada;
    int rotacion=0;
    while(rotacion<360 && rot){
        if(rotate==0)
            rot=false;
        Mat transform=cv::getRotationMatrix2D(Point(imagen.cols/2,imagen.rows/2),rotacion,1.0);
        Mat transform_inv;
        cv::invertAffineTransform(transform,transform_inv);
        transform_inv.convertTo(transform_inv,CV_32FC1);
        cv::warpAffine(imagen,Rotada,transform,Size(imagen.cols,imagen.rows));
        Mat up,down;
        Rotada.copyTo(up);
        Rotada.copyTo(down);
        for (int x=vent_med_x; x<tam_x-vent_med_x; x=x+salto){
            for(int y=vent_med_y; y<tam_y-vent_med_y; y=y+salto){
                Mat ROI_g(Rotada,cv::Rect(x-vent_med_x,y-vent_med_y,tam_base.width,tam_base.height));
                Mat ROI;
                cv::resize(ROI_g,ROI,Size(ventana_x,ventana_y));
                if(ROI.empty()){
                    cout<<"ERROR en Posicion: ROI con tamaño 0"<<endl;
                    this->error=1;
                    return this->error;
                }
                std::vector<cv::Mat> Imagen;
                if(tipo_dato==RGB){
                    Imagen.push_back(ROI);
                }
                else {
                    vector<Mat> img,des;
                    img.push_back(ROI);
                    descrip->Extract(img,des);
                    Imagen.push_back(des[0]);
                }
                vector<float> Label;
                if(tipo==MULTICLASIFICADOR){
                    if(Tipo_Multi.tipo==CASCADA)
                        e=Multi->Cascada(Imagen,Tipo_Multi.tipo_regla,Tipo_Multi.label_ref,Label);
                    else if(Tipo_Multi.tipo==VOTACION)
                        e=Multi->Votacion(Imagen,Tipo_Multi.w_clasif,Label);
                    else{
                        cout<<"ERROR en Posicion: Tipo_Multi con valor erroneo debe ser CASCADA O VOTACION"<<endl;
                        this->error=1;
                        return this->error;
                    }
                }
                else
                    e=clasificador->Autoclasificacion(Imagen,Label,true,false);
                if(e==1){
                    cout<<"ERROR en Posicion: Error en Autoclasificacion"<<endl;
                    this->error=1;
                    return this->error;
                }
                if(Label[0]>0){
                    Mat punto_rotado(3,1,CV_32FC1);
                    punto_rotado.at<float>(0,0)=x;
                    punto_rotado.at<float>(1,0)=y;
                    punto_rotado.at<float>(2,0)=1;
                    Mat punto_desrotado=transform_inv*punto_rotado;
                    int x2=punto_desrotado.at<float>(0,0);
                    int y2=punto_desrotado.at<float>(1,0);
                    recuadros.push_back(cv::RotatedRect(Point2f((float)x2,(float)y2),Size2f((float)tam_base.width,(float)tam_base.height),-rotacion));
                    Labels.push_back(Label[0]);
                }
            }
        }
        for(int k=1; k<=escalas_down_up; k++){
            Mat imagen_up;
            pyrUp( up, imagen_up, Size( up.cols*2, up.rows*2));
            int tam_x_up=imagen_up.cols;
            int tam_y_up=imagen_up.rows;
            for (int x=vent_med_x; x<tam_x_up-vent_med_x; x=x+salto){
                for(int y=vent_med_y; y<tam_y_up-vent_med_y; y=y+salto){
                    Mat ROI_up_g(imagen_up,cv::Rect(x-vent_med_x,y-vent_med_y,tam_base.width,tam_base.height));
                    Mat ROI_up;
                    cv::resize(ROI_up_g,ROI_up,Size(ventana_x,ventana_y));
                    if(ROI_up.empty()){
                        cout<<"ERROR en Posicion: ROI con tamaño 0"<<endl;
                        this->error=1;
                        return this->error;
                    }
                    std::vector<cv::Mat> Imagen;
                    if(tipo_dato==RGB){
                        Imagen.push_back(ROI_up);
                    }
                    else {
                        vector<Mat> img,des;
                        img.push_back(ROI_up);
                        descrip->Extract(img,des);
                        Imagen.push_back(des[0]);
                    }
                    vector<float> Label;
                    if(tipo==MULTICLASIFICADOR){
                        if(Tipo_Multi.tipo==CASCADA)
                            e=Multi->Cascada(Imagen,Tipo_Multi.tipo_regla,Tipo_Multi.label_ref,Label);
                        else if(Tipo_Multi.tipo==VOTACION)
                            e=Multi->Votacion(Imagen,Tipo_Multi.w_clasif,Label);
                        else{
                            cout<<"ERROR en Posicion: Tipo_Multi con valor erroneo debe ser CASCADA O VOTACION"<<endl;
                            this->error=1;
                            return this->error;
                        }
                    }
                    else
                        e=clasificador->Autoclasificacion(Imagen,Label,true,false);
                    if(e==1){
                        cout<<"ERROR en Posicion: Error en Autoclasificacion"<<endl;
                        this->error=1;
                        return this->error;
                    }
                    if(Label[0]>0){
                        Mat punto_rotado(3,1,CV_32FC1);
                        punto_rotado.at<float>(0,0)=x/(2*k);
                        punto_rotado.at<float>(1,0)=y/(2*k);
                        punto_rotado.at<float>(2,0)=1;
                        Mat punto_desrotado=transform_inv*punto_rotado;
                        int x2=punto_desrotado.at<float>(0,0);
                        int y2=punto_desrotado.at<float>(1,0);
                        recuadros.push_back(cv::RotatedRect(Point2f((float)x2,(float)y2),Size2f((float)tam_base.width/(2*k),(float)tam_base.height/(2*k)),-rotacion));
                        Labels.push_back(Label[0]);
                    }
                }
            }
            imagen_up.copyTo(up);
            Mat imagen_down;
            if(down.cols/2>tam_base.width && down.rows/2>tam_base.height){
                pyrDown( down, imagen_down, Size( down.cols/2, down.rows/2));
                int tam_x_down=imagen_down.cols;
                int tam_y_down=imagen_down.rows;
                for (int x=vent_med_x; x<tam_x_down-vent_med_x; x=x+salto){
                    for(int y=vent_med_y; y<tam_y_down-vent_med_y; y=y+salto){
                        if(tam_base.width<imagen_down.rows && tam_base.height<imagen_down.cols){
                            Mat ROI_down_g(imagen_down,cv::Rect(x-vent_med_x,y-vent_med_y,tam_base.width,tam_base.height));
                            Mat ROI_down;
                            cv::resize(ROI_down_g,ROI_down,Size(ventana_x,ventana_y));
                            if(ROI_down.empty()){
                                cout<<"ERROR en Posicion: ROI con tamaño 0"<<endl;
                                this->error=1;
                                return this->error;
                            }
                            std::vector<cv::Mat> Imagen;
                            if(tipo_dato==RGB){
                                Imagen.push_back(ROI_down);
                            }
                            else {
                                vector<Mat> img,des;
                                img.push_back(ROI_down);
                                descrip->Extract(img,des);
                                Imagen.push_back(des[0]);
                            }
                            vector<float> Label;
                            if(tipo==MULTICLASIFICADOR){
                                if(Tipo_Multi.tipo==CASCADA)
                                    e=Multi->Cascada(Imagen,Tipo_Multi.tipo_regla,Tipo_Multi.label_ref,Label);
                                else if(Tipo_Multi.tipo==VOTACION)
                                    e=Multi->Votacion(Imagen,Tipo_Multi.w_clasif,Label);
                                else{
                                    cout<<"ERROR en Posicion: Tipo_Multi con valor erroneo debe ser CASCADA O VOTACION"<<endl;
                                    this->error=1;
                                    return this->error;
                                }
                            }
                            else
                                e=clasificador->Autoclasificacion(Imagen,Label,true,false);
                            if(e==1){
                                cout<<"ERROR en Posicion: Error en Autoclasificacion"<<endl;
                                this->error=1;
                                return this->error;
                            }
                            if(Label[0]>0){
                                Mat punto_rotado(3,1,CV_32FC1);
                                punto_rotado.at<float>(0,0)=x*(2*k);
                                punto_rotado.at<float>(1,0)=y*(2*k);
                                punto_rotado.at<float>(2,0)=1;
                                Mat punto_desrotado=transform_inv*punto_rotado;
                                int x2=punto_desrotado.at<float>(0,0);
                                int y2=punto_desrotado.at<float>(1,0);
                                recuadros.push_back(cv::RotatedRect(Point2f((float)x2,(float)y2),Size2f((float)tam_base.width*(2*k),(float)tam_base.height*(2*k)),-rotacion));
                                Labels.push_back(Label[0]);
                            }
                        }
                    }
                }
                imagen_down.copyTo(down);
            }
        }
        rotacion=rotacion+rotate;
    }
    if(juntar_recuadros){
        if(rot==false)
            rotacion_recuadros=99999;
        vector<Rect> recuadros_bounding;
        vector<int> pesos;
        vector<float> angulos;
        for(uint i=0; i<recuadros.size(); i++){
            angulos.push_back(recuadros[i].angle);
            recuadros_bounding.push_back(recuadros[i].boundingRect());
            pesos.push_back(1);
        }
        if(tam_base.height>salto && tam_base.width>salto){
            for(uint i=0; i<recuadros_bounding.size(); i++){
                bool aislado=true;
                for(uint j=i+1; j<recuadros_bounding.size(); j++){
                    if(!(recuadros_bounding[i].x+recuadros_bounding[i].width<recuadros_bounding[j].x) && !(recuadros_bounding[i].x>recuadros_bounding[j].x+recuadros_bounding[j].width)
                                    && !(recuadros_bounding[i].y+recuadros_bounding[i].height<recuadros_bounding[j].y) && !(recuadros_bounding[i].y>recuadros_bounding[j].y+recuadros_bounding[j].height)){
                        aislado=false;
                    }
                }
                if(aislado && aislamiento){
                    recuadros_bounding[i].x=-1;
                    recuadros_bounding[i].y=-1;
                    recuadros_bounding[i].width=-1;
                    recuadros_bounding[i].height=-1;
                }
            }
        }
        bool cambio=true;
        int tam_actual=(int)recuadros_bounding.size();
        int tam_anterior=(int)recuadros_bounding.size();
        while(cambio){
            for(uint i=0; i<recuadros_bounding.size(); i++){
                if(!(recuadros_bounding[i].x==-1 && recuadros_bounding[i].y==-1 && recuadros_bounding[i].width==-1 && recuadros_bounding[i].height==-1)){
                    for(uint j=i+1; j<recuadros_bounding.size(); j++){
                        if(!(recuadros_bounding[j].x==-1 && recuadros_bounding[j].y==-1 && recuadros_bounding[j].width==-1 && recuadros_bounding[j].height==-1)){
                            float med_x_a=(recuadros_bounding[i].x+recuadros_bounding[i].width)/2;
                            float med_y_a=(recuadros_bounding[i].y+recuadros_bounding[i].height)/2;
                            float med_x_b=(recuadros_bounding[j].x+recuadros_bounding[j].width)/2;
                            float med_y_b=(recuadros_bounding[j].y+recuadros_bounding[j].height)/2;
                            float dist=sqrt(pow(med_x_a-med_x_b,2)+pow(med_y_a-med_y_b,2));
                            float rotac=abs(recuadros[i].angle-recuadros[j].angle);
                            bool solapado=false;
                            if(solapamiento && !(recuadros_bounding[i].x+recuadros_bounding[i].width<recuadros_bounding[j].x) && !(recuadros_bounding[i].x>recuadros_bounding[j].x+recuadros_bounding[j].width)
                                    && !(recuadros_bounding[i].y+recuadros_bounding[i].height<recuadros_bounding[j].y) && !(recuadros_bounding[i].y>recuadros_bounding[j].y+recuadros_bounding[j].height)){
                                solapado=true;
                            }
                            else
                                solapado=false;
                            if((dist<distancia_recuadros || solapado) && Labels[i]==Labels[j] && rotac<(float)rotacion_recuadros){
                                int x=999999;
                                int y=999999;
                                int x_2=0;
                                int y_2=0;
                                if (x>recuadros_bounding[i].x)
                                    x=recuadros_bounding[i].x;
                                if (x>recuadros_bounding[j].x)
                                    x=recuadros_bounding[j].x;
                                if (y>recuadros_bounding[i].y)
                                    y=recuadros_bounding[i].y;
                                if (y>recuadros_bounding[j].y)
                                    y=recuadros_bounding[j].y;
                                if (x_2<recuadros_bounding[i].x+recuadros_bounding[i].width)
                                    x_2=recuadros_bounding[i].x+recuadros_bounding[i].width;
                                if (x_2<recuadros_bounding[j].x+recuadros_bounding[j].width)
                                    x_2=recuadros_bounding[j].x+recuadros_bounding[j].width;
                                if (y_2<recuadros_bounding[i].y+recuadros_bounding[i].height)
                                    y_2=recuadros_bounding[i].y+recuadros_bounding[i].height;
                                if (y_2<recuadros_bounding[j].y+recuadros_bounding[j].height)
                                    y_2=recuadros_bounding[j].y+recuadros_bounding[j].height;
                                recuadros_bounding[i].x=x;
                                recuadros_bounding[i].y=y;
                                recuadros_bounding[i].width=abs(x_2-x);
                                recuadros_bounding[i].height=abs(y_2-y);
                                recuadros_bounding[j].x=-1;
                                recuadros_bounding[j].y=-1;
                                recuadros_bounding[j].width=-1;
                                recuadros_bounding[j].height=-1;
                                angulos[i]=angulos[i]+angulos[j];
                                pesos[i]=pesos[i]+pesos[j];
                                angulos[j]=0;
                                pesos[j]=0;
                                Labels[j]=0;
                                tam_actual--;
                            }
                        }
                    }
                }
            }
            if(tam_actual==tam_anterior)
                cambio=false;
            tam_anterior=tam_actual;
        }
        vector<RotatedRect> recs;
        vector<float> labs;
        for(uint i=0; i<recuadros_bounding.size(); i++){
            if(!(recuadros_bounding[i].x==-1 && recuadros_bounding[i].y==-1 && recuadros_bounding[i].width==-1 && recuadros_bounding[i].height==-1)){
                Rect r=recuadros_bounding[i];
                if(r.x<0)
                    r.x=0;
                if(r.y<0)
                    r.y=0;
                if(r.x+r.width>src.cols)
                    r.width=src.cols-r.x;
                if(r.y+r.height>src.rows)
                    r.height=src.rows-r.y;
                RotatedRect r_rotated;
                r_rotated.center.x=(2*r.x+r.width)/2.0;
                r_rotated.center.y=(2*r.y+r.height)/2.0;
                r_rotated.size=Size(r.width,r.height);
                r_rotated.angle=angulos[i]/pesos[i];
                float l=Labels[i];
                recs.push_back(r_rotated);
                labs.push_back(l);
            }
        }
        recuadros.clear();
        Labels.clear();
        recuadros=recs;
        Labels=labs;
    }
    this->error=0;
    return this->error;
}
