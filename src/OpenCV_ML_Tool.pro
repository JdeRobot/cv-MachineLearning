#-------------------------------------------------
#
# Project created by QtCreator 2016-07-14T13:28:49
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = OpenCV_ML_Tool
TEMPLATE = app

SOURCES += \
    main.cpp \
    Clasificadores/clasificador_boosting.cpp \
    Clasificadores/clasificador_cascada.cpp \
    Clasificadores/clasificador_distancias.cpp \
    Clasificadores/clasificador_dtrees.cpp \
    Clasificadores/clasificador_em.cpp \
    Clasificadores/clasificador_gaussiano.cpp \
    Clasificadores/clasificador_histograma.cpp \
    Clasificadores/clasificador_knn.cpp \
    Clasificadores/clasificador_neuronal.cpp \
    Clasificadores/clasificador_rtrees.cpp \
    Clasificadores/clasificador_svm.cpp \
    Clasificadores/miclasificador.cpp \
    Clasificadores/multiclasificador.cpp \
    gui/con_multi.cpp \
    gui/conf_boosting.cpp \
    gui/conf_dtrees.cpp \
    gui/conf_em.cpp \
    gui/conf_ertrees.cpp \
    gui/conf_gb.cpp \
    gui/conf_haar.cpp \
    gui/conf_histograma.cpp \
    gui/conf_hog.cpp \
    gui/conf_knn.cpp \
    gui/conf_multi.cpp \
    gui/conf_neuronal.cpp \
    gui/conf_parametros.cpp \
    gui/conf_rtrees.cpp \
    gui/conf_sc_vali.cpp \
    gui/conf_svm.cpp \
    gui/config_pc.cpp \
    gui/mainwindow.cpp \
    gui/selec_param.cpp \
    Extraccion_Caracteristicas/basic_transformations.cpp \
    Extraccion_Caracteristicas/hog.cpp \
    Extraccion_Caracteristicas/puntos_caracteristicos.cpp \
    Herramientas/analisis.cpp \
    Herramientas/auxiliares.cpp \
    Herramientas/busqueda.cpp \
    Herramientas/clustering.cpp \
    Herramientas/dimensionalidad.cpp \
    Herramientas/generacion.cpp \
    Herramientas/optimizacion.cpp \
    Herramientas/representacion.cpp

HEADERS += \
    Clasificadores/clasificador_boosting.h \
    Clasificadores/clasificador_cascada.h \
    Clasificadores/clasificador_distancias.h \
    Clasificadores/clasificador_dtrees.h \
    Clasificadores/clasificador_em.h \
    Clasificadores/clasificador_gaussiano.h \
    Clasificadores/clasificador_histograma.h \
    Clasificadores/clasificador_knn.h \
    Clasificadores/clasificador_neuronal.h \
    Clasificadores/clasificador_rtrees.h \
    Clasificadores/clasificador_svm.h \
    Clasificadores/clasificador.h \
    Clasificadores/Clasificadores.h \
    Clasificadores/miclasificador.h \
    Clasificadores/multiclasificador.h \
    gui/con_multi.h \
    gui/conf_boosting.h \
    gui/conf_dtrees.h \
    gui/conf_em.h \
    gui/conf_ertrees.h \
    gui/conf_gb.h \
    gui/conf_haar.h \
    gui/conf_histograma.h \
    gui/conf_hog.h \
    gui/conf_knn.h \
    gui/conf_multi.h \
    gui/conf_neuronal.h \
    gui/conf_parametros.h \
    gui/conf_rtrees.h \
    gui/conf_sc_vali.h \
    gui/conf_svm.h \
    gui/config_pc.h \
    gui/mainwindow.h \
    gui/selec_param.h \
    Extraccion_Caracteristicas/basic_transformations.h \
    Extraccion_Caracteristicas/Caracteristicas.h \
    Extraccion_Caracteristicas/descriptor.h \
    Extraccion_Caracteristicas/hog.h \
    Extraccion_Caracteristicas/puntos_caracteristicos.h \
    Herramientas/analisis.h \
    Herramientas/auxiliares.h \
    Herramientas/busqueda.h \
    Herramientas/clustering.h \
    Herramientas/dimensionalidad.h \
    Herramientas/generacion.h \
    Herramientas/Herramientas.h \
    Herramientas/optimizacion.h \
    Herramientas/representacion.h \
    ClasificationSystem.h

FORMS += \
    gui/con_multi.ui \
    gui/conf_boosting.ui \
    gui/conf_dtrees.ui \
    gui/conf_em.ui \
    gui/conf_ertrees.ui \
    gui/conf_gb.ui \
    gui/conf_haar.ui \
    gui/conf_histograma.ui \
    gui/conf_hog.ui \
    gui/conf_knn.ui \
    gui/conf_multi.ui \
    gui/conf_neuronal.ui \
    gui/conf_parametros.ui \
    gui/conf_rtrees.ui \
    gui/conf_sc_vali.ui \
    gui/conf_svm.ui \
    gui/config_pc.ui \
    gui/mainwindow.ui \
    gui/selec_param.ui

LIBS +=-L/usr/local/lib \
    -lopencv_core \
    -lopencv_highgui \
    -lopencv_tracking \
    -lopencv_videoio \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lopencv_ml \
    -lopencv_features2d \
    -lopencv_xfeatures2d \
    -lopencv_objdetect \
    -lpthread

DESTDIR = ../build

OBJECTS_DIR = $$DESTDIR/.obj
MOC_DIR = $$DESTDIR/.moc
RCC_DIR = $$DESTDIR/.qrc
UI_DIR = $$DESTDIR/.ui

DEFINES +=GUI\
        WARNINGS

