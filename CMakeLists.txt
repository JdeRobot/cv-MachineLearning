cmake_minimum_required(VERSION 2.8)
project(MachineLearning_OpenCV)

add_definitions( -DGUI )
add_definitions( -DWARNINGS )

#if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
#endif()

##CMake will search packages in /usr/local first. This avoids ROS OpenCV package.
set(CMAKE_PREFIX_PATH /usr/local)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_definitions(${Qt5Widgets_DEFINITIONS})

FIND_PROGRAM(QT_UIC_EXECUTABLE uic
  $ENV{QTDIR}/bin
)

IF (QT_UIC_EXECUTABLE)
  SET ( QT_WRAP_UI "YES")
ENDIF (QT_UIC_EXECUTABLE)

QT5_WRAP_UI(UIS_HDRS
	src/gui/con_multi.ui
	src/gui/conf_boosting.ui
	src/gui/conf_dtrees.ui
	src/gui/conf_em.ui
	src/gui/conf_ertrees.ui
	src/gui/conf_gb.ui
	src/gui/conf_haar.ui
	src/gui/conf_histograma.ui
	src/gui/conf_hog.ui
	src/gui/conf_knn.ui
	src/gui/conf_multi.ui
	src/gui/conf_neuronal.ui
	src/gui/conf_parametros.ui
	src/gui/conf_rtrees.ui
	src/gui/conf_sc_vali.ui
	src/gui/conf_svm.ui
	src/gui/config_pc.ui
	src/gui/mainwindow.ui
	src/gui/selec_param.ui
)



SET(SOURCES
	src/main.cpp
	src/Clasificadores/clasificador_boosting
	src/Clasificadores/clasificador_cascada
	src/Clasificadores/clasificador_distancias
	src/Clasificadores/clasificador_dtrees
	src/Clasificadores/clasificador_em
	src/Clasificadores/clasificador_gaussiano
	src/Clasificadores/clasificador_histograma
	src/Clasificadores/clasificador_knn
	src/Clasificadores/clasificador_neuronal
	src/Clasificadores/clasificador_rtrees
	src/Clasificadores/clasificador_svm
	src/Clasificadores/clasificador.h
	src/Clasificadores/Clasificadores.h
	src/Clasificadores/miclasificador
	src/Clasificadores/multiclasificador
	src/gui/con_multi
	src/gui/conf_boosting
	src/gui/conf_dtrees
	src/gui/conf_em
	src/gui/conf_ertrees
	src/gui/conf_gb
	src/gui/conf_haar
	src/gui/conf_histograma	
	src/gui/conf_hog
	src/gui/conf_knn
	src/gui/conf_multi
	src/gui/conf_neuronal
	src/gui/conf_parametros
	src/gui/conf_rtrees
	src/gui/conf_sc_vali
	src/gui/conf_svm
	src/gui/config_pc
	src/gui/mainwindow
	src/gui/selec_param
        src/gui/running
	src/Extraccion_Caracteristicas/basic_transformations
	src/Extraccion_Caracteristicas/Caracteristicas.h
	src/Extraccion_Caracteristicas/descriptor.h
	src/Extraccion_Caracteristicas/hog
	src/Extraccion_Caracteristicas/puntos_caracteristicos
	src/Herramientas/analisis
	src/Herramientas/auxiliares
	src/Herramientas/busqueda
	src/Herramientas/clustering
	src/Herramientas/dimensionalidad
	src/Herramientas/generacion
	src/Herramientas/Herramientas
	src/Herramientas/optimizacion
	src/Herramientas/representacion
	src/ClasificationSystem.h
)

include_directories(
	${CMAKE_CURRENT_LIST_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
	${PROJECT_SOURCE_DIR}
        ${JdeRobot_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
	${Qt5Widgets_INCLUDE_DIRS}
	${Qt5Core_INCLUDE_DIRS}
	${Qt5Gui_INCLUDE_DIRS}
)

add_executable(MachineLearning_OpenCV ${SOURCES} ${UIS_HDRS})


TARGET_LINK_LIBRARIES(MachineLearning_OpenCV
	${OpenCV_LIBRARIES}
	${Qt5Widgets_LIBRARIES}
	${Qt5Core_LIBRARIES}
        ${Qt5Gui_LIBRARIES}
        -lpthread

)
