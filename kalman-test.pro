#-------------------------------------------------
#
# Project created by QtCreator 2015-12-05T00:11:22
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = kalman-test
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

INCLUDEPATH += $$(OPENCV_PATH)\include
LIBS += -L$$(OPENCV_PATH)\x64\vc12\lib

LIBS +=  \
    -lopencv_core300 \
    -lopencv_highgui300 \
    -lopencv_imgproc300 \
    -lopencv_imgcodecs300 \
    -lopencv_features2d300 \
    -lopencv_calib3d300 \
    -lopencv_video300 \
    -lopencv_videoio300 \

DEFINES += MODEL_ACCEL=1
