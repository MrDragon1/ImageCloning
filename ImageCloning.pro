#-------------------------------------------------
#
# Project created by QtCreator 2020-08-12T13:11:16
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ImageCloning
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++14
CONFIG += console

INCLUDEPATH += G:\Opencv\install\include \
            G:\CGAL\CGAL5.0.3\include \
            G:\CGAL\CGAL5.0.3\auxiliary\gmp\include \
            D:\boost_1_74_0


LIBS += -LG:\Opencv\install\x64\mingw\lib -llibopencv_core440 -llibopencv_imgproc440 -llibopencv_highgui440 -llibopencv_photo440 -llibopencv_imgcodecs440 \
        -LG:\CGAL\CGAL5.0.3\auxiliary\gmp\lib -llibgmp-10 \
        -LG:\CGAL\CGAL5.0.3\build\lib
SOURCES += \
        info.cpp \
        main.cpp \
        mainwindow.cpp \
        vector2d.cpp \
        vector3d.cpp

HEADERS += \
        info.h \
        mainwindow.h \
        vector2d.h \
        vector3d.h

FORMS += \
        info.ui \
        mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
