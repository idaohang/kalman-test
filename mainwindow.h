#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QQueue>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void setMasterRate(int rate);

    void setMeasurementNoiseVariance(double value);

    void setProcessNoiseVariance(double value);

    void update();

    void drawTracks();

    void clearTrack();

private:
    void initKalmanFilter();

    cv::Point getMousePos();

    Ui::MainWindow *ui;

    QTimer masterTimer;

    cv::KalmanFilter KF;

    double measurNoiseVar;
    double processNoiseVar;

    QQueue<cv::Point> mouseTrack;
    QQueue<cv::Point> filterTrack;

    int iters;
    int lastMeasurIter;

    bool paused;

    // QWidget interface
protected:
    virtual void keyPressEvent(QKeyEvent *) override;
};

void drawCross(cv::Mat img, cv::Point center, cv::Scalar color, float d);                                 \

#endif // MAINWINDOW_H
