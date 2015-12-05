#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

#define MAX_TRACK_LINE 1000

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    iters(0),
    lastMeasurIter(-1)
{
    ui->setupUi(this);

    connect(ui->pushButtonClear, SIGNAL(clicked(bool)), this, SLOT(clearTrack()));

    connect(ui->doubleSpinBoxMeasurementVariance, SIGNAL(valueChanged(double)), this, SLOT(setMeasurementNoiseVariance(double)));
    connect(ui->doubleSpinBoxProcessVariance, SIGNAL(valueChanged(double)), this, SLOT(setProcessNoiseVariance(double)));

    ui->doubleSpinBoxMeasurementVariance->setValue(10.);
    ui->doubleSpinBoxProcessVariance->setValue(1e-4);

    connect(ui->spinBoxMasterRate, SIGNAL(valueChanged(int)), this, SLOT(setMasterRate(int)));
    connect(&masterTimer, SIGNAL(timeout()), this, SLOT(update()));

    ui->spinBoxMasterRate->setValue(50);
    ui->spinBoxMeasurementRate->setValue(45);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setMasterRate(int rate)
{
    masterTimer.stop();

    int interval = int(1000. / rate);
    masterTimer.start(interval);
}

void MainWindow::setMeasurementNoiseVariance(double value)
{
    measurNoiseVar = value;

    initKalmanFilter();
}

void MainWindow::setProcessNoiseVariance(double value)
{
    processNoiseVar = value;

    initKalmanFilter();
}

void MainWindow::update()
{
    // Update mouse position

    int x_s = 0;
    int y_s = 0;

    QPoint mousePos = ui->labelTrackPreview->mapFromGlobal(QCursor::pos());

    // clamp x and y

    const int x_max = ui->labelTrackPreview->width();
    const int y_max = ui->labelTrackPreview->height();
    x_s = mousePos.x() < 0 ? 0 : mousePos.x();
    x_s = x_s > x_max ? x_max : x_s;
    y_s = mousePos.y() < 0 ? 0 : mousePos.y();
    y_s = y_s > y_max ? y_max : y_s;

    // add noise to measurements

    Mat_<float> measurNoise = Mat_<float>(2, 1);
    cv::randn(measurNoise, 0, sqrt(measurNoiseVar));

    x_s += measurNoise.at<float>(0);
    y_s += measurNoise.at<float>(1);

    mouseTrack.push_back(Point(x_s, y_s));

    if (mouseTrack.size() > MAX_TRACK_LINE) {
        mouseTrack.pop_front();
    }

    // Prediction step

    Mat prediction = KF.predict();

    // Correction step

    // do we have a new measurement?
    int masterRate = ui->spinBoxMasterRate->value();
    int measurementRate = ui->spinBoxMeasurementRate->value();
    int iterSkip = masterRate / measurementRate;
    bool haveNewMeasurement = lastMeasurIter + iterSkip <= iters;

    Mat_<float> measurement(2, 1);

    if (haveNewMeasurement) {
        measurement.at<float>(0) = x_s;
        measurement.at<float>(1) = y_s;

        lastMeasurIter = iters;
    }
    else {
        measurement.at<float>(0) = prediction.at<float>(0);
        measurement.at<float>(1) = prediction.at<float>(1);

        bool addNoise = ui->checkBoxAddNoiseFakeMeasur->isChecked();
        if (addNoise) {
            Mat_<float> noise = Mat_<float>(measurement.rows, measurement.cols);
            cv::randn(noise, 0, sqrt(measurNoiseVar));
            measurement += noise;
        }
    }

    Mat corrected = KF.correct(measurement);
    Point correctedPt(corrected.at<float>(0), corrected.at<float>(1));

    filterTrack.push_back(correctedPt);

    if (filterTrack.size() > MAX_TRACK_LINE) {
        filterTrack.pop_front();
    }

    drawTracks();

    // show info

    ui->labelMouseCurX->setText(QString::number(x_s));
    ui->labelMouseCurY->setText(QString::number(y_s));

    ++iters;
}

void MainWindow::drawTracks()
{
    Mat image = Mat::zeros(ui->labelTrackPreview->height(), ui->labelTrackPreview->width(), CV_8UC3);

    bool showMouseTrack = ui->checkBoxShowMouseTrack->isChecked();
    if (showMouseTrack) {
        for (int i = 0; i < mouseTrack.size()-1; i++) {
             line(image, mouseTrack[i], mouseTrack[i+1], Scalar(255, 255, 0), 1);
        }
    }

    bool showFilterTrack = ui->checkBoxFilterTrack->isChecked();
    if (showFilterTrack) {
        for (int i = 0; i < filterTrack.size()-1; i++) {
             line(image, filterTrack[i], filterTrack[i+1], Scalar(0, 155, 255), 1);
        }
    }

    bool showCross = ui->checkBoxShowCross->isChecked();
    if (showCross) {
        drawCross(image, mouseTrack.last(), Scalar(255, 255, 0), 5);
        drawCross(image, filterTrack.last(), Scalar(0, 155, 255), 5);
    }

    QPixmap pixmap = QPixmap::fromImage(QImage(image.data, image.cols, image.rows, QImage::Format_RGB888));
    ui->labelTrackPreview->setPixmap(pixmap);
}

void MainWindow::clearTrack()
{
    mouseTrack.clear();
    filterTrack.clear();
}

void MainWindow::initKalmanFilter()
{
#ifdef MODEL_ACCEL
    KF = KalmanFilter(6, 2);

    KF.transitionMatrix = (Mat_<float>(6, 6) << 1,0,1,0,0.5,0,   0,1,0,1,0,0.5,  0,0,1,0,1,0,  0,0,0,1,0,1,  0,0,0,0,1,0,  0,0,0,0,0,1);
#else
    KF = KalmanFilter(4, 2);

    KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
#endif

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(processNoiseVar));
    setIdentity(KF.measurementNoiseCov, Scalar::all(measurNoiseVar));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    setIdentity(KF.statePost, Scalar::all(0));
}

void drawCross(Mat img, Point center, Scalar color, float d)
{
    line( img, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, CV_AA, 0 );
    line( img, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 );
}
