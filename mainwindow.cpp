#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QKeyEvent>

#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

#define MAX_TRACK_LINE 1000

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    iters(0),
    lastMeasurIter(-1),
    paused(false)
{
    ui->setupUi(this);

    connect(ui->pushButtonClear, SIGNAL(clicked(bool)), this, SLOT(clearTrack()));

    connect(ui->checkBoxUse3orderModel, SIGNAL(clicked(bool)), this, SLOT(use3OrderModel(bool)));
    connect(ui->doubleSpinBoxMeasurementVariance, SIGNAL(valueChanged(double)), this, SLOT(setMeasurementNoiseVariance(double)));
    connect(ui->doubleSpinBoxProcessVariance, SIGNAL(valueChanged(double)), this, SLOT(setProcessNoiseVariance(double)));

    ui->checkBoxUse3orderModel->setChecked(true);
    ui->doubleSpinBoxMeasurementVariance->setValue(50.);
    ui->doubleSpinBoxProcessVariance->setValue(1e-5);

    connect(ui->spinBoxMasterRate, SIGNAL(valueChanged(int)), this, SLOT(setMasterRate(int)));
    connect(&masterTimer, SIGNAL(timeout()), this, SLOT(update()));

    ui->spinBoxMasterRate->setValue(50);
    ui->spinBoxMeasurSampling->setValue(1);

//    grabKeyboard();
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

    bool use3orderModel = ui->checkBoxUse3orderModel->isChecked();
    initKalmanFilter(use3orderModel);
}

void MainWindow::setProcessNoiseVariance(double value)
{
    processNoiseVar = value;

    bool use3orderModel = ui->checkBoxUse3orderModel->isChecked();
    initKalmanFilter(use3orderModel);
}

void MainWindow::use3OrderModel(bool use)
{
    initKalmanFilter(use);
}

void MainWindow::update()
{
    if (paused) {
        return;
    }

    // do we have a new measurement?
    int skipIters = ui->spinBoxMeasurSampling->value();
    bool haveNewMeasurement = (lastMeasurIter + skipIters) < iters;

    Point mpos = getMousePos();

    mouseRealTrack.push_back(mpos);

    if (mouseRealTrack.size() > MAX_TRACK_LINE) {
        mouseRealTrack.pop_front();
    }

    // add noise to measurements
    Mat_<float> noise = Mat_<float>(2, 1);
    cv::randn(noise, 0, sqrt(measurNoiseVar));

    mpos = Point(mpos.x + noise.at<float>(0), mpos.y + noise.at<float>(1));

    mouseMeasurTrack.push_back(mpos);

    if (mouseMeasurTrack.size() > MAX_TRACK_LINE) {
        mouseMeasurTrack.pop_front();
    }

    Mat prediction = KF.predict();
    Point predictiedPt(prediction.at<float>(0), prediction.at<float>(1));

    bool fakeMeasurement = ui->checkBoxUseFakeMeausur->isChecked();

    if (!haveNewMeasurement && fakeMeasurement) {
        Mat_<float> measurement(2, 1);
        measurement.at<float>(0) = prediction.at<float>(0);
        measurement.at<float>(1) = prediction.at<float>(1);

        bool addNoise = ui->checkBoxAddNoiseFakeMeasur->isChecked();
        if (addNoise) {
            Mat_<float> noise = Mat_<float>(measurement.rows, measurement.cols);
            cv::randn(noise, 0, sqrt(measurNoiseVar));
            measurement += noise;
        }

        Mat corrected = KF.correct(measurement);
        Point correctedPt(corrected.at<float>(0), corrected.at<float>(1));

        filterTrack.push_back(correctedPt);
    }
    else if (!haveNewMeasurement && !fakeMeasurement) {
        filterTrack.push_back(predictiedPt);
    }
    else {
        Mat_<float> measurement(2, 1);
        measurement.at<float>(0) = mpos.x;
        measurement.at<float>(1) = mpos.y;

        Mat corrected = KF.correct(measurement);
        Point correctedPt(corrected.at<float>(0), corrected.at<float>(1));

        filterTrack.push_back(correctedPt);

        lastMeasurIter = iters;
    }

    if (filterTrack.size() > MAX_TRACK_LINE) {
        filterTrack.pop_front();
    }

    drawTracks();

    // show info

    ui->labelMouseCurX->setText(QString::number(mpos.x));
    ui->labelMouseCurY->setText(QString::number(mpos.y));

    ++iters;
}

void MainWindow::drawTracks()
{
    Mat image = Mat::zeros(ui->labelTrackPreview->height(), ui->labelTrackPreview->width(), CV_8UC3);

    bool showRealTrack = ui->checkBoxShowRealTrack->isChecked();
    if (showRealTrack) {
        for (int i = 0; i < mouseRealTrack.size()-1; i++) {
             line(image, mouseRealTrack[i], mouseRealTrack[i+1], Scalar(255, 0, 0), 2);
        }
    }

    bool showMeasurTrack = ui->checkBoxShowMeasurTrack->isChecked();
    if (showMeasurTrack) {
        for (int i = 0; i < mouseMeasurTrack.size()-1; i++) {
             line(image, mouseMeasurTrack[i], mouseMeasurTrack[i+1], Scalar(255, 255, 0), 1);
        }
    }

    bool showFilterTrack = ui->checkBoxFilterTrack->isChecked();
    if (showFilterTrack) {
        for (int i = 0; i < filterTrack.size()-1; i++) {
             line(image, filterTrack[i], filterTrack[i+1], Scalar(0, 155, 255), 2);
        }
    }

    bool showCross = ui->checkBoxShowCross->isChecked();
    if (showCross) {
        drawCross(image, mouseRealTrack.last(), Scalar(255, 255, 0), 5);
        drawCross(image, filterTrack.last(), Scalar(0, 155, 255), 5);
    }

    QPixmap pixmap = QPixmap::fromImage(QImage(image.data, image.cols, image.rows, QImage::Format_RGB888));
    ui->labelTrackPreview->setPixmap(pixmap);
}

void MainWindow::clearTrack()
{
    mouseMeasurTrack.clear();
    mouseRealTrack.clear();
    filterTrack.clear();
}

void MainWindow::initKalmanFilter(bool use3orderModel)
{
    if (use3orderModel) {
        KF = KalmanFilter(6, 2);

        KF.transitionMatrix = (Mat_<float>(6, 6) << 1,0,1,0,0.5,0,   0,1,0,1,0,0.5,  0,0,1,0,1,0,  0,0,0,1,0,1,  0,0,0,0,1,0,  0,0,0,0,0,1);
    }
    else {
        KF = KalmanFilter(4, 2);

        KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    }

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(processNoiseVar));
    setIdentity(KF.measurementNoiseCov, Scalar::all(measurNoiseVar));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    setIdentity(KF.statePost, Scalar::all(0));
}

Point MainWindow::getMousePos()
{
    int mx = 0;
    int my = 0;

    QPoint mousePos = ui->labelTrackPreview->mapFromGlobal(QCursor::pos());

    // clamp x and y
    const int x_max = ui->labelTrackPreview->width();
    const int y_max = ui->labelTrackPreview->height();
    mx = mousePos.x() < 0 ? 0 : mousePos.x();
    mx = mx > x_max ? x_max : mx;
    my = mousePos.y() < 0 ? 0 : mousePos.y();
    my = my > y_max ? y_max : my;

    return Point(mx, my);
}

void drawCross(Mat img, Point center, Scalar color, float d)
{
    line( img, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, CV_AA, 0 );
    line( img, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 );
}


void MainWindow::keyPressEvent(QKeyEvent *ev)
{
    if (ev->key() == Qt::Key_Space) {
        paused = !paused;
    }
}
