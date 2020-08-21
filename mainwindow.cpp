#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "info.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    IsDraw = true;
    Moved = false;
    Movearea = false;

    ui->groupBox_3->setGeometry(9,9,this->width() - 18,60);
    ui->groupBox->setGeometry(9,75,this->width()/2 - 18,this->height() - 85);
    ui->groupBox_2->setGeometry(this->width()/2 + 9 ,75,this->width()/2 - 18,this->height() - 85);
    srcpoint = QPoint(ui->groupBox->pos().x() + 30,ui->groupBox->pos().y() + 30);
    tarpoint = QPoint(ui->groupBox_2->pos().x() + 30,ui->groupBox_2->pos().y() + 30);
    ui->Load_source_btn->setGeometry( 30 ,ui->groupBox->height() - 40 ,75,25);
    ui->Set_btn->setGeometry(ui->groupBox->width() / 3 + 30 ,ui->groupBox->height() - 40 ,75,25);
    ui->Reset_btn->setGeometry(2*ui->groupBox->width() / 3 + 30 ,ui->groupBox->height() - 40 ,75,25);
    ui->Load_target_btn->setGeometry(30 , ui->groupBox_2->height() - 40 ,75,25);
    ui->Clone_btn->setGeometry(ui->groupBox_2->width() / 3 + 30 ,ui->groupBox_2->height() - 40 ,75,25);
    ui->Reclone_btn->setGeometry(2*ui->groupBox_2->width() / 3 + 30 ,ui->groupBox_2->height() - 40 ,75,25);

    src_img.load("G:\\Code\\LOLTimeTracker\\img\\champion\\Yasuo.png");
    src_img_copy = src_img.copy();
    tar_img.load("G:\\Code\\LOLTimeTracker\\img\\champion\\Zed.png");
    tar_img_copy = tar_img.copy();

    src_label = new QLabel;
    tar_label = new QLabel;
    src_label->setAlignment(Qt::AlignTop);
    tar_label->setAlignment(Qt::AlignTop);

    src_label->setPixmap(QPixmap::fromImage(src_img));
    ui->scrollArea->setWidget(src_label);
    ui->scrollArea->setGeometry(30,30,ui->groupBox->width() - 80 ,ui->groupBox->height() - 80);
    tar_label->setPixmap(QPixmap::fromImage(tar_img));
    ui->scrollArea_2->setWidget(tar_label);
    ui->scrollArea_2->setGeometry(30,30,ui->groupBox_2->width() - 80,ui->groupBox_2->height() - 80);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::paintEvent(QPaintEvent*)
{

    ui->groupBox_3->setGeometry(9,9,this->width() - 18,60);
    ui->groupBox->setGeometry(9,75,this->width()/2 - 18,this->height() - 85);
    ui->groupBox_2->setGeometry(this->width()/2 + 9 ,75,this->width()/2 - 18,this->height() - 85);
    srcpoint = QPoint(ui->groupBox->pos().x() + 30,ui->groupBox->pos().y() + 30);
    tarpoint = QPoint(ui->groupBox_2->pos().x() + 30,ui->groupBox_2->pos().y() + 30);
    ui->Load_source_btn->setGeometry( 30 ,ui->groupBox->height() - 40 ,75,25);
    ui->Set_btn->setGeometry(ui->groupBox->width() / 3 + 30 ,ui->groupBox->height() - 40 ,75,25);
    ui->Reset_btn->setGeometry(2*ui->groupBox->width() / 3 + 30 ,ui->groupBox->height() - 40 ,75,25);
    ui->Load_target_btn->setGeometry(30 , ui->groupBox_2->height() - 40 ,75,25);
    ui->Clone_btn->setGeometry(ui->groupBox_2->width() / 3 + 30 ,ui->groupBox_2->height() - 40 ,75,25);
    ui->Reclone_btn->setGeometry(2*ui->groupBox_2->width() / 3 + 30 ,ui->groupBox_2->height() - 40 ,75,25);

    //QPainter painter(this);
    //painter.drawImage(srcpoint,src_img);
    //painter.drawImage(tarpoint,tar_img);

    src_label->setPixmap(QPixmap::fromImage(src_img));
    ui->scrollArea->setWidget(src_label);
    ui->scrollArea->setGeometry(30,30,ui->groupBox->width() - 80,ui->groupBox->height() - 80);
    tar_label->setPixmap(QPixmap::fromImage(tar_img));
    ui->scrollArea_2->setWidget(tar_label);
    ui->scrollArea_2->setGeometry(30,30,ui->groupBox_2->width() - 80,ui->groupBox_2->height() - 80);

}

void MainWindow::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton)
    {
        if(!Moved
                && event->pos().x() >= ui->groupBox->geometry().x() + 30 && event->pos().y() >= ui->groupBox->geometry().y() + 30
                && event->pos().x() <= ui->groupBox->geometry().x() + 30 + std::min(src_img.width(),ui->scrollArea->width()) && event->pos().y() <= ui->groupBox->geometry().y() + 30 + std::min(src_img.height(),ui->scrollArea->height()))
        {
            IsDraw = true;
            QPoint scrollbar_offset  = QPoint(ui->scrollArea->horizontalScrollBar()->value(),ui->scrollArea->verticalScrollBar()->value());
            startpoint = event->pos() - QPoint(ui->groupBox->geometry().x() + 30, ui->groupBox->geometry().y() + 30) + scrollbar_offset;
            endpoint = startpoint;
            line.push_back(cv::Point(endpoint.x(),endpoint.y()));
        }
        else if(Moved)
//                && event->pos().x() - ui->groupBox_2->pos().x()  >= ui->area_label->x()
//                && event->pos().y() - ui->groupBox_2->pos().y() >= ui->area_label->y()
//                && event->pos().x() - ui->groupBox_2->pos().x() <= ui->area_label->x() + ui->area_label->width()
//                && event->pos().y() - ui->groupBox_2->pos().y() <= ui->area_label->y() + ui->area_label->height())
        {
            relative_position = event->pos() - ui->groupBox_2->pos() - ui->area_label->pos();
            if(map[relative_position.y()][relative_position.x()]){
                Movearea = true;
            }


        }
    }
}

void MainWindow::mouseMoveEvent(QMouseEvent* event)
{
    if (!Moved && IsDraw && (event->buttons() & Qt::LeftButton))
    {
        QPoint scrollbar_offset  = QPoint(ui->scrollArea->horizontalScrollBar()->value(),ui->scrollArea->verticalScrollBar()->value());
        endpoint = event->pos() - srcpoint + scrollbar_offset;
        line.push_back(cv::Point(endpoint.x(),endpoint.y()) );
        paint(src_img);
    }
    else if(Movearea && (event->buttons() & Qt::LeftButton))
    {
        QPoint area_pos = event->pos() - ui->groupBox_2->pos() - relative_position;
        ui->area_label->setGeometry(area_pos.x(),area_pos.y(),src_img.width(),src_img.height());
        ui->area_label->setPixmap(QPixmap::fromImage(cvMat2QImage(res_mat)));
        update();
    }
}

void MainWindow::mouseReleaseEvent(QMouseEvent*)
{
    if(!Moved && IsDraw)
    {
        IsDraw = false;
        paint(src_img);
    }
    else if(Movearea)
    {
        Movearea = false;
    }
}

void MainWindow::paint(QImage& Img)
{
    QPainter painter(&Img);
    QPen pen;
    pen.setWidth(1);
    pen.setColor(QColor(255,0,0));
    painter.setPen(pen);
    painter.drawLine(startpoint , endpoint);
    startpoint = endpoint;
    update();
}

void MainWindow::on_Info_btn_clicked()
{
    Info* info = new Info();
    info->show();
}

void MainWindow::on_Load_source_btn_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
            tr("File Select "),
            "G:\\Code\\Opencv\\ImageCloning\\",
            tr("Image File(*png *jpg);;"));
    src_img_copy.load(fileName);
    src_img = src_img_copy.copy();
    src_mat = cv::imread(fileName.toStdString());
    src_mat.copyTo(src_mat_copy);
}

void MainWindow::on_Set_btn_clicked()
{
    using namespace std;
    using namespace cv;
    QTime startTime = QTime::currentTime();
    cout << "Begin preprocess ..."<<endl;

    //get mask area
    Mat mask = Mat::zeros(src_mat.size(),CV_8UC1);
    polylines(mask,line,true,cv::Scalar(0), 2, 8, 0);
    fillPoly(mask,line,cv::Scalar(255), 8, 0);
    vector<vector<Point>> contours;
    vector<Vec4i> hierachy;
    findContours(mask, contours, hierachy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0,0));
    //drawContours(img, contours, -1, Scalar(0,0,255), 1, 8, hierachy);
    vector<Point> contours_ploy;
    approxPolyDP(contours[0],contours_ploy,5,true);

    mask = Mat::zeros(src_mat.size(),CV_8UC3);
    polylines(mask,contours_ploy,true,cv::Scalar(0,0,0), 2, 8, 0);
    fillPoly(mask,contours_ploy,cv::Scalar(255,255,255), 8, 0);
    imshow("mask",mask);

    //save area in map
    map.resize(mask.rows);
    for( size_t i = 0 ;i < mask.rows;i++)
        map[i].resize(mask.cols);
    for(size_t i = 0 ;i < mask.rows;i++)
    {
        for(size_t j = 0 ;j < mask.cols;j++)
        {
            if((mask.at<Vec3b>(i,j)[0] == 0 && mask.at<Vec3b>(i,j)[1] == 0 && mask.at<Vec3b>(i,j)[2] == 0)){
                map[i][j] = false;
            }
            else map[i][j] = true;
        }
    }
    //create transparent image
    res_mat = Mat::zeros(src_mat.size(),CV_8UC4);
    for(size_t i = 0 ;i < res_mat.rows;i++)
    {
        for(size_t j = 0 ;j < res_mat.cols;j++)
        {
            if(map[i][j]){
                res_mat.at<Vec4b>(i,j)[3] = 255;
                res_mat.at<Vec4b>(i,j)[2] = src_mat.at<Vec3b>(i,j)[2];
                res_mat.at<Vec4b>(i,j)[1] = src_mat.at<Vec3b>(i,j)[1];
                res_mat.at<Vec4b>(i,j)[0] = src_mat.at<Vec3b>(i,j)[0];
            }
            else {
                res_mat.at<Vec4b>(i,j)[3] = 0;
            }
        }
    }

    //get roi
    vector<vector2d> ROIBoundList;
    for(auto i:contours_ploy)
    {
        ROIBoundList.push_back(vector2d(i.x,i.y));
    }
    CalBoundPoint(ROI,ROIBoundList);
    cout<<"contours size == "<< contours_ploy.size()<<" ROI size == "<< ROI.size() << endl;

    //show transparent img
    Moved = true;
    IsDraw = false;
    ui->area_label->setVisible(true);
    ui->area_label->setGeometry(30,30,src_img.width(),src_img.height());
    ui->area_label->setPixmap(QPixmap::fromImage(cvMat2QImage(res_mat)));

    QTime stopTime = QTime::currentTime();
    int elapsed = startTime.msecsTo(stopTime);
    cout<<"Preprocess time "<<elapsed<<" ms" <<endl;
}

void MainWindow::on_Reset_btn_clicked()
{
    src_img = src_img_copy.copy();
    tar_img = tar_img_copy.copy();
    src_mat_copy.copyTo(src_mat);
    tar_mat_copy.copyTo(tar_mat);
    line.clear();
    map.clear();
    ROI.clear();
    IsDraw = true;
    Moved = false;
    Movearea = false;
    update();
}

void MainWindow::on_Load_target_btn_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
            tr("File Select "),
            "G:\\Code\\Opencv\\ImageCloning\\",
            tr("Image File(*png *jpg);;"));
    tar_img.load(fileName);
    tar_mat = cv::imread(fileName.toStdString());
    tar_mat.copyTo(tar_mat_copy);
    tar_img_copy = tar_img.copy();
}

void MainWindow::on_Reclone_btn_clicked()
{
    ui->area_label->setVisible(true);
    Movearea = false;
    tar_img = tar_img_copy.copy();
    tar_mat_copy.copyTo(tar_mat);//深拷贝
    update();
}

void MainWindow::on_Clone_btn_clicked()
{
    using namespace std;
    using namespace cv;
    MVC_Compute_Optimized();
    //MVC_Compute();
    tar_img = cvMat2QImage(tar_mat);
    //imshow("target",tar_mat);
    ui->area_label->setVisible(false);
    update();
}

QImage MainWindow::cvMat2QImage(const cv::Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS = 1
    if(mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        // Set the color table (used to translate colour indexes to qRgb values)
        image.setColorCount(256);
        for(int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        // Copy input Mat
        uchar *pSrc = mat.data;
        for(int row = 0; row < mat.rows; row ++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    // 8-bits unsigned, NO. OF CHANNELS = 3
    else if(mat.type() == CV_8UC3)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else if(mat.type() == CV_8UC4)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image.copy();
    }
    else
    {
        return QImage();
    }
}

void MainWindow::MVC_Compute()
{
    QTime startTime = QTime::currentTime();
    using namespace std;
    using namespace cv;
    QPoint area_tar_position = ui->area_label->pos() - QPoint(30,30) + QPoint(ui->scrollArea_2->horizontalScrollBar()->value(),ui->scrollArea_2->verticalScrollBar()->value());
    vector<vector<vector<double>>> MVC(src_mat.rows,vector<vector<double>>(src_mat.cols,vector<double>(ROI.size(),0))); //[x][y][z] 坐标x,y像素的lamda(z)的值
    cout<< "Calculating the mean-value coordinates ..."<<endl;

    //calculate the mean-value coordinates
    for( int i = 0;i<src_mat.rows;i++)
    {
        for( int j = 0;j<src_mat.cols;j++)
        {
            if(map[i][j])
            {
                vector2d x(j,i);
                vector<double> w;
                double sum_w = 0;
                for(unsigned int  k = 0;k < ROI.size();k++)
                {
                    vector2d dot1 = ROI[k%ROI.size()],dot2 = ROI[(k+1)%ROI.size()] ,dot3 = ROI[(k-1)%ROI.size()];
                    double tan1 = Tan_Compute(x,dot1,dot2);
                    double tan2 = Tan_Compute(x,dot1,dot3);
                    //if(tan1+tan2 < 0) cout <<"tan < 0 =>"<<i<<","<< j <<" "<<tan1 << " "<< tan2 << " " <<endl;
                    w.push_back((tan1+tan2)/(x-dot1).Mod());
                    if(_isnan(w[k])==1)
                    {
                        w[k] = 0;
                    }
                    sum_w += w[k];
                }

                for(unsigned int k = 0;k < w.size();k++)
                {
                    MVC[i][j][k] = w[k] / sum_w;
                    //if(w[k] /sum_w >= 0.5 ) cout <<i<<","<< j << "sum_w = " <<sum_w << " w["<<k<<"] = "<<w[k]<<endl;
                }
            }
        }
    }
    cout<< "Calculate the mean-value coordinates done!"<<endl;
    cout<< "Calculating the diff ..."<<endl;
    //compute diff
    vector<vector<int>> diff(ROI.size(),vector<int>(3));
    for(unsigned int k = 0;k<ROI.size();k++)
    {
        int i = ROI[k].y,j = ROI[k].x;
        diff[k][0] = - src_mat.at<Vec3b>(i,j)[0] + tar_mat.at<Vec3b>(i + area_tar_position.y(),j + area_tar_position.x())[0];
        diff[k][1] = - src_mat.at<Vec3b>(i,j)[1] + tar_mat.at<Vec3b>(i + area_tar_position.y(),j + area_tar_position.x())[1];
        diff[k][2] = - src_mat.at<Vec3b>(i,j)[2] + tar_mat.at<Vec3b>(i + area_tar_position.y(),j + area_tar_position.x())[2];
        //cout << "diff["<< k << "] = " << diff[k][0] << " " << diff[k][1] << " " <<diff[k][2] << " " <<endl;
    }
    cout<< "Calculate the diff done!"<< endl;
    cout<< "Evaluating the mean-value interpolant ..."<<endl;
    //Evaluate the mean-value interpolant at x
    for(int i = 0;i<src_mat.rows;i++)
    {
        for(int j = 0;j<src_mat.cols;j++)
        {
            if(map[i][j])
            {
               // cout << i <<","<< j <<"  ";
                for(int k = 0;k<3;k++)
                {
                    double r = 0;
                    for(unsigned int l = 0;l<diff.size();l++)
                    {
                        r+=MVC[i][j][l]*diff[l][k];
                    }
                    int pix = r + src_mat.at<Vec3b>(i,j)[k];
                    //cout << r <<" + "<< (int)res_mat.at<Vec3b>(i,j)[k] << " = ";
                    if(pix > 255) pix = 255;
                    if(pix < 0) pix = 0;
                    tar_mat.at<Vec3b>(i + area_tar_position.y(),j + area_tar_position.x())[k] = pix ;
                    //cout << (int)tar_mat.at<Vec3b>(i + area_tar_position.y(),j + area_tar_position.x())[k] << " >>> ";
                }
                //cout << endl;
            }
        }
    }
    cout << "Evaluate the mean-value interpolant done!" << endl;
    QTime stopTime = QTime::currentTime();
    int elapsed = startTime.msecsTo(stopTime);
    cout<<"total time "<<elapsed<<" ms" << endl;
}

void MainWindow::MVC_Compute_Optimized()
{
    using namespace std;
    using namespace cv;
    QTime startTime = QTime::currentTime();
    QPoint area_tar_position = ui->area_label->pos() - QPoint(30,30) + QPoint(ui->scrollArea_2->horizontalScrollBar()->value(),ui->scrollArea_2->verticalScrollBar()->value());

    //Step1:找到边界上所有的像素点
//    vector<vector2d> ROIBoundPointList;
//    CalBoundPoint(ROIBoundPointList);
//    cout << "ROIBoundPointList size == "<<ROI.size()<<endl;

    CDT cdt;
    vector<Vertex_handle> vertexList;
    for(size_t i = 0; i<ROI.size(); i++)
    {
        vertexList.push_back(cdt.insert(CDTPoint(ROI[i].x, ROI[i].y )));
    }

    for(size_t i =0;i<vertexList.size()-1;i++)
    {
        cdt.insert_constraint(vertexList[i],vertexList[i+1]);
    }

    cout << "Number of vertices: " << cdt.number_of_vertices() <<endl;
    cout << "Meshing the triangulation..." << endl;

    CGAL::refine_Delaunay_mesh_2(cdt, Criteria());
    cout << "Number of vertices: " << cdt.number_of_vertices() <<endl;

    vector<vector2d> vertex_list;
    std::map<vector2d, size_t> vertex_map;
    for(CDT::Vertex_iterator vit = cdt.vertices_begin(); vit!= cdt.vertices_end(); ++vit)
    {
        vertex_map.insert(make_pair(vector2d(vit->point().x(), vit->point().y()), vertex_list.size()));
        vertex_list.push_back(vector2d(vit->point().x(), vit->point().y()));
    }

    cout << "Calculating diff ..."<<endl;
    //计算边界的像素差值
    vector<int> diff;
    for(size_t k = 0; k < ROI.size()-1; k++)
    {
        for(size_t bi = 0; bi < 3; bi++)
        {

            int i = ROI[k].y,j = ROI[k].x;
            int d = (int)(- src_mat.at<Vec3b>(i,j)[bi] + tar_mat.at<Vec3b>(i + area_tar_position.y(),j + area_tar_position.x())[bi]);
            diff.push_back(d);
        }
    }
    cout << "Calculate diff done!"<<endl;
    cout << "Calculating mean-value coordinates..." << endl;
    vector<Vec3d> tri_mesh_vertex_R(vertex_list.size());
    #pragma omp parallel for        //开启OpenMP并行加速
    for (size_t vi = 0; vi < vertex_list.size(); ++vi)
    {
        //逐点计算MVC
        vector<double> MVC(ROI.size()-1, 0);
        for(size_t pi = 1; pi < ROI.size(); pi++)
        {
            double tan1 = Tan_Compute(vertex_list[vi], ROI[pi-1], ROI[pi]);
            double tan2 = Tan_Compute(vertex_list[vi], ROI[pi-1], ROI[pi-2]);
            double w_a = tan1 + tan2;
            double w_b = (ROI[pi-1] - vertex_list[vi]).Mod();
            MVC[pi-1] = w_a / w_b;
            if(_isnan(MVC[pi-1])==1)
            {
                MVC[pi-1] = 0;
            }
        }

        double sum = 0;
        for(size_t pi = 0; pi < MVC.size(); pi++)
        {
            sum = sum + MVC[pi];
        }

        for(size_t pi = 0; pi < MVC.size(); pi++)
        {
            MVC[pi] = MVC[pi] / sum;
        }

        Vec3d r(0.0,0.0,0.0);
        for(size_t pi = 0; pi < MVC.size(); pi++)
        {
            for(int bi = 0; bi < 3; bi++)
            {
                r[bi] = r[bi] + MVC[pi] * diff[pi * 3 + bi];
            }
        }

        tri_mesh_vertex_R[vi] = r;
    }
    cout<<"Calculate mean-value coordinates done!" << endl;

    cout << "Split triangle ..."<<endl;
    //遍历每一个三角面
    vector<vector<size_t>> face_vertex_index;
    CDT::Face_iterator fit;
    for (fit = cdt.faces_begin(); fit!= cdt.faces_end(); ++fit)
    {
        vector<size_t> index(3);
        for(size_t i = 0; i<3; i++)
        {
            auto iter = vertex_map.find(vector2d(fit->vertex(i)->point().x(), fit->vertex(i)->point().y()));
            if(iter == vertex_map.end())
            {
                continue;
            }
            index[i] = iter->second;
        }
        face_vertex_index.push_back(index);
    }

    vector<vector<int>> clipMap(src_mat.rows,vector<int>(src_mat.cols));//标识范围内的点: 0标识初始不能写入，1以上标识在那个三角形

    #pragma omp parallel for        //开启OpenMP并行加速
    for(size_t fi = 0; fi < face_vertex_index.size(); fi++)
    {
        vector2d v0 = vertex_list[face_vertex_index[fi][0]];
        vector2d v1 = vertex_list[face_vertex_index[fi][1]];
        vector2d v2 = vertex_list[face_vertex_index[fi][2]];

        double minX = std::min(std::min(v0.x, v1.x), v2.x);
        double minY = std::min(std::min(v0.y, v1.y), v2.y);
        double maxX = std::max(std::max(v0.x, v1.x), v2.x);
        double maxY = std::max(std::max(v0.y, v1.y), v2.y);

        int sX = std::max(int(floor(minX)), 0);
        int sY = std::max(int(floor(minY)), 0);
        int eX = std::max(int(ceil(maxX)), src_mat.cols - 1);
        int eY = std::max(int(ceil(maxY)), src_mat.rows - 1);

        for(int yi = sY; yi <= eY; yi++)
        {
            for(int xi = sX; xi <= eX; xi++)
            {
                if(PointinTriangle(v0, v1, v2, vector2d(xi, yi)))
                {
                    clipMap[yi][xi] = fi+1;
                }
            }
        }
    }

    cout << "Split triangle done!"<<endl;
    cout<<"Evaluating the mean-value interpolant ..." << endl;

    #pragma omp parallel for
    for (size_t ri = 0; ri < src_mat.rows; ++ri)
    {
        for (size_t ci = 0; ci < src_mat.cols; ++ci)
        {
            if(!clipMap[ri][ci]||!map[ri][ci])
            {
                continue;
            }

            size_t fi = clipMap[ri][ci]-1;
            size_t index0 = face_vertex_index[fi][0];
            size_t index1 = face_vertex_index[fi][1];
            size_t index2 = face_vertex_index[fi][2];

            vector<double> r(3, 0);
            for(int bi = 0; bi < 3; bi++)
            {
                vector3d p0(vertex_list[index0].x, vertex_list[index0].y, tri_mesh_vertex_R[index0][bi]);
                vector3d p1(vertex_list[index1].x, vertex_list[index1].y, tri_mesh_vertex_R[index1][bi]);
                vector3d p2(vertex_list[index2].x, vertex_list[index2].y, tri_mesh_vertex_R[index2][bi]);
                vector3d vp(ci, ri, 0);

                vp.CalPlanePointZ(p0, p1, p2);
                r[bi] = vp.z;

            }

            for(int bi = 0; bi < 3; bi++)
            {
                tar_mat.at<Vec3b>(ri + area_tar_position.y(),ci + area_tar_position.x())[bi] = min(max(src_mat.at<Vec3b>(ri,ci)[bi] + r[bi], 0.0), 255.0);
            }
        }
    }
    cout<<"Evaluate the mean-value interpolant done!" << endl;

    QTime stopTime = QTime::currentTime();
    int elapsed = startTime.msecsTo(stopTime);
    cout<<"Total time "<<elapsed<<" ms"<<endl;
}

double MainWindow::Tan_Compute(vector2d x ,vector2d dot1,vector2d dot2)
{
    double l,l1,l2;
    l = (dot1-dot2).Mod();
    l1 = (x-dot1).Mod();
    l2 = (x-dot2).Mod();
    double c = (l1*l1+l2*l2-l*l)/(2*l1*l2);
    return tan(cosh(c)/2);
}

bool MainWindow::PointinTriangle(vector2d A,vector2d B,vector2d C,vector2d P)
{
    vector2d v0 = C - A ;
    vector2d v1 = B - A ;
    vector2d v2 = P - A ;

    float dot00 = v0.Dot(v0);
    float dot01 = v0.Dot(v1) ;
    float dot02 = v0.Dot(v2) ;
    float dot11 = v1.Dot(v1) ;
    float dot12 = v1.Dot(v2) ;

    float inverDeno = 1 / (dot00 * dot11 - dot01 * dot01) ;

    float u = (dot11 * dot02 - dot01 * dot12) * inverDeno ;
    if (u < 0 || u > 1) // if u out of range, return directly
    {
        return false ;
    }

    float v = (dot00 * dot12 - dot01 * dot02) * inverDeno ;
    if (v < 0 || v > 1) // if v out of range, return directly
    {
        return false ;
    }

    return u + v <= 1 ;
}

void MainWindow::CalBoundPoint(std::vector<vector2d>& ROIBoundPointList,std::vector<vector2d>& ROI_)
{
    using namespace std;
    vector<pair<vector2d, vector2d>> lineList;
    int Split_num = 1;
    for(int i = 0; i<(ROI_.size() - 1 )/Split_num; i++)
    {

        lineList.push_back(make_pair(ROI_[i*Split_num], ROI_[Split_num*i+Split_num]));
    }

    //遍历所有边，并栅格化
    vector<vector2d> tmpPointList;
    for(size_t i = 0; i < lineList.size(); i++)
    {
        std::vector<vector2d> linePointList;
        RasterLine(lineList[i], linePointList);
        std::copy(linePointList.begin(), linePointList.end(), std::back_inserter(tmpPointList));
    }

    ROIBoundPointList.clear();
    ROIBoundPointList.push_back(ROI_[0]);
    for(size_t i = 0; i< tmpPointList.size(); i++)
    {
        //与最后一个值比较，去重
        if(!tmpPointList[i].Equel(ROIBoundPointList[ROIBoundPointList.size()-1]))
        {
            ROIBoundPointList.push_back(tmpPointList[i]);
        }

    }
    if(!ROIBoundPointList[0].Equel(ROIBoundPointList[ROIBoundPointList.size()-1]))
    {
        ROIBoundPointList.push_back(ROIBoundPointList[0]);
    }
}

void MainWindow::RasterLine(std::pair<vector2d, vector2d> line, std::vector<vector2d>& linePointList)
{
    using namespace std;
    vector2d vecLine = line.second-line.first;
    double lineLength = vecLine.Mod();
    double step = 1.0;

    vector<vector2d> tmpPointList;
    double curLength = 0;
    while(curLength<lineLength)
    {
        curLength = curLength + step;
        vector2d P = line.first + vecLine.Scalar(curLength/lineLength);
        P.x = (int)(P.x + 0.5);
        P.y = (int)(P.y + 0.5);
        tmpPointList.push_back(P);
    }

    //保存起点，不保存终点
    linePointList.push_back(line.first);
    for(size_t i = 0; i< tmpPointList.size(); i++)
    {
        //与最后一个值比较，去重
        if(!tmpPointList[i].Equel(linePointList[linePointList.size()-1]))
        {
            linePointList.push_back(tmpPointList[i]);
        }
    }
}
