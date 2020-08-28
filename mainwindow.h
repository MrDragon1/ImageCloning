#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMouseEvent>
#include <QPainter>
#include <QFileDialog>
#include <QLabel>
#include <QTime>
#include <QScrollArea>
#include <QScrollBar>
#include <vector>
#include <map>
#include "opencv2//opencv.hpp"
#include "opencv2//core//core.hpp"
#include "opencv2//highgui//highgui.hpp"
#include "opencv2//imgproc//types_c.h"
#include <numeric>

#include "vector3d.h"
#include "vector2d.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point CDTPoint;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void mousePressEvent(QMouseEvent*);
    void mouseMoveEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);
    void paintEvent(QPaintEvent* event);
    void paint(QImage& Img);
    QImage cvMat2QImage(const cv::Mat& mat);
    void MVC_Compute();
    void MVC_Compute_Optimized();
    void MVC_Compute_Optimized_HBS();
    double Angle_Compute(vector2d,vector2d,vector2d);
    bool PointinTriangle(vector2d,vector2d,vector2d,vector2d);
    void CalBoundPoint(std::vector<vector2d>& ROIBoundPointList,std::vector<vector2d>& ROI_);
    void RasterLine(std::pair<vector2d, vector2d> line, std::vector<vector2d>& linePointList);
    void HierarchicalBoundarySampling(vector2d point);
    void HierarchyBoundary(int index,std::vector<vector2d>& newROI, std::vector<vector2d>& MeshPoint,double dist,double ang);

private slots:
    void on_Info_btn_clicked();

    void on_Load_source_btn_clicked();

    void on_Set_btn_clicked();

    void on_Reset_btn_clicked();

    void on_Load_target_btn_clicked();

    void on_Clone_btn_clicked();

    void on_Reclone_btn_clicked();

private:
    Ui::MainWindow *ui;
    QImage src_img,tar_img,src_img_copy,tar_img_copy;
    QPoint startpoint,endpoint;//画笔功能所需变量
    std::vector<cv::Point> line; //存储画笔所选区域
    std::vector<vector2d> ROI;
    std::vector<std::vector<bool>> map;//存储图像选择的区域
    std::vector<int> newROI;
    cv::Mat src_mat,tar_mat,res_mat,tar_mat_copy,src_mat_copy;//Mat类型图片
    QPoint srcpoint,tarpoint,relative_position;//原图片和目标图片位置，光标与图片的相对位置
    bool IsDraw,Moved,Movearea;//画笔模式，是否已经加入选择的区域，光标是否点击在area图片内
    QLabel* src_label,*tar_label;
};

#endif // MAINWINDOW_H
