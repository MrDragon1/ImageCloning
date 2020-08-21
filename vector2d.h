#ifndef VECTOR2D_H
#define VECTOR2D_H

#include <cmath>
#include <vector>
#include "opencv2//opencv.hpp"
#include "opencv2//core//core.hpp"
const double EPSILON  = 0.000001;


class vector2d
{
public:
    double x,y;
    vector2d()
    {
    }
    vector2d(cv::Point p)
    {
        x = p.x;
        y = p.y;
    }

    vector2d(double dx, double dy)
    {
        x = dx;
        y = dy;
    }

    // 矢量赋值
    void set(double dx, double dy)
    {
        x = dx;
        y = dy;
    }

    // 矢量相加
    vector2d operator + (const vector2d& v) const
    {
        return vector2d(x + v.x, y + v.y);
    }

    // 矢量相减
    vector2d operator - (const vector2d& v) const
    {
        return vector2d(x - v.x, y - v.y);
    }

    //矢量数乘
    vector2d Scalar(double c) const
    {
        return vector2d(c*x, c*y);
    }

    // 矢量点积
    double Dot(const vector2d& v) const
    {
        return x * v.x + y * v.y;
    }

    //向量的模
    double Mod() const
    {
        return sqrt(x * x + y * y);
    }

    bool Equel(const vector2d& v) const
    {
        if(abs(x-v.x) < EPSILON && abs(y-v.y)< EPSILON)
        {
            return true;
        }
        return false;
    }

    bool operator == (const vector2d& v) const
    {
        if (abs(x - v.x) < EPSILON && abs(y - v.y) < EPSILON)
        {
            return true;
        }
        return false;
    }

    bool operator < (const vector2d& v) const
    {
        if (abs(x - v.x) < EPSILON)
        {
            return y < v.y ? true : false;
        }
        return x<v.x ? true : false;
    }


};

#endif // VECTOR2D_H
