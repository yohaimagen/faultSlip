#include <math.h>
#include<stdio.h>
#include<stdlib.h>
#include <stdbool.h>


void c_dist(double *p1, double *p2, double *p3, double *p4, double *out)
{
//    printf('%f', p1[0]);
    double u[] = {p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
    double v[] = {p3[0] - p4[0], p3[1] - p4[1], p3[2] - p4[2]};
    double w[] = {p2[0] - p4[0], p2[1] - p4[1], p2[2] - p4[2]};

    double a, b, c, d, e, D, sD, tD, epsilon, sN, tN, tc, sc, dP0, dP1, dP2;

    a = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    b = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    c = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    d = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    e = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
    D = a * c - b * b;
    sD = D;
    tD = D;

    epsilon = 1e-8;
    if(D < epsilon)
    {
        sN = 0.0;
        sD = 1.0;
        tN = e;
        tD = c;
    }
    else
    {
        sN = b * e - c * d;
        tN = a * e - b * d;
        if(sN < 0.0)
        {
            sN = 0.0;
            tN = e;
            tD = c;
        }
        else if(sN > sD)
        {
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }
    if(tN < 0.0)
    {
        tN = 0.0;
        if(-d < 0.0)
        {
            sN = 0.0;
        }
        else if(-d > a)
        {
            sN = sD;
        }
        else
        {
            sN = -d;
            sD = a;
        }
    }
    else if(tN > tD)
    {
        tN = tD;
        if((-d + b) < 0.0)
        {
            sN = 0.0;
        }
        else if((-d + b) > a)
        {
            sN = sD;
        }
        else
        {
            sN = (-d + b);
            sD = a;
        }
    }
    if(fabs(sN) < epsilon)
    {
        sc = 0.0;
    }
    else
    {
        sc = sN / sD;
    }
    if(fabs(tN) < epsilon)
    {
        tc = 0.0;
    }
    else
    {
        tc = tN / tD;
    }
    dP0 = w[0] + (sc * u[0]) - (tc * v[0]);
    dP1 = w[1] + (sc * u[1]) - (tc * v[1]);
    dP2 = w[2] + (sc * u[2]) - (tc * v[2]);
    out[0] = sqrt(dP0 * dP0 + dP1 * dP1 + dP2 * dP2);
}

double r_dist(double *p1, double *p2, double *p3, double *p4)
{
    double u[] = {p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
    double v[] = {p3[0] - p4[0], p3[1] - p4[1], p3[2] - p4[2]};
    double w[] = {p2[0] - p4[0], p2[1] - p4[1], p2[2] - p4[2]};

    double a, b, c, d, e, D, sD, tD, epsilon, sN, tN, tc, sc, dP0, dP1, dP2;

    a = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    b = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    c = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    d = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    e = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
    D = a * c - b * b;
    sD = D;
    tD = D;

    epsilon = 1e-8;
    if(D < epsilon)
    {
        sN = 0.0;
        sD = 1.0;
        tN = e;
        tD = c;
    }
    else
    {
        sN = b * e - c * d;
        tN = a * e - b * d;
        if(sN < 0.0)
        {
            sN = 0.0;
            tN = e;
            tD = c;
        }
        else if(sN > sD)
        {
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }
    if(tN < 0.0)
    {
        tN = 0.0;
        if(-d < 0.0)
        {
            sN = 0.0;
        }
        else if(-d > a)
        {
            sN = sD;
        }
        else
        {
            sN = -d;
            sD = a;
        }
    }
    else if(tN > tD)
    {
        tN = tD;
        if((-d + b) < 0.0)
        {
            sN = 0.0;
        }
        else if((-d + b) > a)
        {
            sN = sD;
        }
        else
        {
            sN = (-d + b);
            sD = a;
        }
    }
    if(fabs(sN) < epsilon)
    {
        sc = 0.0;
    }
    else
    {
        sc = sN / sD;
    }
    if(fabs(tN) < epsilon)
    {
        tc = 0.0;
    }
    else
    {
        tc = tN / tD;
    }
    dP0 = w[0] + (sc * u[0]) - (tc * v[0]);
    dP1 = w[1] + (sc * u[1]) - (tc * v[1]);
    dP2 = w[2] + (sc * u[2]) - (tc * v[2]);
    return sqrt(dP0 * dP0 + dP1 * dP1 + dP2 * dP2);
}

void c_neighbors(double *p1, double *p2, double *p3, double *p4, double *t1, double *t2, double *t3, double *t4, double *out)
{
    double *ar1[] = {p1, p2, p3, p4};
    double *ar2[] = {t1, t2, t3, t4};
    double dist[16];
    int len = 4;

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            dist[i * len + j] = r_dist(ar1[i], ar1[(i + 1) % 4], ar2[j], ar2[(j + 1) % 4]);
        }
    }
    bool neighbors = false;
    for(int i = 0; i < 16; i++)
    {
        neighbors = neighbors || (dist[i] <= 1.0);
    }
    if(neighbors)
    {
        out[0] = 1;
    }
    else
    {
        out[0] = 0;
    }
}
