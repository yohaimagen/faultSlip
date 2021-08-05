/* disloc.c -- Computes surface displacements for dislocations in an elastic half-space.
   Based on code by Y. Okada.

   Version 1.2, 10/28/2000

   Record of revisions:

   Date          Programmer            Description of Change
   ====          ==========            =====================
   10/28/2000    Peter Cervelli        Removed seldom used 'reference station' option; improved
                                       detection of dip = integer multiples of pi/2.
   09/01/2000    Peter Cervelli        Fixed a bug that incorrectly returned an integer absolute value
                                       that created a discontinuity for dip angle of +-90 - 91 degrees.
                                       A genetically related bug incorrectly assigned a value of 1 to
                                       sin(-90 degrees).
   08/25/1998    Peter Cervelli        Original Code


*/

#include <math.h>
#include<stdio.h>
#include<stdlib.h>
#include <pthread.h>

#define DEG2RAD 0.017453292519943295L
#define PI2INV 0.15915494309189535L


typedef struct Args
{
    double *pEOutput;
    double *pNOutput;
    double *pZOutput;
    double *pModel;
    double *pECoords;
    double *pNCoords;
    double nu;
    int NumStat;
    int NumDisl;
}Args;



void Okada(double *pSS, double *pDS, double *pTS, double alp, double sd, double cd, double len, double wid,
           double dep, double X, double Y, double SS, double DS, double TS)
{
    double depsd, depcd, x, y, ala[2], awa[2], et, et2, xi, xi2, q2, r, r2, r3, p, q, sign;
    double a1, a3, a4, a5, d, ret, rd, tt, re, dle, rrx, rre, rxq, rd2, td, a2, req, sdcd, sdsd, mult;
    int j, k;

    ala[0] = len;
    ala[1] = 0.0;
    awa[0] = wid;
    awa[1] = 0.0;
    sdcd = sd * cd;
    sdsd = sd * sd;
    depsd = dep * sd;
    depcd = dep * cd;

    p = Y * cd + depsd;
    q = Y * sd - depcd;

    for (k = 0; k <= 1; k++)
    {
        et = p - awa[k];
        for (j = 0; j <= 1; j++)
        {
            sign = PI2INV;
            xi = X - ala[j];
            if (j + k == 1)
                sign = -PI2INV;
            xi2 = xi * xi;
            et2 = et * et;
            q2 = q * q;
            r2 = xi2 + et2 + q2;
            r = sqrt(r2);
            r3 = r * r2;
            d = et * sd - q * cd;
            y = et * cd + q * sd;
            ret = r + et;
            if (ret < 0.0)
                ret = 0.0;
            rd = r + d;
            if (q != 0.0)
                tt = atan(xi * et / (q * r));
            else
                tt = 0.0;
            if (ret != 0.0)
            {
                re = 1 / ret;
                dle = log(ret);
            }
            else
            {
                re = 0.0;
                dle = -log(r - et);
            }
            rrx = 1 / (r * (r + xi));
            rre = re / r;
            if (cd == 0.0)
            {
                rd2 = rd * rd;
                a1 = -alp / 2 * xi * q / rd2;
                a3 = alp / 2 * (et / rd + y * q / rd2 - dle);
                a4 = -alp * q / rd;
                a5 = -alp * xi * sd / rd;
            }
            else
            {
                td = sd / cd;
                x = sqrt(xi2 + q2);
                if (xi == 0.0)
                    a5 = 0;
                else
                    a5 = alp * 2 / cd * atan( (et * (x + q * cd) + x * (r + x) * sd) / (xi * (r + x) * cd) );

                a4 = alp / cd * (log(rd) - sd * dle);
                a3 = alp * (y / rd / cd - dle) + td * a4;
                a1 = -alp / cd * xi / rd - td * a5;
            }

            a2 = -alp * dle - a3;
            req = rre * q;
            rxq = rrx * q;

            if (SS != 0)
            {
                mult = sign * SS;
                pSS[0] -= mult * (req * xi + tt + a1 * sd);
                pSS[1] -= mult * (req * y + q * cd * re + a2 * sd);
                pSS[2] -= mult * (req * d + q * sd * re + a4 * sd);
            }

            if (DS != 0)
            {
                mult = sign * DS;
                pDS[0] -= mult *(q / r - a3 * sdcd);
                pDS[1] -= mult * (y * rxq + cd * tt - a1 * sdcd);
                pDS[2] -= mult * (d * rxq + sd * tt - a5 * sdcd);
            }
            if (TS != 0)
            {
                mult = sign * TS;
                pTS[0] += mult * (q2 * rre - a3 * sdsd);
                pTS[1] += mult * (-d * rxq - sd * (xi * q * rre - tt) - a1 * sdsd);
                pTS[2] += mult * (y * rxq + cd * (xi * q * rre - tt) - a5 * sdsd);
            }
        }
    }
}

void c_disloc(double *pEOutput, double *pNOutput, double *pZOutput, double *pModel, double *pECoords, double *pNCoords, double nu, int NumStat, int NumDisl)
{
    int i,j, dIndex;
    double sd, cd, Angle, cosAngle, sinAngle, SS[3],DS[3],TS[3], x, y;


    /*Loop through dislocations*/

    for (i=0; i< NumDisl; i++)
    {
        dIndex=i*10;

        cd = cos(pModel[dIndex+3] * DEG2RAD);
        sd = sin(pModel[dIndex+3] * DEG2RAD);

        if (pModel[0]<0 || pModel[1]<0 || pModel[2]<0 || (pModel[2]-sin(pModel[3]*DEG2RAD)*pModel[1])<-1e-12)
        {
            printf("Warning: model %d is not physical. It will not contribute to the deformation.\n",i+1);
            continue;
        }

        if (fabs(cd)<2.2204460492503131e-16)
        {
            cd=0;
            if (sd>0)
                sd=1;
            else
                sd=0;
        }

        Angle = -(90 - pModel[dIndex+4]) * DEG2RAD;
        cosAngle = cos(Angle);
        sinAngle = sin(Angle);

        /*Loop through stations*/

        for(j=0; j < NumStat; j++)
        {
            SS[0] = SS[1] = SS[2] = 0;
            DS[0] = DS[1] = DS[2] = 0;
            TS[0] = TS[1] = TS[2] = 0;


            Okada(&SS[0],&DS[0],&TS[0],1 - 2 * nu,sd,cd,pModel[dIndex],pModel[dIndex+1],pModel[dIndex+2],
                  cosAngle * (pECoords[j] - pModel[dIndex+5]) - sinAngle * (pNCoords[j] - pModel[dIndex+6]) +  0.5 * pModel[dIndex],
                  sinAngle * (pECoords[j] - pModel[dIndex+5]) + cosAngle * (pNCoords[j] - pModel[dIndex+6]),
                  pModel[dIndex+7], pModel[dIndex+8], pModel[dIndex+9]);


            if (pModel[dIndex+7])
            {
                x=SS[0];
                y=SS[1];
                SS[0] = cosAngle * x + sinAngle * y;
                SS[1] = -sinAngle * x + cosAngle * y;
//                printf("%lf %lf %lf\n", SS[0], SS[1], SS[2]);
                pEOutput[j]+=SS[0];
                pNOutput[j]+=SS[1];
                pZOutput[j]+=SS[2];
//                printf("%lf %lf %lf\n", pEOutput[j], pNOutput[j], pZOutput[j]);

            }
            if (pModel[dIndex+8])
            {
                x=DS[0];
                y=DS[1];
                DS[0] = cosAngle * x + sinAngle * y;
                DS[1] = -sinAngle * x + cosAngle * y;
                pEOutput[j]+=DS[0];
                pNOutput[j]+=DS[1];
                pZOutput[j]+=DS[2];
            }

            if (pModel[dIndex+9])
            {
                x=TS[0];
                y=TS[1];
                TS[0] = cosAngle * x + sinAngle * y;
                TS[1] = -sinAngle * x + cosAngle * y;
                pEOutput[j]+=TS[0];
                pNOutput[j]+=TS[1];
                pZOutput[j]+=TS[2];

            }
        }
    }
}


void c_disloc_1d(double *pEOutput, double *pNOutput, double *pZOutput, double *pModel, double *pECoords, double *pNCoords, double nu, int NumStat, int NumDisl)
{
    int i,j, dIndex;
    double sd, cd, Angle, cosAngle, sinAngle, SS[3],DS[3],TS[3], x, y;


    /*Loop through dislocations*/

    for (i=0; i< NumDisl; i++)
    {
        dIndex=i*10;

        cd = cos(pModel[dIndex+3] * DEG2RAD);
        sd = sin(pModel[dIndex+3] * DEG2RAD);

        if (pModel[0]<0 || pModel[1]<0 || pModel[2]<0 || (pModel[2]-sin(pModel[3]*DEG2RAD)*pModel[1])<-1e-12)
        {
            printf("Warning: model %d is not physical. It will not contribute to the deformation.\n",i+1);
            continue;
        }

        if (fabs(cd)<2.2204460492503131e-16)
        {
            cd=0;
            if (sd>0)
                sd=1;
            else
                sd=0;
        }

        Angle = -(90 - pModel[dIndex+4]) * DEG2RAD;
        cosAngle = cos(Angle);
        sinAngle = sin(Angle);

        /*Loop through stations*/

        for(j=0; j < NumStat; j++)
        {
            SS[0] = SS[1] = SS[2] = 0;
            DS[0] = DS[1] = DS[2] = 0;
            TS[0] = TS[1] = TS[2] = 0;


            Okada(&SS[0],&DS[0],&TS[0],1 - 2 * nu,sd,cd,pModel[dIndex],pModel[dIndex+1],pModel[dIndex+2],
                  cosAngle * (pECoords[i] - pModel[dIndex+5]) - sinAngle * (pNCoords[i] - pModel[dIndex+6]) +  0.5 * pModel[dIndex],
                  sinAngle * (pECoords[i] - pModel[dIndex+5]) + cosAngle * (pNCoords[i] - pModel[dIndex+6]),
                  pModel[dIndex+7], pModel[dIndex+8], pModel[dIndex+9]);


            if (pModel[dIndex+7])
            {
                x=SS[0];
                y=SS[1];
                SS[0] = cosAngle * x + sinAngle * y;
                SS[1] = -sinAngle * x + cosAngle * y;
//                printf("%lf %lf %lf\n", SS[0], SS[1], SS[2]);
                pEOutput[i]+=SS[0];
                pNOutput[i]+=SS[1];
                pZOutput[i]+=SS[2];
//                printf("%lf %lf %lf\n", pEOutput[j], pNOutput[j], pZOutput[j]);

            }
            if (pModel[dIndex+8])
            {
                x=DS[0];
                y=DS[1];
                DS[0] = cosAngle * x + sinAngle * y;
                DS[1] = -sinAngle * x + cosAngle * y;
                pEOutput[i]+=DS[0];
                pNOutput[i]+=DS[1];
                pZOutput[i]+=DS[2];
            }

            if (pModel[dIndex+9])
            {
                x=TS[0];
                y=TS[1];
                TS[0] = cosAngle * x + sinAngle * y;
                TS[1] = -sinAngle * x + cosAngle * y;
                pEOutput[i]+=TS[0];
                pNOutput[i]+=TS[1];
                pZOutput[i]+=TS[2];

            }
        }
    }
}

void *disloc_m(void *in_arg)
{

    Args *args;
    args = in_arg;
//    printf("%p -- %p -- %p -- %p -- %p -- %p\n", args->pEOutput,  args->pNOutput, args->pZOutput, args->pModel, args->pECoords, args->pNCoords);
//    printf("%d\n", args->NumDisl);
//    printf("1");
//        for(int j = 0; j < 10; j++)
//        {
//            printf("%f -- ", args->pModel[j]);
//        }
//        printf("\n");
    c_disloc(args->pEOutput, args->pNOutput, args->pZOutput, args->pModel,
           args->pECoords, args->pNCoords, args->nu, args->NumStat, args->NumDisl);
    return NULL;
}

void c_disloc_m(double *pEOutput, double *pNOutput, double *pZOutput, double *pModel, double *pECoords, double *pNCoords, double nu, int NumStat, int NumDisl, int thread_num)
{
//    printf("%p -- %p -- %p -- %p -- %p -- %p\n", pEOutput,  pNOutput, pZOutput, pModel, pECoords, pNCoords);
    pthread_t thread_arr[thread_num];
    Args thread_args[thread_num];
    int data_p = 0;
    int step = NumStat / thread_num;
    for(int i=0; i < thread_num; i++)
    {

        thread_args[i].pEOutput = pEOutput + data_p;
        thread_args[i].pNOutput = pNOutput + data_p;
        thread_args[i].pZOutput = pZOutput + data_p;


        thread_args[i].pModel = (double *)malloc(10 * NumDisl * sizeof(double));
        for(int j = 0; j < 10 * NumDisl; j++)
        {

            thread_args[i].pModel[j] = pModel[j];
        }
        thread_args[i].pECoords = pECoords + data_p;
        thread_args[i].pNCoords = pNCoords + data_p;
        thread_args[i].nu = nu;
        if(i != thread_num - 1)
        {
            thread_args[i].NumStat = step;
        }
        else
        {
            thread_args[i].NumStat = NumStat - data_p;
        }
        thread_args[i].NumDisl = NumDisl;
        int err = pthread_create(&thread_arr[i], NULL, disloc_m, (void*)&thread_args[i]);
        if(err != 0)
        {
            printf("error %d im creating thread", err);
        }
        data_p += step;
    }
   for(int k = 0; k < thread_num; k++)
   {
        pthread_join(thread_arr[k], NULL);
   }
   for(int k = 0; k < thread_num; k++)
   {
        free(thread_args[k].pModel);
   }

}

//void Okada85(float* E, float* N, float* e, float* n, float* z, int cols, int rows, double alp, double sd, double cd, double len, double wid,
//           double dep, double X, double Y, double SS, double DS, double TS)
//{
//    int index = 0;
//    double SS[3], DS[3], TS[3]
//    for (int i = 0;, i < cols; i++)
//    {
//        for (int j = 0; j < rows; j++)
//        {
//            SS[0] = SS[1] = SS[2] = 0;
//            DS[0] = DS[1] = DS[2] = 0;
//            TS[0] = TS[1] = TS[2] = 0;
//
//            Okada()
//        }
//    }
//}

//int main()
//{
//    double pOutput[3], pModel[10], pCoords[2], nu;
//    int NumStat, NumDisl;
//    nu = 0.25;
//    NumStat = 1;
//    NumDisl = 1;
//    pModel[0] = 20;
//    pModel[1] = 5;
//    pModel[2] = 4.6985;
//    pModel[3] = 70;
//    pModel[4] = 0;
//    pModel[5] = 0;
//    pModel[6] = 0;
//    pModel[7] = 1;
//    pModel[8] = 0;
//    pModel[9] = 0;
//    pCoords[0] = -21.710;
//    pCoords[1] = -20;
//	pOutput[0] = pOutput[1] = pOutput[2] = 0;
//    Disloc(pOutput, pModel, pCoords, nu, NumStat, NumDisl);
//    printf("%lf %lf %lf\n", pOutput[0], pOutput[1], pOutput[2]);
//}

