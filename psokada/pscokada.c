
/*   Copyright (C) 2015 by Camilla Cattania and Fahad Khalid.
 *
 *   This file is part of CRS.
 *
 *   CRS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   CRS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CRS.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <math.h>
#include <pthread.h>

#include "pscokada.h"
#include "defines.h"
#include "util1.h"
#include "dc3d.h"


typedef struct Args
{
    double x1;
    double y1;
    double z1;
    double strike;
    double dip;
    double L;
    double W;
    double slip_strike;
    double slip_dip;
    double open;
    double *x2;
    double *y2;
    double *z2;
    double *s;
    double lame_lambda;
    double mu;
    int pop_mum;
}Args;

void *stress(void *in_arg)
{

    Args *args;
    args = in_arg;
    for(int i = 0; i < args->pop_mum; i++)
    {
        c_okada_stress(args->x1, args->y1, args->z1, args->strike, args->dip, args->L, args->W, args->slip_strike,
         args->slip_dip, args->open, args->x2[i], args->y2[i], args->z2[i], args->s + (i * 9), args->lame_lambda, args->mu);
    }
    return NULL;
}

void c_okada_stress_thread(double x1, double y1, double z1, double strike, double dip, double L, double W, double slip_strike, double slip_dip, double open,
		double *x2, double *y2, double *z2, double *s,
		double lame_lambda, double mu, int pop_num, int thread_num)
{

    int data_p = 0;
    int step = pop_num / thread_num;
    if(step == 0)
    {
        step = 1;
        thread_num = pop_num;
    }
    pthread_t thread_arr[thread_num];
    Args thread_args[thread_num];
    for(int i=0; i < thread_num; i++)
    {

        thread_args[i].x1 = x1;
        thread_args[i].y1 = y1;
        thread_args[i].z1 = z1;
        thread_args[i].x1 = x1;
        thread_args[i].strike = strike;
        thread_args[i].dip = dip;
        thread_args[i].L = L;
        thread_args[i].W = W;
        thread_args[i].slip_strike = slip_strike;
        thread_args[i].slip_dip = slip_dip;
        thread_args[i].open = open;
        thread_args[i].x2 = x2 + data_p;
        thread_args[i].y2 = y2 + data_p;
        thread_args[i].z2 = z2 + data_p;
        thread_args[i].s = s + (data_p * 9);
        thread_args[i].lame_lambda = lame_lambda;
        thread_args[i].mu = mu;

        if(i != thread_num - 1)
        {
            thread_args[i].pop_mum = step;
        }
        else
        {
            thread_args[i].pop_mum = pop_num - data_p;
        }
        int err = pthread_create(&thread_arr[i], NULL, stress, (void*)&thread_args[i]);
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
}


void c_okada_stress(double x1, double y1, double z1, double strike1, double dip1, double L, double W, double slip_strike, double slip_dip, double open,
		double x2, double y2, double z2, double *s,
		double lame_lambda, double mu)
{

	pscokada(x1, y1, z1, strike1, dip1, L, W, slip_strike, slip_dip, open, x2, y2, z2, &s[0], &s[4], &s[8], &s[1], &s[2], &s[5], lame_lambda, mu);
	s[3] = s[1];
	s[6] = s[2];
	s[7] = s[5];

}

void pscokada(double x1, double y1, double z1, double strike1, double dip1, double L, double W, double slip_strike, double slip_dip, double open,
		double x2, double y2, double z2, double *sxx, double *syy, double *szz, double *sxy, double *syz, double *szx,
		double lame_lambda, double mu) {

/*
 * Input:
 *  strike, dip: patch strike and dip, in radians.
 *  L, W: patch length and width
 *  slip_strike, slip_dip, open: slip along strike, along dip, and opening.
 *
 *  x1, y1, z1 are coordinates of center top of the patch:  -------X-------
 * 														   \             \
 *														    \             \
 *														     \             \
 *														      ---------------
 *
 *  x2, y2, z2: receiver coordinates.
 *  lambda, mu: lame' parameters.
 *
 *
 * Output:
 *  sxx, sxy, ... : components of the stress tensor.

 *
 * NB: x is northward, y is eastward, and z is downward.
 *
 */

	double alpha = (lame_lambda+mu)/(lame_lambda + 2*mu);
	double depth, stk, sin_stk, cos_stk, sin_2stk, di, csdi, ssdi;
	double DISL1, DISL2, DISL3, AL1, AL2, AW1, AW2, X, Y, Z, UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ;
	double strain1, strain2, strain3, strain4, strain5, strain6, eii, dum1, dum2;
	int IRET;

	// kilometer to meter
	x1 *= KM2M;
	y1 *= KM2M;
	z1 *= KM2M;
	x2 *= KM2M;
	y2 *= KM2M;
	z2 *= KM2M;
	L  *= KM2M;
	W  *= KM2M;

	stk = strike1;
	cos_stk = cos(stk);
	sin_stk = sin(stk);
	sin_2stk = sin(2.0 * stk);

	di = dip1;
	csdi = cos(di);
	ssdi = sin(di);

	/* rake: gegen Uhrzeigersinn  & slip_z: positiv in Richtung Tiefe */
	DISL1 = slip_strike; //cos(DEG2RAD * rake1) * slip;
	DISL2 = slip_dip;    //sin(DEG2RAD * rake1) * slip;
	DISL3 = open;

	// neg und pos Werte? sind das nicht alongdip und againstdip längen?
	AL1 = -0.5 * L;
	AL2 =  0.5 * L;
	AW1=-W;		//top of patch is passed to function.
	AW2=0;

	/*  transform from Aki's to Okada's system */
	X = (x2 - x1) * cos_stk + (y2 - y1) * sin_stk;
	Y = (x2 - x1) * sin_stk - (y2 - y1) * cos_stk;
	Z = -z2;
	depth = z1;
	/* z2 corresponds to the recording depth zrec */
	/* z1 corresponds to the depth of slip (reference depth: zref) */

	DC3D(alpha, X, Y, Z, depth, RAD2DEG*di, AL1, AL2, AW1, AW2, DISL1, DISL2, DISL3, &UX, &UY, &UZ, &UXX, &UYX, &UZX, &UXY, &UYY, &UZY, &UXZ, &UYZ, &UZZ, &IRET);

	/* transform from Okada's to Aki's system */
	strain1 = UXX * cos_stk * cos_stk + UYY * sin_stk * sin_stk + 0.5 * (UXY + UYX) * sin_2stk;
	strain2 = UXX * sin_stk * sin_stk + UYY * cos_stk * cos_stk - 0.5 * (UXY + UYX) * sin_2stk;
	strain3 = UZZ;
	strain4 = 0.5 * (UXX - UYY) * sin_2stk - 0.5 * (UXY + UYX) * cos(2.0 * stk);
	dum1 = -0.5 * (UZX + UXZ);
	dum2 = 0.5 * (UYZ + UZY);
	strain5 = dum1 * sin_stk + dum2 * cos_stk;
	strain6 = dum1 * cos_stk - dum2 * sin_stk;
	eii = strain1 + strain2 + strain3;

	dum1 = lame_lambda * eii;
	dum2 = 2.0 * mu;
	*sxx = dum1 + dum2 * strain1;
	*syy = dum1 + dum2 * strain2;
	*szz = dum1 + dum2 * strain3;
	*sxy =        dum2 * strain4;
	*syz =        dum2 * strain5;
	*szx =        dum2 * strain6;
}
