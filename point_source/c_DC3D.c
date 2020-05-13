

#include <math.h>
#include <pthread.h>
#include <stdio.h>

#include "c_DC3D.h"
#include "defines.h"

typedef struct Args
{
    float x1;
    float y1;
    float z1;
    float strike;
    float dip;
    float moment_strike;
    float moment_dip;
    float moment_inflation;
    float moment_open;
    float *x2;
    float *y2;
    float *z2;
    float *s;
    float lame_lambda;
    float mu;
    int pop_mum;
}Args;

void *strain(void *in_arg)
{

    Args *args;
    args = in_arg;
    for(int i = 0; i < args->pop_mum; i++)
    {
        c_point_source_strain(args->x1, args->y1, args->z1, args->strike, args->dip, args->moment_strike, args->moment_dip, args->moment_inflation,
         args->moment_open, args->x2[i], args->y2[i], args->z2[i], args->s + (i * 9), args->lame_lambda, args->mu);
    }
    return NULL;
}

void *stress(void *in_arg)
{

    Args *args;
    args = in_arg;
    for(int i = 0; i < args->pop_mum; i++)
    {
        c_point_source_stress(args->x1, args->y1, args->z1, args->strike, args->dip, args->moment_strike, args->moment_dip, args->moment_inflation,
         args->moment_open, args->x2[i], args->y2[i], args->z2[i], args->s + (i * 9), args->lame_lambda, args->mu);
    }
    return NULL;
}

void c_ps_stress(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float *x2, float *y2, float *z2, float *s, float lame_lambda, float mu, int pop_num, int thread_num)
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
        thread_args[i].strike = strike1;
        thread_args[i].dip = dip1;
        thread_args[i].moment_strike = moment_strike;
        thread_args[i].moment_dip = moment_dip;
        thread_args[i].moment_inflation = moment_inflation;
        thread_args[i].moment_open = moment_open;
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

void c_ps_strain(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float *x2, float *y2, float *z2, float *s, float lame_lambda, float mu, int pop_num, int thread_num)
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
        thread_args[i].strike = strike1;
        thread_args[i].dip = dip1;
        thread_args[i].moment_strike = moment_strike;
        thread_args[i].moment_dip = moment_dip;
        thread_args[i].moment_inflation = moment_inflation;
        thread_args[i].moment_open = moment_open;
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
        int err = pthread_create(&thread_arr[i], NULL, strain, (void*)&thread_args[i]);
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



void stress_point_source(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *sxx, float *syy, float *szz, float *sxy, float *syz, float *szx,
		float lame_lambda, float mu)
{

/*
 * Input:
 *  dip: point source dip, in radians.
 *  moment_strike, moment_dip, moment_inflation moment_open: moment released  along strike, along dip, along opening and inflation.
 *
 *  x1, y1, z1 are coordinates of the point source:
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

	float alpha = (lame_lambda + mu) / (lame_lambda + 2 * mu);
	float depth, stk, sin_stk, cos_stk, sin_2stk, di, deg_di, csdi, ssdi, Angle, sinAngle, cosAngle;
	float POT1, POT2, POT3, POT4, X, Y, Z, UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ;
	float strain1, strain2, strain3, strain4, strain5, strain6, eii, dum1, dum2;
	int IRET;

	// kilometer to meter
	x1 *= KM2M;
	y1 *= KM2M;
	z1 *= KM2M;
	x2 *= KM2M;
	y2 *= KM2M;
	z2 *= KM2M;


	stk = strike1;
	cos_stk = cos(stk);
	sin_stk = sin(stk);
	sin_2stk = sin(2.0 * stk);

	di = dip1;
	deg_di = di * RAD2DEG;
	csdi = cos(di);
	ssdi = sin(di);

	/* rcalculate potensy out of the moment*/
	POT1 = moment_strike / mu;
	POT2 = moment_dip / mu;
	POT3 = moment_open / lame_lambda;
	POT4 = moment_inflation / mu;


	Angle = -((PI / 2) - strike1);
    cosAngle = cos(Angle);
    sinAngle = sin(Angle);


	/*  transform from cartesian to Okada's system */
	X = (x2 - x1) * cosAngle - (y2 - y1) * sinAngle;
	Y = (x2 - x1) * sinAngle + (y2 - y1) * cosAngle;
	Z = -z2;
	depth = z1;
	/* z2 corresponds to the recording depth zrec */
	/* z1 corresponds to the depth of slip (reference depth: zref) */

	DC3D0(&alpha, &X, &Y, &Z, &depth, &deg_di, &POT1, &POT2, &POT3, &POT4, &UX, &UY, &UZ, &UXX, &UYX, &UZX, &UXY, &UYY, &UZY, &UXZ, &UYZ, &UZZ, &IRET);

	/* transform from Okada's to cartesian system */




	strain1 = UXX * cosAngle * cosAngle + UYY * sinAngle * sinAngle + (UXY + UYX) * cosAngle * sinAngle;
	strain2 = UXX * sinAngle * sinAngle + UYY * cosAngle * cosAngle -  (UXY + UYX) * cosAngle * sinAngle;
	strain3 = UZZ;
	strain4 = 0.5 * (cos(2.0 * Angle) * (UXY + UYX)) + cosAngle * sinAngle * (UYY - UXX);
	dum1 = 0.5 * (UZX + UXZ);
	dum2 = 0.5 * (UYZ + UZY);
	strain5 = - dum1 * sinAngle + dum2 * cosAngle;
	strain6 = dum1 * cosAngle + dum2 * sinAngle;

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

void strain_point_source(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *sxx, float *syy, float *szz, float *sxy, float *syz, float *szx,
		float lame_lambda, float mu)
{

/*
 * Input:
 *  dip: point source dip, in radians.
 *  moment_strike, moment_dip, moment_inflation moment_open: moment released  along strike, along dip, along opening and inflation.
 *
 *  x1, y1, z1 are coordinates of the point source:
 *
 *  x2, y2, z2: receiver coordinates.
 *  lambda, mu: lame' parameters.
 *
 *
 * Output:
 *  sxx, sxy, ... : components of the strain tensor.

 *
 * NB: x is northward, y is eastward, and z is downward.
 *
 */

	float alpha = (lame_lambda + mu) / (lame_lambda + 2 * mu);
	float depth, di, deg_di, csdi, ssdi, Angle, sinAngle, cosAngle;
	float POT1, POT2, POT3, POT4, X, Y, Z, UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ;
	float dum1, dum2;
	int IRET;

	// kilometer to meter
	x1 *= KM2M;
	y1 *= KM2M;
	z1 *= KM2M;
	x2 *= KM2M;
	y2 *= KM2M;
	z2 *= KM2M;



	di = dip1;
	deg_di = di * RAD2DEG;
	csdi = cos(di);
	ssdi = sin(di);

	/* rcalculate potensy out of the moment*/
	POT1 = moment_strike / mu;
	POT2 = moment_dip / mu;
	POT3 = moment_open / lame_lambda;
	POT4 = moment_inflation / mu;


	Angle = -((PI / 2) - strike1);
    cosAngle = cos(Angle);
    sinAngle = sin(Angle);


	/*  transform from cartesian to Okada's system */
	X = (x2 - x1) * cosAngle - (y2 - y1) * sinAngle;
	Y = (x2 - x1) * sinAngle + (y2 - y1) * cosAngle;
	Z = -z2;
	depth = z1;
	/* z2 corresponds to the recording depth zrec */
	/* z1 corresponds to the depth of slip (reference depth: zref) */

	DC3D0(&alpha, &X, &Y, &Z, &depth, &deg_di, &POT1, &POT2, &POT3, &POT4, &UX, &UY, &UZ, &UXX, &UYX, &UZX, &UXY, &UYY, &UZY, &UXZ, &UYZ, &UZZ, &IRET);

	/* transform from Okada's to cartesian system */


	*sxx = UXX * cosAngle * cosAngle + UYY * sinAngle * sinAngle + (UXY + UYX) * cosAngle * sinAngle;
	*syy = UXX * sinAngle * sinAngle + UYY * cosAngle * cosAngle -  (UXY + UYX) * cosAngle * sinAngle;
	*szz = UZZ;
	*sxy = 0.5 * (cos(2.0 * Angle) * (UXY + UYX)) + cosAngle * sinAngle * (UYY - UXX);
	dum1 = 0.5 * (UZX + UXZ);
	dum2 = 0.5 * (UYZ + UZY);
	*syz = - dum1 * sinAngle + dum2 * cosAngle;
	*szx = dum1 * cosAngle + dum2 * sinAngle;
}

void disp_point_source(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *sx, float *sy, float *sz,
		float lame_lambda, float mu)
{

/*
 * Input:
 *  dip: point source dip, in radians.
 *  moment_strike, moment_dip, moment_inflation moment_open: moment released  along strike, along dip, along opening and inflation.
 *
 *  x1, y1, z1 are coordinates of the point source:
 *
 *  x2, y2, z2: receiver coordinates.
 *  lambda, mu: lame' parameters.
 *
 *
 * Output:
 *  sx, sy, sz: displacement component.

 *
 * NB: x is northward, y is eastward, and z is downward.
 *
 */

	float alpha = (lame_lambda + mu) / (lame_lambda + 2 * mu);
	float depth, di, deg_di, csdi, ssdi, Angle, cosAngle, sinAngle;
	float POT1, POT2, POT3, POT4, X, Y, Z, UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ;
	int IRET;

	// kilometer to meter
	x1 *= KM2M;
	y1 *= KM2M;
	z1 *= KM2M;
	x2 *= KM2M;
	y2 *= KM2M;
	z2 *= KM2M;




	di = dip1;
	deg_di = di * RAD2DEG;
	csdi = cos(di);
	ssdi = sin(di);

	/* rcalculate potensy out of the moment*/
	POT1 = moment_strike / mu;
	POT2 = moment_dip / mu;
	POT3 = moment_open / lame_lambda;
	POT4 = moment_inflation / mu;

	Angle = -((PI / 2) - strike1);
    cosAngle = cos(Angle);
    sinAngle = sin(Angle);


	/*  transform from cartesian to Okada's system */
	X = (x2 - x1) * cosAngle - (y2 - y1) * sinAngle;
	Y = (x2 - x1) * sinAngle + (y2 - y1) * cosAngle;
	Z = -z2;
	depth = z1;
	/* z2 corresponds to the recording depth zrec */
	/* z1 corresponds to the depth of slip (reference depth: zref) */



//	printf("alpha: %f\nX: %f\nY: %f\nZ: %f\ndepth: %f\ndeg_di: %f\nPOT1: %f\nPOT2: %f\nPOT3: %f\nPOT4: %f\n", alpha, X, Y, Z, depth, deg_di, POT1, POT2, POT3, POT4);

	DC3D0(&alpha, &X, &Y, &Z, &depth, &deg_di, &POT1, &POT2, &POT3, &POT4, &UX, &UY, &UZ, &UXX, &UYX, &UZX, &UXY, &UYY, &UZY, &UXZ, &UYZ, &UZZ, &IRET);

//    printf("UX: %f\nUY: %f\nUZ: %f\nUXX: %f\nUYX: %f\nUZX: %f\nUXY: %f\nUYY: %f\nUZY: %f\nUXZ: %f\nUYZ: %f\nUZZ: %f\nIRE: %d\n", UX, UY, UZ, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ, IRET);



//    printf("Angle: %f\ncosAngle%f\nsinAngle%f\n", Angle, cosAngle, sinAngle);

    /* transform from Okada's to cartesian system */
    *sx = cosAngle * UX + sinAngle * UY;
    *sy = -sinAngle * UX + cosAngle * UY;
    *sz = UZ;
}

void c_point_source_strain(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *s,
		float lame_lambda, float mu)
{

//    printf("x: %f\ny: %f\nz: %f\n" , x2, y2, z2);
	strain_point_source(x1, y1, z1, strike1, dip1, moment_strike, moment_dip, moment_inflation, moment_open, x2, y2, z2, &s[0], &s[4], &s[8], &s[1], &s[2], &s[5], lame_lambda, mu);
	s[3] = s[1];
	s[6] = s[2];
	s[7] = s[5];

}

void c_point_source_stress(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *s,
		float lame_lambda, float mu)
{

//    printf("x: %f\ny: %f\nz: %f\n" , x2, y2, z2);
	stress_point_source(x1, y1, z1, strike1, dip1, moment_strike, moment_dip, moment_inflation, moment_open, x2, y2, z2, &s[0], &s[4], &s[8], &s[1], &s[2], &s[5], lame_lambda, mu);
	s[3] = s[1];
	s[6] = s[2];
	s[7] = s[5];

}
