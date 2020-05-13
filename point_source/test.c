#include <stdio.h>
#include "c_DC3D.h"


int main()
{
	float ALPHA,X,Y,Z,DEPTH,DIP,POT1,POT2,POT3,POT4,UX,UY,UZ,UXX,UYX,UZX,UXY,UYY,UZY,UXZ,UYZ,UZZ;
	int IRET;
	ALPHA = 0.7272727272727273;
	X = 20000.0;
	Y = 20000.0;
	Z = 0.0;
	DEPTH = 5000.0;
	DIP = 90;
	POT1 = 1182711.2974452535;
	POT2 = 0.0;
	POT3 = 0.0;
	POT4 = 0.0;

	DC3D0(&ALPHA, &X, &Y, &Z, &DEPTH, &DIP, &POT1, &POT2, &POT3, &POT4,
		  &UX, &UY, &UZ, &UXX, &UYX, &UZX, &UXY, &UYY, &UZY, &UXZ, &UYZ, &UZZ, &IRET);
	printf("UX: %f\nUY: %f\nUZ: %f\nUXX: %f\nUYX: %f\nUZX: %f\nUXY: %f\nUYY: %f\nUZY: %f\nUXZ: %f\nUYZ: %f\nUZZ: %f\n", UX,UY,UZ,UXX,UYX,UZX,UXY,UYY,UZY,UXZ,UYZ,UZZ );
}