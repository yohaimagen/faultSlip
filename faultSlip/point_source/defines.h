
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


#ifndef DEFINES_H
#define DEFINES_H


#define sq(A) ((A)*(A))
#define sign(A) ((int) (A/fabs(A)))
#define max(A,B) (A>B)? A : B
#define eps0 1.0e-3
#define PI (3.141592653589793)
#define pi (3.141592653589793)
#define EUL (2.718281828459)
#define ASIZE 100000
#define CSIZE 512
#define Re (6370)
#define DEG2RAD  (0.0174532925)
#define KM2M     (1000.0)
#define RAD2DEG  (57.2957795147)
#define SEC2DAY	(1.0/(24.0*3600.0))
#define T2SEC(x) (x/(double)CLOCKS_PER_SEC)
#define tol0 1e-10	//tolerance for double comparison.
#define	MIN(a,b) (((a)<(b))?(a):(b))
#define	MAX(a,b) (((a)>(b))?(a):(b))


#endif //DEFINES_H
