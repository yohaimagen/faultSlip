CC=gcc
CF=gfortran



INCS=-I.
CFLAGS = -Wall -Wextra -g -c 
LOADLIBES = -L./ 




all: okada_point_source



dc3d.o: DC3D.f 
	$(CF) -fPIC  -c DC3D.f -o DC3D.o

c_dc3d.o: c_DC3D.c c_DC3D.h 
	$(CC) $(CFLAGS) c_DC3D.c -pthread -fPIC -lm  -o c_DC3D.o


test.o: test.c c_DC3D.h
	$(CC) $(CFLAGS) test.c -o test.o

test: dc3d.o c_dc3d.o test.o
	$(CC)  -o test -pthread test.o c_DC3D.o DC3D.o -lm


okada_point_source:  dc3d.o c_dc3d.o
	ar rcs libokada_point_source.a c_DC3D.o DC3D.o


clean:
	rm *.o 
	rm *.a
	rm test
