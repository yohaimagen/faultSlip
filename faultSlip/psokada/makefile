CC=gcc



INCS=-I.
CFLAGS = -Wall -Wextra -fPIC -g -c 
LOADLIBES = -L./ 




all: okada_stress


pscokada.o: pscokada.c pscokada.h defines.h util1.o
	$(CC) $(CFLAGS) -pthread pscokada.c -o pscokada.o

dc3d.o: dc3d.c dc3d.h defines.h
	$(CC) $(CFLAGS) dc3d.c -o dc3d.o

util1.o: util1.c util1.h defines.h
	$(CC) $(CFLAGS) util1.c -o util1.o

main.o: main.c pscokada.h
	$(CC) $(CFLAGS) main.c -o main.o

main: util1.o dc3d.o pscokada.o main.o
	$(CC)  -o okada_stress  main.o pscokada.o util1.o dc3d.o


okada_stress: util1.o dc3d.o pscokada.o
	ar rcs libokada_stress.a pscokada.o util1.o dc3d.o

clean:
	rm *.o 
	rm *.a
	rm okada_stress
