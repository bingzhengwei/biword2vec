
CC = g++
CPPFLAGS = -Wall -O3 -fPIC -std=c++11 -march=native
INCLUDES = -I.
LDFLAGS = -pthread

all: biword2vec distance

COMMON_SRC = src/util.cpp \
	  src/word_table.cpp \
	  src/sampler.cpp

COMMON_OBJ = $(subst .cpp,.o, $(COMMON_SRC))

%.o : %.cpp src/*.h
	$(CC) -c $< -o $@ $(INCLUDES) $(CPPFLAGS)

biword2vec: src/biword2vec.o $(COMMON_OBJ)
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

distance: src/distance.o $(COMMON_OBJ)
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -f src/*.o biword2vec distance
