CXX=clang++
INCLUDES=-Iincludes/ -Ilib/
CXXEXTRAS=`pkg-config --libs --cflags opencv4`
CXXFLAGS=-std=c++20 -g -fstandalone-debug -lboost_system -lboost_filesystem
SRC=./src/driver.cc ./src/data_loader.cc ./src/augmentations.cc ./src/random_rotation_utilities.cc ./src/utilities.cc

exec: bin/exec
main: bin/main
tests: bin/tests

bin/exec: ./src/example.cc ./src/data_loader.cc ./src/augmentations.cc ./src/random_rotation_utilities.cc ./src/utilities.cc
	$(CXX) $(CXXFLAGS) $(CXXEXTRAS) $(INCLUDES) $^ -o $@

bin/main: ./src/main.cc ./src/data_loader.cc ./src/augmentations.cc ./src/random_rotation_utilities.cc ./src/utilities.cc
	$(CXX) $(CXXFLAGS) $(CXXEXTRAS) $(INCLUDES) $^ -o $@

bin/tests: ./tests/tests.cc obj/catch.o ./src/data_loader.cc ./src/augmentations.cc ./src/random_rotation_utilities.cc ./src/utilities.cc
	$(CXX) $(CXXFLAGS) $(CXXEXTRAS) $(INCLUDES) $^ -o $@

obj/catch.o: tests/catch.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

.DEFAULT_GOAL := exec
.PHONY: clean exec tests

clean:
	rm -rf bin/* obj/*
