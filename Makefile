TOP_DIR=~/ml_cpp
SRC_DIR=$(TOP_DIR)/src
BIN_DIR=$(TOP_DIR)/bin
LIB_DIR=$(TOP_DIR)/lib
TEST_DIR=$(TOP_DIR)/test

CC=g++
INC=-Iinclude  
PYTHONINCLUDE=-I/usr/include/python3.8/
PYTHONLINKS=-L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/ -lpython3.8
FLAGS=-Wall -Werror -Wextra -g 
LIB=$(wildcard $(LIB_DIR)/*.cpp)
TEST=$(wildcard $(TEST_DIR)/*.cpp)

xor: $(SRC_DIR)/xor.cpp 
	$(CC) $(FLAGS) $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/xor.cpp -o $(BIN_DIR)/xor.o $(PYTHONLINKS)

iris: $(SRC_DIR)/iris.cpp 
	$(CC) $(FLAGS) $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/iris.cpp -o $(BIN_DIR)/iris.o $(PYTHONLINKS)

mnistnocnn: $(SRC_DIR)/mnist.cpp 
	$(CC) $(FLAGS) $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/mnist.cpp -o $(BIN_DIR)/mnist.o $(PYTHONLINKS)

test: $(TEST)
	$(CC) $(FLAGS) $(INC) -I/usr/include/cppunit $(TEST) -o $(TEST_DIR)/$(NAME).o

clean:
	rm -f $(BIN_DIR)/*.o 
