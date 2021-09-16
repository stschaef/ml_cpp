SRC_DIR=src
BIN_DIR=bin
LIB_DIR==lib
TEST_DIR=test

CC=g++
INC=-Iinclude  
PYTHONINCLUDE=-I/usr/include/python3.8/
OPEN_CV=`pkg-config --cflags --libs opencv`
PYTHONLINKS=-L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/ -lpython3.8
FLAGS=-Wall -Werror -Wextra -g 
LIB=$(wildcard $(LIB_DIR)/*.cpp)
TEST=$(wildcard $(TEST_DIR)/*.cpp)

xor: $(SRC_DIR)/xor.cpp 
	$(CC) $(FLAGS) $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/xor.cpp -o $(BIN_DIR)/xor.o $(PYTHONLINKS)

iris: $(SRC_DIR)/iris.cpp 
	$(CC) $(FLAGS) $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/iris.cpp -o $(BIN_DIR)/iris.o $(PYTHONLINKS)

animals: $(SRC_DIR)/animals.cpp 
	$(CC) $(FLAGS) $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/animals.cpp -o $(BIN_DIR)/animals.o $(PYTHONLINKS) $(OPEN_CV)

test: $(TEST)
	$(CC) $(FLAGS) $(INC) -I/usr/include/cppunit $(TEST) -o $(TEST_DIR)/$(NAME).o

clean:
	rm -f $(BIN_DIR)/*.o 
