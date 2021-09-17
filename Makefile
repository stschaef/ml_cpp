CUR_ROOT=/home/stschaef/ml_cpp/
SRC_DIR=$(CUR_ROOT)/src
BIN_DIR=$(CUR_ROOT)/bin
LIB_DIR=$(CUR_ROOT)/lib
TEST_DIR=$(CUR_ROOT)/test
FRONT_END_DIR=$(CUR_ROOT)/frontend

CXX=em++
INC=-I/home/stschaef/ml_cpp/include  
PYTHONINCLUDE=-I/usr/include/python3.8/
OPEN_CV=`pkg-config --cflags --libs opencv`
PYTHONLINKS=-L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/ -lpython3.8
FLAGS=-Wall -Werror -Wextra -g 
LIB=$(wildcard $(LIB_DIR)/*.cpp)
TEST=$(wildcard $(TEST_DIR)/*.cpp)

xor: $(SRC_DIR)/xor.cpp 
	$(CXX) $(FLAGS) $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/xor.cpp -o $(BIN_DIR)/xor.o $(PYTHONLINKS) $(OPEN_CV)

iris: $(SRC_DIR)/iris.cpp 
	$(CXX) -v $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/iris.cpp -s WASM=1 -o $(FRONT_END_DIR)/iris.html $(PYTHONLINKS)

animals: $(SRC_DIR)/animals.cpp D
	$(CXX) $(FLAGS) $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/animals.cpp -o $(BIN_DIR)/animals.o $(PYTHONLINKS) $(OPEN_CV)

mnist: $(SRC_DIR)/mnist.cpp 
	$(CXX) $(FLAGS) $(INC) $(LIB) $(PYTHONINCLUDE) $(SRC_DIR)/mnist.cpp -o $(BIN_DIR)/mnist.o $(PYTHONLINKS) $(OPEN_CV)

seven: $(SRC_DIR)/seven.cpp 
	$(CXX) $(FLAGS) $(INC) $(LIB) $(PYTHONINCLUDE) $(SRC_DIR)/seven.cpp -o $(BIN_DIR)/seven.o $(PYTHONLINKS) $(OPEN_CV)

test: $(TEST)
	$(CXX) $(FLAGS) $(INC) -I/usr/include/cppunit $(TEST) -o $(TEST_DIR)/$(NAME).o

clean:
	rm -f $(BIN_DIR)/*.o 
