CC=g++

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
ROOT_D := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

ifeq ($(CFG),debug)
	CXXFLAGS=-Wall -O2 -g -DDEBUG -DROOT_DIR=\"$(ROOT_D)\"
	LDFLAGS=-Wall -g -DDEBUG
	EXEC=$(BINPATH)/ataridebug
else
	CXXFLAGS=-Wall -O2 -DROOT_DIR=\"$(ROOT_D)\"
	LDFLAGS=-Wall
	EXEC=$(BINPATH)/atari
endif

INCPATH=$(ROOT_D)/inc
SRCPATH=$(ROOT_D)/src
OBJPATH=$(ROOT_D)/obj
BINPATH=$(ROOT_D)/bin

SRC=$(SRCPATH)/util.cpp \
    $(SRCPATH)/main.cpp \
    $(SRCPATH)/interface.cpp \
    $(SRCPATH)/info.cpp \
    $(SRCPATH)/ql.cpp
OBJ=$(OBJPATH)/util.o \
    $(OBJPATH)/main.o \
    $(OBJPATH)/interface.o \
    $(OBJPATH)/info.o \
    $(OBJPATH)/ql.o

INCLUDES=-I $(INCPATH)

default: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

$(OBJPATH)/%.o: $(SRCPATH)/%.cpp $(INCPATH)/%.h
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: clean cleanall

clean:
	rm -f $(OBJPATH)/*.o

cleanall: clean
	rm -f $(BINPATH)/*