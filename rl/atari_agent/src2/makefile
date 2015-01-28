CC=nvcc

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
ROOT_D := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

ifeq ($(CFG),debug)
	CXXFLAGS= -O2 -g -DDEBUG -DROOT_DIR=\"$(ROOT_D)\" -I /home/student/dhruv/cudnn-6.5-linux-x64-v2-rc2 -arch sm_35
	LDFLAGS= -g -DDEBUG -L /home/student/dhruv/cudnn-6.5-linux-x64-v2-rc2 -lcudnn
	EXEC=$(BINPATH)/ataridebug
else
	CXXFLAGS= -O2 -DROOT_DIR=\"$(ROOT_D)\" -I /home/student/dhruv/cudnn-6.5-linux-x64-v2-rc2 -arch sm_35
	LDFLAGS= -L /home/student/dhruv/cudnn-6.5-linux-x64-v2-rc2 -lcudnn
	EXEC=$(BINPATH)/atari 
endif

INCPATH=$(ROOT_D)/inc
SRCPATH=$(ROOT_D)/src
OBJPATH=$(ROOT_D)/obj
BINPATH=$(ROOT_D)/bin

SRC=$(SRCPATH)/util.cu \
    $(SRCPATH)/main.cu \
    $(SRCPATH)/interface.cu \
    $(SRCPATH)/info.cu \
    $(SRCPATH)/ql.cu \
    $(SRCPATH)/cnn3.cu
OBJ=$(OBJPATH)/util.o \
    $(OBJPATH)/main.o \
    $(OBJPATH)/interface.o \
    $(OBJPATH)/info.o \
    $(OBJPATH)/ql.o \
    $(OBJPATH)/cnn3.o

INCLUDES=-I $(INCPATH) 

default: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

$(OBJPATH)/%.o: $(SRCPATH)/%.cu $(INCPATH)/%.h
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: clean cleanall

clean:
	rm -f $(OBJPATH)/*.o

cleanall: clean
	rm -f $(BINPATH)/*
