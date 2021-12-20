CC := gcc
LD := gcc
LDFLAGS := -fPIC -shared

ifeq ($(DEBUG),1)
OPT :=
else
OPT := -Ofast
#OPT += -march=native
endif
CFLAGS := -Wall -fPIC $(OPT)

LIB := libqp_solver.so

.PHONY: all
all: $(LIB)

.PHONY: clean
clean:
	$(RM) $(LIB) qp_solver.o nstx_math.o

.PHONY: check
check: qp_test.py $(LIB)
	python3 $<

$(LIB): qp_solver.o nstx_math.o
	$(LINK.o) $(OUTPUT_OPTION) $^

qp_solver.o: qp_solver.h nstx_math.h

nstx_math.c: nstx_math.h
