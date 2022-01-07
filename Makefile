CC := gcc
LD := gcc
LDFLAGS := -fPIC -shared
CFLAGS := -fPIC

WARN := -Wall -Wextra
GDB := -ggdb3
ASAN := -fsanitize=address
#UBSAN := -fsanitize=undefined

ifeq ($(DEBUG),1)
OPT :=
CFLAGS += $(ASAN)
LDFLAGS += $(ASAN)
CFLAGS += $(UBSAN)
LDFLAGS += $(UBSAN)
else
OPT := -Ofast
#OPT += -march=native
endif

CFLAGS += $(WARN) $(GDB) $(OPT)

LIB := libqp_solver.so

.PHONY: all
all: $(LIB)

.PHONY: clean
clean:
	$(RM) $(LIB) qp_solver.o nstx_math.o

.PHONY: check_qp
check_qp: qp_test.py $(LIB)
	python3 $<

.PHONY: check_mpc
check_mpc: mpc_test.py $(LIB)
	python3 $<

$(LIB): qp_solver.o nstx_math.o
	$(LINK.o) $(OUTPUT_OPTION) $^

qp_solver.o: qp_solver.h nstx_math.h

nstx_math.o: nstx_math.h
