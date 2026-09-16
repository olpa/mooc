run: run_va run_vas

run_va: vector_add
	./vector_add

run_vas: vector_add
	./vector_add --stride

vector_add: vector_add.cu
	nvcc -o $@ $<
