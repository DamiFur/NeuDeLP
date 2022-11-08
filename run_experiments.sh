for complexity in simple complex
do
    for program_size in 200 500 1000
    do
        for lr in 2e-03 5e-03 5e-02
        do
	    for layer_size in 300
            do
		    python train_nn.py --lr ${lr} --num_layers 10 --complexity ${complexity} --program_size ${program_size} --layers_size ${layer_size} --blocking True --output_size 3
            done
	done
    done
done
