#! bin/bash 

scp Dataset_2_OPTIMIZATION.tar.xz mcmercado@10.0.9.31:/home/mcmercado/Training_MicroSpore/TRAINING_WD 

zip -r train_models_output.zip train_models_output

scp mcmercado@10.0.9.31:/home/mcmercado/Training_MicroSpore/trained_models_output.zip $PWD 

scp mcmercado@10.0.9.31:/home/mcmercado/Training_MicroSpore/color.zip $PWD 

scp mcmercado@10.0.9.31:~/Training_MicroSpore/color.zip $PWD