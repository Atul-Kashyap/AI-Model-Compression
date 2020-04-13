# AI-Model-Compression
AI model compression using weight pruning and 8-bit quantization


In this module the input model is pruned and retrained. For the input model different sparsity is set, for each sparsity the pruning 
operations are set. These pruning operations are applied on the model. After applying pruning operations, the model is retrained so 
that the layers are adapted to the pruning operation. For each sparsity the model is saved, best model is chosen accounting the 
sparsity vs accuracy trade-off. The selected model is saved as pruned model and give to the next module for quantization



In this module the pruned model is extracted and converted to tflite file. Quantization is applied on the converted model and delivered 
as the compressed model.
