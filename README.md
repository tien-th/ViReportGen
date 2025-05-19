# Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation



Our code base includes folowing stages:
1. Finetune Vision Encoder
 - Finetune CTViT model on Vietnamese PET/CT-report dataset. Details can be found in [pet-clip/README.md](pet-clip/README.md)
 - Finetune Cosmos model on Vietnamese PET/CT-report dataset. Details can be found in [Cosmos/README.md](Cosmos/README.md)
2. Training and Inference VLMs
 - Training and Inference LLaVA model. Details can be found in [VLMs/README.md](VLMs/README.md)
3. Clinical Evaluation
 - Extract structured lesion information from the LLM output and clinically evaluate the predictions. Details can be found in [clinical_evaluation/README.md](clinical_evaluation/README.md)
