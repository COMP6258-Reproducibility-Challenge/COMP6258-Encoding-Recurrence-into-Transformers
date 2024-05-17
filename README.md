Reproducibility results from the paper: Encoding Recurrence into Transformers (https://openreview.net/pdf?id=7YfHla7IxBJ)

Experiments reproduced: 
* Informer and RSA-Informer on ETT and weather datasets for time series forecasting (found in rsa-Informer/)
* Transformer-xl and RSA-Transformer-xl on Enwik8 and Text8 for natural language modelling (found in rsa-transformer-xl)

Main changes made:
* rsa-Informer/models/attn_rsa.py self attention mechanism updated with REM module
* rsa-Informer/models/rem.py created to reimplement the REM module
* rsa-transformer-xl/pytorch/rem.py created to reimplement the REM module (no truncation)
* rsa-transformer-xl/pytorch/mem_transformer.py create RSA versions of attention modules and decoder module
* rsa-transformer-xl/pytorch/train.py add necessary arguments inluding additional attention type 4: original transformer-xl with RSA
