Reproducibility results from the paper: Encoding Recurrence into Transformers (https://openreview.net/pdf?id=7YfHla7IxBJ)

Experiments reproduced: 
* Informer and RSA-Informer on ETT and weather datasets for time series forecasting
* Transformer-xl and RSA-Transformer-xl on Enwik8 and Text8 for natural language modelling

Main changes made:
* rsa-Informer/models/attn_rsa.py self attention mechanism updated with REM module
* rsa-Informer/models/rem.py created to reimplement the REM module
* 
