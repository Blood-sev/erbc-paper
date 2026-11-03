# Main result tables

[Overview](README.md) | [Protocols and interpretation](RESULTS.md)

Errors are percentage points; lower is better. CS denotes computer science. Categories refer to task-specific numerical-estimator training. ERBC uses the NTL-Ens numerical B estimate. Values are transcribed from the current manuscript without recomputation.

## 148-unit C component

| Category | Method | pMAE Overall | pMAE CS | pRMSE Overall | pRMSE CS |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Unsupervised | Qwen3-4B-Instruct | 26.8860 | 37.4335 | 37.2338 | 45.6597 |
| Unsupervised | GPT-6 (Sol) | 23.6016 | 29.3707 | 35.7862 | 40.8841 |
| Unsupervised | GLM-5.3 | 20.7889 | 28.2443 | 32.8633 | 39.9025 |
| Supervised | NTL-a | 17.5606 | 27.2917 | 32.3301 | 39.5373 |
| Supervised | NTL-b | 17.5686 | 27.3055 | 32.4396 | 39.6931 |
| Supervised | NTL-Ens | 17.5520 | 27.2767 | 32.3789 | 39.6067 |
| Supervised | R2CCP | 20.7073 | 26.5316 | 30.4981 | 36.1880 |
| Supervised | RAFT | 17.6372 | 27.3389 | 32.3462 | 39.5659 |
| Supervised | TRACT | 17.9429 | 26.7812 | 33.2858 | 39.4319 |
| Supervised | ERBC | 15.1457 | 23.0870 | 29.5223 | 35.5079 |

## 148-unit B-to-C pipeline

| Category | Method | B pMAE Overall | B pMAE CS | C pMAE Overall | C pMAE CS |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Unsupervised | Qwen3-4B-Instruct | 31.8710 | 30.2835 | 27.1732 | 35.6830 |
| Unsupervised | GPT-6 (Sol) | 25.7386 | 32.4546 | 25.9296 | 33.8382 |
| Unsupervised | GLM-5.3 | 24.6684 | 30.6854 | 22.5032 | 29.4625 |
| Supervised | NTL-a | 14.9230 | 22.3905 | 17.3149 | 26.8602 |
| Supervised | NTL-b | 14.9350 | 22.9558 | 17.0138 | 26.3224 |
| Supervised | NTL-Ens | 14.7260 | 22.4798 | 17.1158 | 26.5058 |
| Supervised | R2CCP | 28.1216 | 41.1686 | 21.2216 | 27.0969 |
| Supervised | RAFT | 15.8783 | 23.3013 | 18.0047 | 26.3045 |
| Supervised | TRACT | 16.2717 | 24.4931 | 18.0505 | 28.2430 |
| Supervised | ERBC | 14.7260 | 22.4798 | 14.9537 | 22.7527 |

## 162-unit frozen pipeline check

| Category | Method | B pMAE Overall | B pMAE CS | C pMAE Overall | C pMAE CS |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Unsupervised | Qwen3-4B-Instruct | 36.1570 | 35.0387 | 31.0898 | 41.4688 |
| Unsupervised | GPT-6 (Sol) | 26.8766 | 28.5601 | 26.3345 | 29.2722 |
| Unsupervised | GLM-5.3 | 25.4763 | 28.3416 | 22.1262 | 27.1478 |
| Supervised | NTL-a | 16.7602 | 23.8854 | 17.9194 | 26.6750 |
| Supervised | NTL-b | 16.9514 | 23.7563 | 17.9014 | 26.6389 |
| Supervised | NTL-Ens | 16.6436 | 23.5913 | 17.8538 | 26.5647 |
| Supervised | R2CCP | 33.7189 | 48.7096 | 23.7739 | 30.9647 |
| Supervised | RAFT | 16.6571 | 23.4314 | 18.9284 | 26.6956 |
| Supervised | TRACT | 17.7716 | 25.4485 | 19.2640 | 28.8216 |
| Supervised | ERBC | 16.6436 | 23.5913 | 15.5594 | 22.8563 |
