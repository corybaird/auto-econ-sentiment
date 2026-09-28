# Paper verification tables (generated)

## Task 1: 51-bank inventory

|                               |   value |
|:------------------------------|--------:|
| banks with 2_clean file       |      51 |
| banks with sentence file      |      51 |
| banks re-segmentation aligned |      51 |
| docs in 2_clean               |    6678 |
| docs in panel                 |    6678 |
| sentences cached              |  245781 |
| raw .txt statements           |    6693 |

## Task 1: banks with misalignment

None: every bank aligns row for row.

## Task 2: transformer recomputation, undefined and saturation (lines 195, 254)

|                                                    | CentralBankRoBERTa     | FinBERT                | FOMC-RoBERTa           |
|:---------------------------------------------------|:-----------------------|:-----------------------|:-----------------------|
| max |re_posneg - stored posneg|                    | 0.0                    | 0.0                    | 0.0                    |
| max |re_allsent - stored allsent|                  | 0.0                    | 0.0                    | 0.0                    |
| max |counted.fillna(0) - stored net_sentiment|     | 1.1102230246251565e-16 | 1.1102230246251565e-16 | 1.1102230246251565e-16 |
| stored net_sentiment NaN                           | 0                      | 0                      | 0                      |
| stored net_sentiment == 0 on PosNeg-undefined docs | 228/228                | 626/626                | 526/526                |
| PosNeg undefined % (all docs)                      | 3.4141958670260557     | 9.374064091045224      | 7.876609763402216      |
| PosNeg at +-1 % of all docs                        | 11.185983827493262     | 7.457322551662174      | 11.575321952680444     |
| PosNeg at +-1 % of defined docs                    | 11.185983827493262     | 7.457322551662174      | 11.575321952680444     |
| AllSent at +-1 % of all docs                       | 1.3477088948787062     | 0.0                    | 0.2994908655286014     |
| old net_sentiment at +-1 %                         | 11.185983827493262     | 0.10482180293501049    | 0.3444144953578916     |
| docs with S == 0                                   | 0                      | 0                      | 0                      |

## Task 3: pooled Pearson matrix, All-Sentences (the figure's configuration)

|                    |   Correa |   Hubert-Labondance |   Loughran-McDonald |   General Inquirer |   Bennani-Neuenkirch |   Apel-Blix Grimaldi |   CentralBankRoBERTa |   FinBERT |   FOMC-RoBERTa |
|:-------------------|---------:|--------------------:|--------------------:|-------------------:|---------------------:|---------------------:|---------------------:|----------:|---------------:|
| Correa             |     1    |                0.09 |                0.59 |               0.23 |                 0.19 |                 0.02 |                 0.47 |      0.43 |           0.1  |
| Hubert-Labondance  |     0.09 |                1    |                0.15 |               0.03 |                 0.66 |                 0.77 |                 0.1  |      0.33 |           0.52 |
| Loughran-McDonald  |     0.59 |                0.15 |                1    |               0.3  |                 0.3  |                 0.08 |                 0.45 |      0.44 |           0.16 |
| General Inquirer   |     0.23 |                0.03 |                0.3  |               1    |                 0.08 |                -0.01 |                 0.35 |      0.24 |          -0.14 |
| Bennani-Neuenkirch |     0.19 |                0.66 |                0.3  |               0.08 |                 1    |                 0.72 |                 0.16 |      0.43 |           0.49 |
| Apel-Blix Grimaldi |     0.02 |                0.77 |                0.08 |              -0.01 |                 0.72 |                 1    |                 0.02 |      0.25 |           0.43 |
| CentralBankRoBERTa |     0.47 |                0.1  |                0.45 |               0.35 |                 0.16 |                 0.02 |                 1    |      0.55 |          -0.02 |
| FinBERT            |     0.43 |                0.33 |                0.44 |               0.24 |                 0.43 |                 0.25 |                 0.55 |      1    |           0.35 |
| FOMC-RoBERTa       |     0.1  |                0.52 |                0.16 |              -0.14 |                 0.49 |                 0.43 |                -0.02 |      0.35 |           1    |

## Task 3: correlation family averages (line 266)

|                                                    |   within lexical |   within transformer |   cross-family |   Correa-CentralBankRoBERTa |   Correa-FinBERT |   Correa-FOMC-RoBERTa | top transformer for Correa   |
|:---------------------------------------------------|-----------------:|---------------------:|---------------:|----------------------------:|-----------------:|----------------------:|:-----------------------------|
| ('net_sentiment (old)', 'pooled', 'pearson')       |            0.281 |                0.285 |          0.292 |                       0.435 |            0.438 |                 0.104 | FinBERT                      |
| ('net_sentiment (old)', 'pooled', 'spearman')      |            0.283 |                0.291 |          0.296 |                       0.468 |            0.462 |                 0.112 | CentralBankRoBERTa           |
| ('net_sentiment (old)', 'within-bank', 'pearson')  |            0.286 |                0.296 |          0.292 |                       0.445 |            0.405 |                 0.106 | CentralBankRoBERTa           |
| ('net_sentiment (old)', 'within-bank', 'spearman') |            0.294 |                0.301 |          0.298 |                       0.485 |            0.419 |                 0.111 | CentralBankRoBERTa           |
| ('All-Sentences', 'pooled', 'pearson')             |            0.281 |                0.297 |          0.29  |                       0.465 |            0.435 |                 0.099 | CentralBankRoBERTa           |
| ('All-Sentences', 'pooled', 'spearman')            |            0.283 |                0.291 |          0.301 |                       0.502 |            0.462 |                 0.11  | CentralBankRoBERTa           |
| ('All-Sentences', 'within-bank', 'pearson')        |            0.286 |                0.308 |          0.289 |                       0.484 |            0.402 |                 0.101 | CentralBankRoBERTa           |
| ('All-Sentences', 'within-bank', 'spearman')       |            0.294 |                0.302 |          0.299 |                       0.508 |            0.42  |                 0.109 | CentralBankRoBERTa           |
| ('PosNeg', 'pooled', 'pearson')                    |            0.281 |                0.292 |          0.305 |                       0.441 |            0.429 |                 0.127 | CentralBankRoBERTa           |
| ('PosNeg', 'pooled', 'spearman')                   |            0.283 |                0.296 |          0.314 |                       0.479 |            0.468 |                 0.127 | CentralBankRoBERTa           |
| ('PosNeg', 'within-bank', 'pearson')               |            0.286 |                0.316 |          0.304 |                       0.451 |            0.397 |                 0.14  | CentralBankRoBERTa           |
| ('PosNeg', 'within-bank', 'spearman')              |            0.294 |                0.323 |          0.315 |                       0.492 |            0.426 |                 0.149 | CentralBankRoBERTa           |

## Levels: central tendency by measure (lines 256, 275)

|                                   |   mean |   median |    sd |   share > 0 |
|:----------------------------------|-------:|---------:|------:|------------:|
| Lexical: Correa                   |  0.005 |    0     | 0.49  |       0.411 |
| Lexical: Hubert-Labondance        |  0.288 |    0.333 | 0.511 |       0.65  |
| Lexical: Loughran-McDonald        | -0.196 |   -0.2   | 0.396 |       0.218 |
| Lexical: General Inquirer         | -0.038 |    0     | 0.37  |       0.39  |
| Lexical: Bennani-Neuenkirch       |  0.241 |    0.238 | 0.477 |       0.633 |
| Lexical: Apel-Blix Grimaldi       |  0.321 |    0.333 | 0.551 |       0.633 |
| CentralBankRoBERTa (allsentences) | -0.243 |   -0.25  | 0.256 |       0.135 |
| CentralBankRoBERTa (posneg)       | -0.384 |   -0.405 | 0.402 |       0.135 |
| FinBERT (allsentences)            |  0.107 |    0.094 | 0.2   |       0.621 |
| FinBERT (posneg)                  |  0.224 |    0.231 | 0.41  |       0.621 |
| FOMC-RoBERTa (allsentences)       |  0.038 |    0.014 | 0.255 |       0.504 |
| FOMC-RoBERTa (posneg)             |  0.088 |    0.111 | 0.539 |       0.504 |

## Task 4: lexical PosNeg vs All-Words (lines 252-254, 284)

|                                  |   Correa |   Hubert-Labondance |   Loughran-McDonald |   General Inquirer |   Bennani-Neuenkirch |   Apel-Blix Grimaldi |
|:---------------------------------|---------:|--------------------:|--------------------:|-------------------:|---------------------:|---------------------:|
| max |recount posneg - stored|    |   0      |              0      |              0      |             0      |               0      |               0      |
| no matches %                     |  11.186  |              9.5538 |              8.8949 |             2.2761 |               8.5954 |              12.7733 |
| median matches                   |   8      |              7      |             14      |            18      |              10      |               5      |
| PosNeg at +-1 %                  |  12.6235 |             19.991  |              6.1995 |             4.0881 |              15.3639 |              27.673  |
| PosNeg at +-1 % (excl. no-match) |  14.2135 |             22.1026 |              6.8047 |             4.1833 |              16.8087 |              31.7253 |
| PosNeg sd                        |   0.4901 |              0.5109 |              0.3959 |             0.3697 |               0.4771 |               0.5514 |
| AllWords sd                      |   0.0148 |              0.0156 |              0.0203 |             0.0233 |               0.0162 |               0.0125 |
| PosNeg IQR                       |   0.6304 |              0.6667 |              0.4615 |             0.4595 |               0.5714 |               0.8    |
| AllWords IQR                     |   0.0167 |              0.0178 |              0.0234 |             0.0296 |               0.0176 |               0.0136 |
| AllWords max |x|                 |   0.0769 |              0.125  |              0.1141 |             0.129  |               0.122  |               0.1098 |
| PosNeg mean                      |   0.0047 |              0.2879 |             -0.1964 |            -0.0376 |               0.241  |               0.321  |
| corr(PosNeg, AllWords)           |   0.8708 |              0.8171 |              0.8761 |             0.9223 |               0.8486 |               0.7825 |

## Task 5: US speech monthly coverage for the VAR

|                                      |   speeches (since start) | first month   | last month   |   months in span |   months with >=1 speech |   empty months (interpolated) | empty month list   |   median speeches/month |   months with 1 speech |
|:-------------------------------------|-------------------------:|:--------------|:-------------|-----------------:|-------------------------:|------------------------------:|:-------------------|------------------------:|-----------------------:|
| hubert_sentiment_posneg              |                     4699 | 2000-01-31    | 2023-12-31   |              288 |                      288 |                             0 | none               |                      17 |                      0 |
| cbroberta_net_sentiment_allsentences |                     4699 | 2000-01-31    | 2023-12-31   |              288 |                      288 |                             0 | none               |                      17 |                      0 |
| FRED macro                           |                      nan | 1990-01-01    | 2026-08-01   |              440 |                      nan |                           nan | nan                |                     nan |                    nan |
