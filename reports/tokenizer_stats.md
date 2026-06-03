# Tokenizer Report (Stage 3)

Generated at (UTC): `2026-06-03T08:10:09.583447+00:00`
Model type: `bpe`
Vocab size: `16000`
Character coverage: `1.0`
Training corpus lines: `816030`

## Split Stats

| Split | Rows | Whitespace words | Subword pieces | Compression | Mean pieces/sentence | P95 pieces/sentence | UNK rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 408015 | 6392545 | 10027528 | 1.569 | 12.29 | 33 | 0.000000 |
| validation | 901 | 16136 | 25316 | 1.569 | 14.05 | 36 | 0.000000 |
| test | 897 | 15758 | 24561 | 1.559 | 13.69 | 34 | 0.000000 |

## Long Word Split Stats

Long word means length >= 12. Desired pattern: common long words split less than rare long words.

| Frequency bucket | Word types | Mean pieces/word | P95 pieces/word |
|---|---:|---:|---:|
| freq_1 | 18045 | 4.09 | 6 |
| freq_2_5 | 9285 | 3.59 | 6 |
| freq_6_50 | 3869 | 2.85 | 5 |
| freq_gt_50 | 410 | 1.60 | 2 |

## Common Long Word Examples

| Word | Train frequency | Pieces |
|---|---:|---:|
| `rozporządzenia` | 1718 | 1 |
| `członkowskie` | 1340 | 1 |
| `europejskiej` | 1009 | 1 |
| `rozporządzenie` | 772 | 1 |
| `szczególności` | 757 | 1 |
| `bezpieczeństwa` | 751 | 1 |
| `członkowskich` | 746 | 1 |
| `requirements` | 641 | 1 |
| `international` | 617 | 1 |
| `europejskiego` | 604 | 1 |
| `przynajmniej` | 585 | 1 |
| `powiedziałem` | 541 | 1 |
| `implementation` | 444 | 1 |
| `uwzględniając` | 413 | 1 |
| `investigation` | 410 | 1 |
| `powiedziałeś` | 398 | 1 |
| `kiedykolwiek` | 382 | 1 |
| `zastosowanie` | 379 | 1 |
| `potrzebujemy` | 377 | 1 |
| `działalności` | 365 | 1 |
| `przedsiębiorstwa` | 349 | 2 |
| `przewodniczący` | 339 | 1 |
| `relationship` | 329 | 1 |
| `institutions` | 322 | 1 |
| `powiedziałam` | 318 | 1 |
| `sprawozdania` | 312 | 1 |
| `członkowskiego` | 308 | 1 |
| `przedsiębiorstw` | 307 | 1 |
| `sprawozdanie` | 306 | 1 |
| `porozumienia` | 303 | 1 |
