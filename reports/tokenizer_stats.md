# Tokenizer Report (Stage 3)

Generated at (UTC): `2026-06-12T10:26:09.538740+00:00`
Model type: `bpe`
Vocab size: `16000`
Character coverage: `1.0`
Training corpus lines: `897088`

## Split Stats

| Split | Rows | Whitespace words | Subword pieces | Compression | Mean pieces/sentence | P95 pieces/sentence | UNK rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 448544 | 6840073 | 10758450 | 1.573 | 11.99 | 32 | 0.000000 |
| validation | 1113 | 19184 | 30222 | 1.575 | 13.58 | 35 | 0.000000 |
| test | 1062 | 17971 | 28061 | 1.561 | 13.21 | 33 | 0.000000 |

## Long Word Split Stats

Long word means length >= 12. Desired pattern: common long words split less than rare long words.

| Frequency bucket | Word types | Mean pieces/word | P95 pieces/word |
|---|---:|---:|---:|
| freq_1 | 18764 | 4.12 | 6 |
| freq_2_5 | 9564 | 3.63 | 6 |
| freq_6_50 | 4002 | 2.89 | 5 |
| freq_gt_50 | 444 | 1.67 | 3 |

## Common Long Word Examples

| Word | Train frequency | Pieces |
|---|---:|---:|
| `rozporządzenia` | 1893 | 1 |
| `członkowskie` | 1427 | 1 |
| `europejskiej` | 1089 | 1 |
| `rozporządzenie` | 936 | 1 |
| `szczególności` | 825 | 1 |
| `bezpieczeństwa` | 788 | 1 |
| `członkowskich` | 787 | 1 |
| `requirements` | 681 | 1 |
| `europejskiego` | 658 | 1 |
| `przynajmniej` | 653 | 1 |
| `international` | 646 | 1 |
| `powiedziałem` | 606 | 1 |
| `uwzględniając` | 492 | 1 |
| `implementation` | 477 | 1 |
| `powiedziałeś` | 435 | 1 |
| `investigation` | 431 | 1 |
| `zastosowanie` | 423 | 1 |
| `potrzebujemy` | 415 | 1 |
| `kiedykolwiek` | 404 | 1 |
| `działalności` | 384 | 1 |
| `przedsiębiorstwa` | 354 | 2 |
| `relationship` | 354 | 1 |
| `powiedziałam` | 350 | 1 |
| `institutions` | 346 | 1 |
| `prawdopodobnie` | 346 | 1 |
| `przewodniczący` | 339 | 1 |
| `rozporządzeniem` | 336 | 1 |
| `porozumienia` | 335 | 1 |
| `sprawozdania` | 334 | 1 |
| `członkowskiego` | 334 | 1 |
