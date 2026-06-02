# Tokenizer Report (Stage 3)

Generated at (UTC): `2026-06-02T16:08:21.246685+00:00`
Model type: `bpe`
Vocab size: `16000`
Character coverage: `1.0`
Training corpus lines: `559870`

## Split Stats

| Split | Rows | Whitespace words | Subword pieces | Compression | Mean pieces/sentence | P95 pieces/sentence | UNK rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 279935 | 4357043 | 6832922 | 1.568 | 12.20 | 33 | 0.000000 |
| validation | 901 | 16136 | 25327 | 1.570 | 14.05 | 36 | 0.000000 |
| test | 897 | 15758 | 24561 | 1.559 | 13.69 | 34 | 0.000000 |

## Long Word Split Stats

Long word means length >= 12. Desired pattern: common long words split less than rare long words.

| Frequency bucket | Word types | Mean pieces/word | P95 pieces/word |
|---|---:|---:|---:|
| freq_1 | 14830 | 4.00 | 6 |
| freq_2_5 | 7167 | 3.49 | 6 |
| freq_6_50 | 2774 | 2.66 | 4 |
| freq_gt_50 | 250 | 1.30 | 2 |

## Common Long Word Examples

| Word | Train frequency | Pieces |
|---|---:|---:|
| `rozporządzenia` | 1139 | 1 |
| `członkowskie` | 907 | 1 |
| `europejskiej` | 646 | 1 |
| `rozporządzenie` | 537 | 1 |
| `szczególności` | 515 | 1 |
| `członkowskich` | 510 | 1 |
| `bezpieczeństwa` | 506 | 1 |
| `requirements` | 441 | 1 |
| `international` | 412 | 1 |
| `europejskiego` | 394 | 1 |
| `przynajmniej` | 394 | 1 |
| `powiedziałem` | 381 | 1 |
| `implementation` | 308 | 1 |
| `uwzględniając` | 290 | 1 |
| `investigation` | 267 | 1 |
| `zastosowanie` | 263 | 1 |
| `potrzebujemy` | 263 | 1 |
| `powiedziałeś` | 260 | 1 |
| `działalności` | 255 | 1 |
| `kiedykolwiek` | 246 | 1 |
| `przedsiębiorstwa` | 245 | 2 |
| `powiedziałam` | 228 | 1 |
| `przewodniczący` | 225 | 1 |
| `sprawozdania` | 223 | 1 |
| `agricultural` | 216 | 1 |
| `potrzebujesz` | 215 | 1 |
| `sprawozdanie` | 215 | 1 |
| `institutions` | 214 | 1 |
| `relationship` | 213 | 1 |
| `członkowskiego` | 213 | 1 |
