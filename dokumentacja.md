# Dokumentacja projektu: OPUS-100 PL -> EN, Model 1.1

## Zakres

Dokument opisuje tylko wariant projektu oparty o zbior OPUS-100 dla tlumaczenia z jezyka polskiego na angielski oraz konfiguracje Modelu 1.1. Glowne pliki odniesienia:

- `configs/project_config.yaml` - konfiguracja etapow pipeline, modelu, treningu i ewaluacji.
- `src/model/transformer_nmt.py` - implementacja modelu Transformer NMT.
- `scripts/train_model.py` i `src/train/model.py` - uruchomienie treningu.
- `scripts/evaluate_model.py` - ewaluacja i generowanie tlumaczen.
- `reports/tokenizer_stats.md` - raport tokenizera.
- `reports/eval_metrics.json` - metryki Modelu 1.1.
- `reports/translations/sample_translations.md` - przykladowe tlumaczenia.

## Decyzje projektowe

Najwazniejsza decyzja: model nie jest gotowym `nn.Transformer` z PyTorch. PyTorch dostarcza tensory, warstwy liniowe, embeddingi, funkcje aktywacji, optimizery i operacje macierzowe, ale sama architektura Transformer NMT jest napisana w projekcie recznie w `src/model/transformer_nmt.py`. Dzieki temu widac dokladnie, jak powstaja `Q`, `K`, `V`, jak dziala multi-head attention, gdzie sa maski i jak dekoder laczy sie z enkoderem.

Glowne decyzje:

- Kierunek tlumaczenia: `pl -> en`, czyli polski tekst jako wejscie, angielski jako wyjscie.
- Dane: wariant OPUS-100 zapisany jako Parquet i przetwarzany etapami.
- Tokenizacja: SentencePiece BPE ze wspolnym slownikiem dla obu jezykow, `vocab_size = 16000`.
- Maksymalna dlugosc: `max_seq_len = 128`, bo krotsze sekwencje sa tansze w treningu, a attention ma koszt rosnacy kwadratowo z dlugoscia.
- Architektura: encoder-decoder Transformer, 6 warstw enkodera i 6 warstw dekodera w wariancie `base`.
- Ewaluacja: beam search, SacreBLEU, chrF oraz zapis przykladowych tlumaczen do raportu Markdown.

Do walidacji jakosci danych testowo uzyto lokalnej Ollamy z modelem `qwen2.5:7b`. To pomagalo w audycie, ale nie bylo najlepszym rozwiazaniem: lokalny LLM bywa wolny, nie zawsze konsekwentny i wymaga dodatkowej kontroli wynikow. Rozwazany byl tez BiCleaner, ale wymaga osobnego stosu zaleznosci, m.in. TensorFlow; w wariancie OPUS-100 opisanym tutaj nie byl finalnie uzyty jako glowny element pipeline, wiec zostaje tylko wzmianka.

## Etapy projektu

Pipeline jest podzielony na etapy zapisane w `configs/project_config.yaml`. Najpierw dane sa pobierane do `data/raw/en-pl`, potem czyszczone do `data/processed/en-pl`, tokenizowane, ladowane do modelu, trenowane i oceniane.

### 1. Pozyskanie danych

Dane wejsciowe trafiaja do katalogu surowego:

```yaml
paths:
  raw_data_dir: data/raw/en-pl
  processed_data_dir: data/processed/en-pl

dataset:
  source_lang: pl
  target_lang: en
```

W projekcie przyjmujemy kierunek `pl -> en`. Oznacza to, ze zdanie polskie jest wejsciem enkodera, a zdanie angielskie jest sekwencja wyjsciowa dekodera.

### 2. Audyt i czyszczenie danych

Konfiguracja zaklada audyt wstepny i czyszczenie. Usuwane sa m.in. duplikaty, pary identyczne, znaki sterujace, nadmiarowe biale znaki oraz przyklady o zbyt duzej roznicy dlugosci.

```yaml
stage2_cleaning:
  filters:
    unicode_normalization: NFKC
    strip_whitespace: true
    collapse_whitespace: true
    remove_control_chars: true
    min_words: 1
    max_words: 200
    max_length_ratio: 4.0
    remove_identical_pairs: true
    dedup_scope: global
```

Cel tego etapu jest prosty: model nie powinien uczyc sie z par zduplikowanych, pustych, zle sparowanych albo technicznie uszkodzonych. Dobre czyszczenie zwykle daje wiecej niz dokladanie warstw do modelu.

W konfiguracji audytu widac tez probe uzycia lokalnego LLM-a przez Ollame:

```yaml
stage1_audit:
  llm:
    model: qwen2.5:7b
    endpoint: http://localhost:11434/api/generate
    temperature: 0.0
```

To byl mechanizm pomocniczy do oceny par tlumaczeniowych, nie rdzen modelu. W praktyce nie zastapil klasycznego czyszczenia i metryk, bo LLM jako walidator danych jest drogi obliczeniowo i nie daje idealnie powtarzalnych decyzji dla duzych zbiorow.

### 3. Podzial na zbiory

Projekt oczekuje trzech splitow zapisanych jako pliki Parquet:

```yaml
dataset:
  splits:
    train_pattern: train-*.parquet
    validation_pattern: validation-*.parquet
    test_pattern: test-*.parquet
```

Split treningowy sluzy do uczenia wag, walidacyjny do kontroli straty i wyboru najlepszego checkpointu, testowy tylko do koncowej oceny. W kodzie splitow kontrolowane jest tez przenikanie danych miedzy zbiorami, bo te same pary w treningu i tescie zawyzalyby wynik.

### 4. Tokenizacja SentencePiece

Projekt uzywa wspolnego slownika BPE SentencePiece dla polskiego i angielskiego:

```yaml
stage3_tokenizer:
  type: sentencepiece
  model_type: bpe
  vocab_size: 16000
  character_coverage: 1.0
  joint_vocab: true
  model_prefix: tokenizers/spm_pl_en
```

Wspolny slownik upraszcza model: enkoder i dekoder pracuja na tej samej przestrzeni tokenow. Slowa sa dzielone na subwordy, wiec model moze obslugiwac rzadkie formy fleksyjne bez osobnego tokenu dla kazdego slowa.

Raport tokenizera z `reports/tokenizer_stats.md` pokazuje:

```text
Model type: bpe
Vocab size: 16000
Character coverage: 1.0
Training corpus lines: 897088

Split      Rows    Whitespace words    Subword pieces    Compression    Mean pieces/sentence    P95    UNK rate
train      448544  6840073             10758450          1.573          11.99                   32     0.000000
validation 1113    19184               30222             1.575          13.58                   35     0.000000
test       1062    17971               28061             1.561          13.21                   33     0.000000
```

Wazne wnioski z raportu:

- `UNK rate = 0.0` dla wszystkich splitow, wiec tokenizer praktycznie nie gubi znakow ani slow.
- Srednio zdanie ma ok. 12-14 subwordow, a 95 percentyl miesci sie w 32-35 tokenach.
- Kompresja ok. `1.56-1.57` oznacza, ze jedno slowo whitespace staje sie srednio ok. 1.6 fragmentu subword.
- Czeste dlugie slowa sa zwykle jednym tokenem, np. `rozporzadzenia`, `czlonkowskie`, `europejskiej`; rzadkie dlugie slowa sa dzielone na wiecej czesci.

Fragment raportu dla dlugich slow:

```text
freq_1      mean pieces/word: 4.12  P95: 6
freq_2_5    mean pieces/word: 3.63  P95: 6
freq_6_50   mean pieces/word: 2.89  P95: 5
freq_gt_50  mean pieces/word: 1.67  P95: 3
```

To jest dobry objaw: czeste slowa dostaja krotsze reprezentacje, a rzadkie moga byc skladane z mniejszych czesci.

Przykladowy poczatek slownika z modelu `tokenizers/spm_pl_en.model` ma specjalne tokeny i najczestsze fragmenty subword:

```text
<pad>   0.0
<unk>   0.0
<bos>   0.0
<eos>   0.0
▁t      -0.0
ie      -1.0
▁w      -2.0
▁s      -3.0
▁p      -4.0
▁a      -5.0
he      -6.0
in      -7.0
▁o      -8.0
▁m      -9.0
re      -10.0
an      -11.0
▁n      -12.0
▁d      -13.0
st      -14.0
▁c      -15.0
on      -16.0
ou      -17.0
at      -18.0
▁b      -19.0
ow      -20.0
▁the   -21.0
er      -22.0
▁z      -23.0
en      -24.0
▁I      -25.0
▁i      -26.0
or      -27.0
▁to    -28.0
le      -29.0
ro      -30.0
ar      -31.0
ch      -32.0
rz      -33.0
sz      -34.0
ia      -35.0
```

Znak `▁` oznacza poczatek slowa po spacji. Widac, ze w slowniku mieszaja sie fragmenty polskie (`rz`, `sz`, `▁w`, `▁z`) i angielskie (`he`, `▁the`, `▁to`), bo slownik jest wspolny dla obu jezykow.

Tokenizacja splitow zamienia tekst na listy identyfikatorow:

```python
src_ids = tokenizer.encode(pl_text) + [token_ids.eos_id]
tgt_ids = tokenizer.encode(en_text)
```

Dla dekodera przygotowywane sa dwie wersje celu: wejscie z tokenem BOS i wyjscie z tokenem EOS.

```python
yield EncodedTranslationExample(
    src_ids=list(src_ids),
    tgt_in_ids=[token_ids.bos_id] + target_piece_ids,
    tgt_out_ids=target_piece_ids + [token_ids.eos_id],
)
```

### 5. Batchowanie i maski

Sekwencje maja rozne dlugosci, wiec batch jest dopelniany tokenem `<pad>`. Collator tworzy tez maski paddingu i maske przyczynowa dekodera.

```python
def make_causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)
```

Maska przyczynowa blokuje dekoderowi podglad przyszlych tokenow. Podczas uczenia model widzi poprzednie tokeny angielskiego zdania, ale nie widzi tokenu, ktory ma dopiero przewidziec.

## Model 1.1

Model 1.1 to klasyczny Transformer encoder-decoder zaimplementowany w PyTorch bez korzystania z gotowego `nn.Transformer`. Najwazniejsze parametry wariantu `base` z konfiguracji:

```yaml
stage5_model:
  preset: base
  presets:
    base:
      d_model: 512
      nhead: 8
      num_encoder_layers: 6
      num_decoder_layers: 6
      dim_feedforward: 2048
      dropout: 0.1
      tie_decoder_embeddings: true
```

Znaczenie parametrow:

- `d_model: 512` - rozmiar wektora reprezentujacego kazdy token.
- `nhead: 8` - liczba glow attention; jedna glowa ma wymiar `512 / 8 = 64`.
- `num_encoder_layers: 6` - liczba warstw enkodera.
- `num_decoder_layers: 6` - liczba warstw dekodera.
- `dim_feedforward: 2048` - rozmiar ukrytej warstwy FFN.
- `dropout: 0.1` - regularyzacja.
- `tie_decoder_embeddings: true` - ta sama macierz wag dla embeddingu dekodera i projekcji wyjsciowej.

### Konfiguracja modelu w kodzie

Podstawowe ustawienia modelu sa zapisane w dataclassie:

```python
@dataclass(frozen=True)
class TransformerNMTConfig:
    vocab_size: int
    pad_id: int = 0
    max_seq_len: int = 128
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    tie_decoder_embeddings: bool = True
```

Podczas treningu konfiguracja jest skladana z YAML-a i informacji z tokenizera:

```python
TransformerNMTConfig(
    vocab_size=int(tokenizer.vocab_size()),
    pad_id=token_ids.pad_id,
    max_seq_len=max_seq_len,
    d_model=int(preset.get("d_model", 256)),
    nhead=int(preset.get("nhead", 8)),
    num_encoder_layers=int(preset.get("num_encoder_layers", 4)),
    num_decoder_layers=int(preset.get("num_decoder_layers", 4)),
    dim_feedforward=int(preset.get("dim_feedforward", 1024)),
    dropout=runtime.dropout,
    tie_decoder_embeddings=bool(preset.get("tie_decoder_embeddings", True)),
)
```

### Embeddingi i pozycja tokenow

Transformer nie ma rekurencji, wiec sam token nie niesie informacji o pozycji w zdaniu. Dlatego do embeddingu dodawane jest sinusoidalne kodowanie pozycyjne:

```python
position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(
    torch.arange(0, d_model, 2, dtype=torch.float)
    * (-math.log(10000.0) / d_model)
)
pe = torch.zeros(max_len, d_model)
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

W `encode` i `decode` embedding jest skalowany przez `sqrt(d_model)`, a potem dodawana jest pozycja:

```python
x = self.position(self.src_embedding(src_ids) * math.sqrt(self.config.d_model))
```

Skalowanie stabilizuje wartosci embeddingow wzgledem pozostalych operacji w modelu.

### Multi-Head Attention

Najwazniejszy fragment modelu to attention. Dla kazdego tokenu model liczy trzy projekcje liniowe:

- `Q` - query, czyli pytanie: czego ten token szuka?
- `K` - key, czyli klucz: co ten token oferuje innym?
- `V` - value, czyli wartosc: jaka informacja ma zostac przekazana?

Kod dzieli `d_model` na wiele glow:

```python
self.nhead = nhead
self.head_dim = d_model // nhead
self.q = nn.Linear(d_model, d_model)
self.k = nn.Linear(d_model, d_model)
self.v = nn.Linear(d_model, d_model)
self.out = nn.Linear(d_model, d_model)
```

W przejsciu do przodu tensory sa projektowane i przestawiane do ksztaltu `batch, heads, seq_len, head_dim`:

```python
Q = self.q(q).view(batch_size, q_len, self.nhead, self.head_dim).transpose(1, 2)
K = self.k(k).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
V = self.v(v).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
```

Potem liczony jest iloczyn skalarny `QK^T`, skalowany przez pierwiastek z wymiaru glowy:

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
if mask is not None:
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

attn = F.softmax(scores, dim=-1)
out = torch.matmul(attn, V)
```

Intuicyjnie: kazdy token wybiera, na ktore inne tokeny powinien patrzec. Wiele glow pozwala patrzec rownoczesnie na rozne typy zaleznosci, np. szyk zdania, zgodnosc podmiotu z orzeczeniem, nazwy wlasne albo fragmenty idiomatyczne.

Na koncu glowy sa laczone z powrotem do `d_model`:

```python
out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
return self.out(out)
```

### Feed Forward Network

Po attention kazdy token przechodzi przez mala siec MLP. W Modelu 1.1 rozszerza ona wymiar z 512 do 2048, stosuje ReLU i wraca do 512:

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

Attention miesza informacje miedzy tokenami, a FFN przetwarza reprezentacje kazdego tokenu osobno.

### Warstwa enkodera

Warstwa enkodera sklada sie z self-attention, FFN, residual connections i LayerNorm:

```python
def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
    x = self.norm1(x + self.dropout(self.attn(x, x, x, src_mask)))
    x = self.norm2(x + self.dropout(self.ff(x)))
    return x
```

`x + ...` to polaczenie resztkowe. Pomaga trenowac glebszy model, bo gradient moze przechodzic latwiejsza sciezka przez warstwy.

W self-attention enkodera `q`, `k` i `v` sa tym samym tensorem. Kazdy polski token patrzy na inne polskie tokeny i buduje kontekst calego zdania zrodlowego.

### Warstwa dekodera

Dekoder ma trzy bloki: self-attention po angielskich tokenach, cross-attention do wyniku enkodera i FFN.

```python
def forward(
    self,
    x: torch.Tensor,
    enc_out: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> torch.Tensor:
    x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
    x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
    x = self.norm3(x + self.dropout(self.ff(x)))
    return x
```

Self-attention dekodera patrzy tylko na dotychczas wygenerowane tokeny angielskie. Cross-attention laczy dekoder z enkoderem: `q` pochodzi z dekodera, a `k` i `v` z zakodowanego zdania polskiego. To jest miejsce, gdzie model wybiera, ktore fragmenty polskiego wejscia sa istotne dla aktualnie generowanego angielskiego tokenu.

### Cala architektura

Model buduje listy warstw enkodera i dekodera:

```python
self.encoder = nn.ModuleList(
    [
        EncoderLayer(
            config.d_model, config.nhead, config.dim_feedforward, config.dropout
        )
        for _ in range(config.num_encoder_layers)
    ]
)
self.decoder = nn.ModuleList(
    [
        DecoderLayer(
            config.d_model, config.nhead, config.dim_feedforward, config.dropout
        )
        for _ in range(config.num_decoder_layers)
    ]
)
```

Przeplyw danych w `forward`:

```python
src_mask = self.make_src_attention_mask(src_key_padding_mask)
tgt_mask = self.make_tgt_attention_mask(tgt_key_padding_mask, tgt_causal_mask)
enc_out = self.encode(src_ids, src_mask)
return self.decode(tgt_in_ids, enc_out, src_mask, tgt_mask)
```

Wejscie: tokeny polskie `src_ids` i poprzednie tokeny angielskie `tgt_in_ids`. Wyjscie: logity dla kazdej pozycji dekodera, czyli rozklad prawdopodobienstwa po calym slowniku.

### Maski w modelu

Maska zrodla usuwa `<pad>` z attention:

```python
def make_src_attention_mask(self, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
    return src_key_padding_mask.logical_not().unsqueeze(1).unsqueeze(2)
```

Maska celu laczy maske paddingu i maske przyczynowa:

```python
pad_mask = tgt_key_padding_mask.logical_not().unsqueeze(1).unsqueeze(2)
causal_mask = tgt_causal_mask.logical_not().unsqueeze(0).unsqueeze(1)
return pad_mask & causal_mask
```

Dzieki temu dekoder nie patrzy ani na padding, ani na przyszle tokeny.

### Wiazanie embeddingow dekodera

W konfiguracji wlaczone jest `tie_decoder_embeddings`. Kod przypisuje te same wagi do embeddingu dekodera i projekcji na slownik:

```python
self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
if config.tie_decoder_embeddings:
    self.output_projection.weight = self.tgt_embedding.weight
```

To zmniejsza liczbe parametrow i czesto poprawia jakosc, bo wejscia i wyjscia dekodera uzywaja tej samej geometrii tokenow.

## Trening

Najwazniejsze ustawienia treningu Modelu 1.1:

```yaml
stage6_train:
  require_cuda: true
  device: cuda
  precision: bf16
  lr_peak: 1.0e-4
  warmup_steps: 4000
  weight_decay: 0.01
  label_smoothing: 0.1
  grad_clip_norm: 1.0
  micro_batch_size: 48
  grad_accum_steps: 8
  num_epochs: 200
  validate_every_steps: 1000
  save_every_steps: 1000
  early_stopping_patience: 3
```

Efektywny batch to `micro_batch_size * grad_accum_steps`, czyli `48 * 8 = 384` przykladow na krok optymalizatora. Gradient accumulation pozwala trenowac wiekszy batch bez trzymania wszystkiego naraz w pamieci GPU.

Strata to cross entropy z label smoothing:

```python
class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, ignore_index: int = 0) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, ignore_index=ignore_index
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        return self.loss(logits.reshape(-1, vocab_size), targets.reshape(-1))
```

Scheduler uzywa warmupu i potem spadku odwrotnie proporcjonalnego do pierwiastka z numeru kroku:

```python
def lr_lambda(step: int) -> float:
    current_step = max(1, step)
    return min(current_step**-0.5, current_step * (warmup**-1.5)) * math.sqrt(warmup)
```

Glowny krok treningowy:

```python
logits = self.components.model(
    batch.src_ids,
    batch.tgt_in_ids,
    batch.src_key_padding_mask,
    batch.tgt_key_padding_mask,
    batch.tgt_causal_mask,
)
loss = self.components.criterion(logits, batch.tgt_out_ids) / self.config.grad_accum_steps
```

Po akumulacji gradientow wykonywany jest clipping, krok optymalizatora, aktualizacja schedulera i zerowanie gradientow:

```python
torch.nn.utils.clip_grad_norm_(
    self.components.model.parameters(),
    self.config.grad_clip_norm,
)
self.scaler.step(self.components.optimizer)
self.scaler.update()
self.components.scheduler.step()
self.components.optimizer.zero_grad(set_to_none=True)
```

Checkpoint `best` jest zapisywany, gdy walidacyjna strata spada. Early stopping zatrzymuje trening po 3 walidacjach bez poprawy.

## Generowanie tlumaczen

Ewaluacja uzywa beam search z konfiguracji:

```yaml
stage7_eval:
  inference_checkpoint: checkpoints/model_inference.pt
  beam_size: 5
  batch_size: 8
  length_penalty: 1.0
  max_new_tokens: 127
```

Beam search utrzymuje kilka najlepszych hipotez zamiast wybierac zawsze jeden najbardziej prawdopodobny token. W kodzie kandydaci sa oceniani suma log-prawdopodobienstw z kara za dlugosc:

```python
candidate_scores = (beam_scores.unsqueeze(-1) + log_probs).view(
    batch_size, beam_size * vocab_size
)
top_scores, top_indices = torch.topk(candidate_scores, k=beam_size, dim=-1)
```

Koncowy wybor uwzglednia `length_penalty`:

```python
def _score_with_length_penalty(score: float, length: int, length_penalty: float) -> float:
    if length_penalty <= 0:
        return score
    return score / (max(1, length) ** length_penalty)
```

## Wyniki Modelu 1.1

Z pliku `reports/eval_metrics.json`:

```json
{
  "batch_size": 8,
  "beam_size": 5,
  "checkpoint": "checkpoints\\v1\\model_inference.pt",
  "decode": "beam",
  "length_penalty": 1.0,
  "max_new_tokens": 127,
  "metrics": {
    "chrf": 41.815577970304574,
    "sacrebleu": 20.345241489741813
  },
  "rows_evaluated": 1062
}
```

Interpretacja ogolna:

- `sacrebleu = 20.35` - model lapie czesc struktury i slownictwa, ale nie jest jeszcze na poziomie systemow produkcyjnych.
- `chrf = 41.82` - wynik znakowy jest wyzszy, co sugeruje, ze model czesto generuje czesciowo podobne slowa i frazy.
- `beam_size = 5` - wynik pochodzi z beam search, nie z greedy decoding.

## Przykladowe tlumaczenia

Kilka przykladow z `reports/translations/sample_translations.md`:

### Przyklad poprawny

```text
PL: Kazdy wie, ze tu jestes, Andre.
Reference EN: Everyone knows you're here, Andre.
Hypothesis EN: Everyone knows you're here, Andre.
```

Model tlumaczy zdanie doslownie i poprawnie. Zachowuje imie oraz sens wypowiedzi.

### Przyklad dobry, ale parafraza

```text
PL: Byles dzisiaj wspanialy, Nate.
Reference EN: Hey, you were really great today, Nate.
Hypothesis EN: You were wonderful today, Nate.
```

Hipoteza nie jest identyczna z referencja, ale znaczenie zostaje zachowane. BLEU moze karac takie roznice, mimo ze tlumaczenie jest akceptowalne.

### Przyklad czesciowo poprawny

```text
PL: Wiedzielismy, ze ta pulapka nie zatrzyma ich na zawsze.
Reference EN: We know that that trap is only gonna hold them for so long.
Hypothesis EN: We knew this trap wouldn't stop them forever.
```

Model dobrze oddaje sens, chociaz czas i styl sa inne niz w referencji.

### Przyklad z bledem semantycznym

```text
PL: Moj stryj zagarnal moje ziemie i przysiagl wiernosc Dunczykom.
Reference EN: My uncle took my lands and pledged his allegiance to the Danes.
Hypothesis EN: My grandfather lost my land and popped the Duke.
```

Tutaj model myli kluczowe informacje: `stryj` zostaje przetlumaczony jako `grandfather`, a druga czesc zdania traci sens. To typowy problem modelu niedotrenowanego albo uczonego na zaszumionych parach.

### Przyklad dlugiego zdania

```text
PL: Chociaz w swoich zaleceniach z roku 2002 i 2004 Miedzynarodowa Rada Badan Morza zwrocila uwage na fakt, ze wiekszosc gatunkow nie osiaga poziomow bezpieczenstwa biologicznego, Unia Europejska nie zmniejszyla w wystarczajacym stopniu swoich nakladow polowowych, by zapewnic zrownowazone polowy.
Reference EN: Although the 2002 and 2004 recommendations of the International Council for the Exploration of the Sea ICES called attention to the fact that most species are below the biosafety levels, the European Union has not reduced its fishing efforts enough to ensure sustainable fishing.
Hypothesis EN: Although, in its recommendations of 2002 and 2004 the International Research Council, the Council considered that most of the species did not achieve the level of biological safety of the European Union, the European Union does not reduce its fishing opportunities to ensure that it is sufficient to ensure sustainable sustainable sustainable.
```

Dlugie zdania sa wyraznie trudniejsze. Model zachowuje czesc terminologii, ale powtarza slowa i miesza relacje skladniowe. To pasuje do ograniczenia `max_seq_len = 128` i relatywnie malego modelu wzgledem trudnosci zadania.

## Wnioski

Model 1.1 ma poprawna architekture Transformer NMT: embeddingi, kodowanie pozycyjne, multi-head attention, warstwy enkodera/dekodera, maski, FFN, residual connections i beam search. Wyniki pokazuja, ze model potrafi tlumaczyc proste zdania i czesc parafraz, ale traci jakosc na dlugich zdaniach, nazwach wlasnych, terminologii i rzadkich konstrukcjach.

Najbardziej widoczne ograniczenia:

- Dlugie zdania generuja powtorzenia i bledy skladniowe.
- Rzadkie slowa i nazwy wlasne bywaja przekrecane.
- Model czasem zachowuje ogolny sens, ale gubi szczegoly semantyczne.
- BLEU 20.35 wskazuje na dzialajacy model bazowy, nie na finalny system tlumaczeniowy.

Najbardziej sensowne dalsze kroki, jesli celem jest jakosc, to wiecej czystych danych, dluzszy lub stabilniejszy trening, kontrola overfittingu oraz analiza bledow na dlugich zdaniach. Dokladanie abstrakcji w kodzie modelu nie jest tu potrzebne; problemem jest jakosc uczenia i danych, nie struktura plikow.

## Dalszy kierunek: Tatoeba

Po wariancie OPUS-100 kolejny model zostal uczony na fragmencie datasetu Tatoeba. Po czyszczeniu ten wariant mial ok. 14 mln par zdan, czyli znacznie wiecej danych niz opisany tutaj OPUS-100. Celem bylo sprawdzenie, czy ta sama architektura lepiej wykorzysta wiekszy i czystszy zbior.

Wstepne uczenie pokazalo, ze dla konfiguracji:

```yaml
d_model: 512
nhead: 8
num_encoder_layers: 6
num_decoder_layers: 6
dim_feedforward: 2048
```

danych moze byc juz nawet za duzo wzgledem rozmiaru modelu i praktycznego budzetu treningowego. Sam zbior nie jest problemem, ale model tej wielkosci moze nie zdazyc dobrze przetworzyc takiej skali bez dluzszego treningu, lepszego harmonogramu albo wiekszej architektury. To byl powod przejscia od OPUS-100 do Tatoeba: nie zmiana samego kodu modelu, tylko sprawdzenie, jak ta sama recznie napisana architektura zachowa sie na znacznie wiekszym i lepiej oczyszczonym materiale.
