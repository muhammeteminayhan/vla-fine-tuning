# Proje: Teach-by-Demonstration VLA Hücresi — Öğretme Maliyeti Eğrisi

Bu dosya projenin kalıcı sözleşmesidir. Her oturumda oku ve buna uy.

---

## 0. Çalışma anlaşması (en önemli bölüm — atlamadan oku)

Ben deneyimli bir yazılımcıyım ama VLA (Vision-Language-Action) modelleri ve robotik
simülasyon benim için yeni. Bu projeyi **hem bitirmek hem öğrenmek** için yapıyorum.
Bana çalışan kod teslim etmen yetmez; ne yaptığını anlamam gerekiyor.

### Dil
- **Bana yaptığın açıklamalar Türkçe.** Teknik terimleri İngilizce bırak
  (fine-tune, checkpoint, rollout, LoRA gibi) — çevirmeye çalışma, kafa karıştırıyor.
- **Kod, yorum satırları, README, commit mesajları, dosya adları İngilizce.**
  Bu repo portföy parçası, dışarıdan bakan biri okuyacak.

### Her iş bloğundan sonra bana şunu ver

Uzun bir iş yaptıktan sonra (bir dosya yazdıktan, bir komut çalıştırdıktan,
bir hatayı çözdükten sonra) şu dört başlıkta kısa bir özet geç:

1. **Ne yaptım** — 2-3 cümle.
2. **Neden böyle** — hangi alternatifi değerlendirdin, neden onu seçmedin.
   Bu en değerli kısım; "şunu yaptım" değil "şunu şu yüzden yaptım" istiyorum.
3. **Yeni kavram** — ortaya yeni bir kavram çıktıysa (action chunking, EGL,
   normalization stats, delta actions gibi) 3-5 cümlede açıkla. Kavram yoksa bu
   başlığı atla, doldurmak için uydurma.
4. **Sırada ne var** — bir sonraki adım.

Bu özetleri aynı zamanda `docs/gunluk.md` dosyasına ekle. Proje sonunda bu dosya
hem benim not defterim hem de repoda "bu insan ne öğrendi" kanıtı olacak.

### Faz kapıları

Proje 6 faza bölünmüş (aşağıda). **Bir fazı bitirince dur ve bana sor.**
Onayımı almadan bir sonraki faza geçme. Faz sonunda:

- O fazın çıktısını ve kapı kriterlerinin geçtiğini kanıtla (gerçek komut çıktısıyla).
- O fazda öğrenilen en önemli 3 şeyi özetle.
- "Bunlardan birini daha derin anlatmamı ister misin?" diye sor.
- Onay bekle.

### Bilmediğin şeyi bilmiyorum de

Bir şeyden emin değilsen tahmin etme, "emin değilim, dokümana bakalım" de ve bak.
Uydurulmuş bir komut, beni bir saat kaybettirir; "bilmiyorum" beni beş dakika kaybettirir.

---

## 1. Projenin amacı ve iş gerekçesi

### Çözülen problem

Bugün bir robot hücresini yeni bir parçaya göre ayarlamak, bir entegratörün gelip
program yazmasını gerektiriyor. Sektör verisi: bir hücrenin devreye alma ve
programlama işçiliği tipik olarak **150-400 saat**, saati **125-200 dolar**.
Yani her yeni ürün varyantı **19.000-80.000 dolarlık** mühendislik işçiliği demek.
Bu yüzden çok çeşit / az adet üreten fabrikalar robot kullanamıyor.

### Bu projenin iddiası

Bir VLA modeli, operatörün elle gösterdiği birkaç düzine örnekle yeni bir parçayı
öğrenebilirse, o kalem entegratör faturasından operatör saatine iner.

### Ölçtüğümüz şey — projenin tek cümlelik özeti

> **Öğretme maliyeti eğrisi:** Daha önce görülmemiş bir görevde,
> K adet demonstration ile fine-tune edilen SmolVLA'nın başarı oranı,
> K'nın ve karşılık gelen operatör dakikasının fonksiyonu olarak.

Nihai çıktı tek bir başarı yüzdesi **değil**. Nihai çıktı, x ekseninde
"operatörün harcadığı süre", y ekseninde "görev başarı oranı" olan,
güven aralıklarıyla birlikte çizilmiş bir eğri. Bir fabrika sahibinin
bakıp karar verebileceği grafik bu.

### Bu proje ne DEĞİL

- Yeni bir model mimarisi önerisi değil. Mevcut SmolVLA'yı kullanıyoruz.
- SOTA kovalamıyoruz. Amaç en yüksek skor değil, **dürüst ölçüm**.
- Gerçek donanım yok. Tamamen simülasyon (LIBERO).

---

## 2. Donanım ve ortam kısıtları — pazarlık payı yok

Aşağıdaki tablo Faz 0'da **ölçülerek** doğrulanmıştır. Ham çıktılar:
`docs/00-environment.md`.

| Kalem | Değer (ölçülen) |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 **Laptop** GPU, Blackwell, compute capability **12.0 = sm_120** |
| VRAM | 8151 MiB nominal → **7.359 GiB gerçekten boş** (`mem_get_info`). Bütçe bu, 8 GB değil |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-31-generic |
| Sürücü | **595.84** (`Dual MIT/GPL` = açık çekirdek modülü ✓) |
| CUDA | Sürücü 13.2 destekliyor; torch **cu130** wheel'i kullanıyor |
| Python | 3.12.14 (conda ortamı: `lerobot`, Miniforge `~/miniforge3`) |
| PyTorch | **2.11.0+cu130** — LeRobot `torch>=2.7,<2.12.0` pinliyor, tavan bu |
| Render | `MUJOCO_GL=egl` (headless GPU rendering) |

### ~7.4 GB VRAM bu projenin merkezi kısıtı

LeRobot dokümanı SmolVLA fine-tune için `--batch_size=64` ve tek A100 öneriyor.
**Bizde A100 yok, 7.36 GiB var.** Bu yüzden:

- LoRA (PEFT) kullanacağız, full fine-tune değil.
- Batch size'ı deneyerek bulacağız, tahminle değil.
- **Her eğitim koşusunda peak VRAM'i ölçüp logla.** Bu bir metrik, yan bilgi değil.
- OOM aldığında **sessizce batch size düşürüp devam etme.** Dur, bana söyle,
  birlikte karar verelim. Sessiz düşürmeler deney tasarımını bozar ve sonuçları
  karşılaştırılamaz hale getirir.

### sm_120 tuzağı

Blackwell yeni bir mimari. Üçüncü parti repoların `requirements.txt` dosyaları
sık sık eski bir torch sürümü pinliyor ve o sürüm sm_120'yi tanımıyor.
**Hiçbir üçüncü parti requirements dosyasını körlemesine kurma.** Kurmadan önce
torch sürümünü kontrol et, gerekirse cu128+ tekerleğini koru.

**Faz 0'da bu tuzağın nasıl kapatıldığı (uygulanmaya devam edecek kural):**

1. torch **her zaman tek başına ve ilk** kurulur, LeRobot'tan önce. Böylece
   sm_120 çalışmıyorsa 40 paket sonra değil, ilk adımda öğrenilir.
2. Kurulumdan sonra sadece metadata'ya bakma — **gerçek bir GPU matmul çalıştır.**
   `get_arch_list()` içinde `sm_120` görmek yetmez, kernel gerçekten fırlamalı.
3. Her `pip install` bir **constraints dosyasıyla** yapılır:
   `-c torch-constraints.txt` içinde `torch==2.11.0`, `torchvision==0.26.0`.
   Böylece hiçbir transitive bağımlılık torch'u sessizce değiştiremez.
4. Kurulum sonrası torch sürümü **tekrar** doğrulanır.

LIBERO artık PyPI'dan `hf-libero>=0.1.4,<0.2.0` olarak geliyor; eski LIBERO
repo'sunun `requirements.txt`'i devrede değil. Bu, tuzağın en büyük kaynağını
ortadan kaldırdı — ama kural yürürlükte kalıyor.

---

## 3. Teknoloji yığını

- **LeRobot** (Hugging Face) — kaynaktan kurulacak. Base kurulum artık kasten
  *lightweight*, o yüzden dört extra gerekiyor: **`[libero,peft,smolvla,training]`**.
  Pinlenen commit: `2774d9bddcbbda50e697e162e89e7eaada8d7105` (2026-09-07).
- **SmolVLA** — checkpoint: `lerobot/smolvla_base`, 450M parametre
- **LIBERO** — simülasyon benchmark'ı, MuJoCo/robosuite tabanlı, 5 suite / 130 görev.
  PyPI paketi: `hf-libero>=0.1.4,<0.2.0` (`[libero]` extra'sı içinden gelir)
- **Dataset** — `lerobot/libero` (1.9 GB, MP4 formatı, önerilen). 1693 episode,
  273.465 frame, 40 görev = 4 suite × 10 görev → görev başına **~42 episode**

### Doğrulanmış komutlar (Faz 0'da güncel dokümana karşı kontrol edildi)

```bash
# Kurulum — torch ONCE ve tek basina, sonra constraints ile LeRobot
pip install "torch==2.11.*" torchvision                    # PyPI default = cu130
pip install -c torch-constraints.txt -e ".[libero,peft,smolvla,training]"
export MUJOCO_GL=egl

# LoRA ile egitim
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/libero \
  --dataset.revision=<commit-sha> \
  --policy.output_features=null \
  --policy.input_features=null \
  --policy.optimizer_lr=1e-3 \
  --policy.scheduler_decay_lr=1e-4 \
  --policy.push_to_hub=false \
  --env.type=libero --env.task=libero_object \
  --peft.method_type=LORA --peft.r=64 --peft.lora_alpha=64 \
  --batch_size=32 --steps=100000

# Degerlendirme
lerobot-eval \
  --policy.path=<checkpoint> \
  --env.type=libero --env.task=libero_object \
  --env.control_mode=relative \
  --env.init_states=true \
  --eval.n_episodes=10 --eval.batch_size=1 \
  --env.max_parallel_tasks=1
```

**Faz 0'da eklenen / değişen flag'ler ve neden önemli oldukları:**

| Flag | Neden |
|---|---|
| `--policy.output_features=null`<br>`--policy.input_features=null` | PEFT dokümanının resmî SmolVLA+LIBERO örneğinde var. Feature'ları checkpoint'ten devral |
| `--policy.scheduler_decay_lr=1e-4` | LoRA'da scheduler hedefi de 10 kat ölçekleniyor, sadece `optimizer_lr` değil |
| `--env.control_mode=relative\|absolute` | **Policy ile eşleşmek zorunda.** Farklı VLA checkpoint'leri farklı action parametrizasyonuyla eğitiliyor. Yanlış mod = sessizce düşük başarı. Tüm koşularda sabitle ve logla |
| `--env.init_states=true` | Sabit başlangıç durumları. Faz 1 determinizm kapısı için şart |
| `--dataset.revision=<sha>` | Hub dataset'leri yeniden yüklenebiliyor. Sonuç raporlarken pinle, yoksa sayılar karşılaştırılamaz |
| `--policy.push_to_hub=false` | Yasak #8'in komut seviyesindeki karşılığı |

Not: LoRA'da learning rate full fine-tune'a göre 10 kat yükseltiliyor (1e-4 yerine 1e-3).
Varsayılan LoRA hedef modülleri: LM expert'teki `q_proj`/`v_proj` + state/action projeksiyonları.
Farklı katman hedeflemek için `--peft.target_modules`, bir katmanı tam eğitmek için
`--peft.full_training_modules` var. `scaling = lora_alpha / r`.

**Faz 1 için dokümandan çıkan determinizm reçetesi:** iki policy'yi aynı episode'larda
karşılaştırmak için aynı `--seed`, `--env.init_states=true`, ve `--eval.batch_size` =
görev başına episode sayısı. `--env.hard_reset=false` daha hızlı ama **bit-identical
değil** — determinizm kapısında hard reset'te kal.

**Referans nokta:** Pi0.5 bu benchmark'ta ortalama %97.5 (Spatial 97 / Object 99 /
Goal 98 / Long 96). SmolVLA + LoRA + 7.4 GB bunun altında kalacak; eğrinin tavanı
için gerçekçi çıpa bu.

**Uyarı:** Yukarıdaki komutlar 2026-09-08'de güncel dokümana karşı doğrulandı.
LeRobot hızlı hareket ediyor — sonraki fazlarda uyumsuzluk görürsen dokümana uy,
bana da neyin değiştiğini söyle.

---

## 4. Faz planı

Her fazın sonunda **kapı kriteri** var. Kriter geçmeden sonraki faza geçme.

---

### Faz 0 — Ortam kurulumu ve doğrulama (2-3 gün)

Amaç: "model mi bozuk, kurulum mu bozuk" belirsizliğini projeye taşımamak.

1. Güncel LeRobot dokümanını oku (installation, LIBERO, SmolVLA, PEFT sayfaları).
   Yukarıdaki komutları doğrula.
2. Conda ortamı (`python=3.12`), sistem bağımlılıkları, LeRobot kaynaktan kurulum.
3. Aşağıdaki 6 kontrolü çalıştır ve **gerçek çıktılarını** `docs/00-environment.md`
   dosyasına yaz:

| # | Kontrol | Beklenen |
|---|---|---|
| 1 | `nvidia-smi` | Kart görünüyor, sürücü 580+ |
| 2 | `modinfo nvidia \| grep license` | `Dual MIT/GPL` (açık modül) |
| 3 | `torch.cuda.is_available()`, `get_device_capability()` | `True`, `(12, 0)` |
| 4 | `torch.cuda.mem_get_info()` | Boştaki VRAM — bu senin bütçen, not al |
| 5 | MuJoCo EGL render FPS testi | **Yüzlerce FPS.** Onlarca ise CPU'ya düşmüş, düzelt |
| 6 | Tek bir LIBERO episode ucu uca | Hatasız tamamlanıyor, video kaydediliyor |

5 numaralı kontrol kritik: MuJoCo EGL zinciri kırıldığında hata vermez, `llvmpipe`
adlı yazılımsal CPU render'ına düşer ve 10-50 kat yavaşlar. 1000 kare render edip
süreyi ölç, sonucu rakamla yaz.

**Kapı kriteri:** 6 kontrolün 6'sı da geçti ve `docs/00-environment.md` gerçek
sayılarla dolu. Commit at.

---

### Faz 1 — Ölçüm altyapısı (3-4 gün)

**Ölçeceğimiz şeyi, ölçüm aletinden önce kurmuyoruz.** Bu fazın tamamı eval harness'ı.

1. `lerobot-eval`'i sar: sabit seed listesi, görev başına episode sayısı,
   sonuçları `results/<run_id>/results.json` olarak kaydeden bir CLI.
2. JSON şeması: run_id, git commit hash, policy path, suite, task, seed,
   success (bool), episode length, timestamp, ortam bilgisi.
3. Özet tablo üreten bir script: görev başına başarı oranı + suite ortalaması
   + seed'ler üzerinden güven aralığı.
4. Başarısız rollout'ların videosunu kaydet — Faz 4'te hata analizi için lazım.

**Kapı kriteri — determinizm testi:** Aynı komutu iki kez çalıştır, `results.json`
içindeki success değerleri **birebir aynı** olmalı. Aynı değilse seed kontrolü
eksiktir, düzelt. Bu test geçmeden ilerlemek, sonraki her sayıyı şüpheli yapar.

---

### Faz 2 — Deney tasarımı ve veri hazırlığı (3-4 gün)

Burada "yeni parça" kavramını somutlaştırıyoruz.

**Kurgu:**
- **Fabrikanın mevcut hattı** = `libero_spatial` + `libero_goal` + `libero_10`
  üzerinde fine-tune edilmiş SmolVLA. Model bu görevleri "biliyor".
- **Yeni gelen parçalar** = `libero_object` görevleri. Model bunları hiç görmedi.
- **Öğretme** = held-out görev başına K adet demonstration ile fine-tune.

**Yapılacaklar:**
1. `lerobot/libero` datasetini indir, içeriğini incele. Görev başına kaç episode var,
   episode başına kaç frame, fps kaç? **Rakamları yaz.**
2. K değerlerini belirle. Hedef: `K ∈ {5, 10, 20, 40}`. Görev başına episode sayısı
   40'tan azsa üst ucu ona göre düşür ve bana söyle.
3. Her K için sabit seed'le alt-küme üreten bir script yaz. Aynı seed, aynı alt-küme.
4. **Operatör dakikası modeli.** K demonstration'ın kaç dakika operatör zamanına
   karşılık geldiğini hesapla. Formül:
   `(episode uzunluğu saniye) + (reset süresi varsayımı)` × K.
   Reset varsayımını açıkça yaz ve gerekçelendir. Bu sayı grafiğin x eksenine gidecek,
   yani projenin iş argümanının temeli. Uydurma, türet ve varsayımı görünür yap.

**Kapı kriteri:** `docs/02-experiment-design.md` içinde held-out split, K değerleri,
seed stratejisi ve operatör dakikası modeli yazılı. Alt-küme scripti çalışıyor ve
tekrarlanabilir.

---

### Faz 3 — Eğitim döngüsü (5-7 gün)

Projenin en compute yoğun kısmı. Sabırlı ol, koşular uzun sürecek.

1. **İlk iş: 8 GB'a ne sığıyor, deneyerek bul.** Batch size 1'den başlayıp yukarı
   çık, her denemede peak VRAM'i kaydet. Tahmin etme, ölç. Sonucu tablo halinde yaz.
2. LoRA konfigürasyonunu sabitle (r, alpha, target modules, lr) ve **bütün K
   değerleri için aynı konfigürasyonu kullan.** K dışında hiçbir şey değişmemeli,
   yoksa eğri bir şey ölçmez.
3. Her K için fine-tune koş. Her koşu şunları loglasın:
   K, seed, adım sayısı, wall-clock süre, peak VRAM, final loss, checkpoint yolu.
4. Checkpoint'leri düzenli isimlendir: `checkpoints/k{K}_seed{S}/`.
5. Hub'a hiçbir şey push etme — sormadan asla.

**Kapı kriteri:** Bütün K değerleri için checkpoint'ler mevcut, eğitim logları
tam, hiçbir koşuda sessiz bir konfigürasyon değişikliği olmamış.

---

### Faz 4 — Öğretme maliyeti eğrisi (3-4 gün)

1. Faz 1'deki harness'ı her checkpoint üzerinde koştur.
2. **Ana grafik:** x = operatör dakikası (üstte ikinci eksen olarak K),
   y = başarı oranı, seed'ler üzerinden güven aralığı bandıyla.
3. **Referans çizgileri:** K=0 (fine-tune edilmemiş base model) ve varsa
   tüm veriyle eğitilmiş üst sınır. Eğrinin nereden nereye gittiği bu iki çizgiyle anlaşılır.
4. **Hata analizi:** Başarısız rollout videolarını izle, hataları sınıflandır
   (yanlış nesne, kavrama ıskası, erken bırakma, hedefi ıskalama). K arttıkça
   hangi hata tipi kayboluyor? Bu, grafiğin altındaki en değerli paragraf olacak.
5. **Dürüstlük bölümü:** Eğri nerede doyuma ulaşıyor? Hangi görevlerde hiç
   öğrenemedi? Bunları saklamıyoruz, yazıyoruz.

**Kapı kriteri:** Grafik + tablo + hata analizi hazır. Rapordaki **her sayı**
`results/` altındaki bir dosyadan izlenebilir olmalı.

---

### Faz 5 — Teslim paketi (3-4 gün)

Repo'yu, hiç tanımadığın biri açtığında 5 dakikada ne yaptığını anlayacak hale getir.

1. **README.md (İngilizce):** Bir paragraf iş problemi (entegratör maliyeti argümanı),
   ana grafik en üstte, ne ölçtüğümüz, nasıl tekrarlanacağı, dürüst kısıtlar bölümü.
2. **Tek komutla tekrar üretim:** `make reproduce` veya eşdeğeri.
3. **Yan yana video:** solda base model yeni görevde başarısız, sağda K=20 sonrası başarılı.
4. **Limitations bölümü:** Simülasyon, tek model, tek benchmark, sim2real açığı
   ölçülmedi. Bunları açıkça yaz — dürüstlük, abartılmış iddiadan daha çok saygı görüyor.
5. `docs/gunluk.md`'yi temizle ve repoda bırak.

**Kapı kriteri:** Temiz git history, README ana grafikle açılıyor, reproduce scripti çalışıyor.

---

## 5. Yasaklar

Bunlar tercih değil, kural:

1. **Üçüncü parti `requirements.txt`'i körlemesine kurma.** sm_120 tuzağı.
2. **Çalıştırmadığın kod için "çalışıyor" deme.** "Çalışması lazım" cümlesini kurma, çalıştır.
3. **Bir hata aldığında hatayı özetleme, ham çıktıyı göster.** Sonra yorumla.
4. **Sonuç dosyasında olmayan hiçbir sayıyı rapora yazma.** Hatırladığın, tahmin
   ettiğin veya "yaklaşık" sayı yok. İzlenebilir olmayan sayı yok.
5. **OOM'u sessizce çözme.** Dur, söyle, karar birlikte verilsin.
6. **Deney tasarımını koşu ortasında değiştirme.** Değiştirmen gerekiyorsa dur,
   gerekçeyi söyle, onay al, sonra baştan koş.
7. **Bir seferde 150 satırdan fazla kod yazıp sonra çalıştırma.** Küçük parça yaz,
   çalıştır, doğrula, devam et.
8. **Hugging Face Hub'a sormadan hiçbir şey push etme.**
9. **Faz atlamak yok.** Kapı kriteri geçmeden ilerleme.

---

## 6. Repo yapısı

```
.
├── CLAUDE.md                 # bu dosya
├── README.md                 # İngilizce, ana grafikle açılıyor
├── Makefile                  # make setup / make eval / make train / make reproduce
├── pyproject.toml
├── docs/
│   ├── 00-environment.md     # Faz 0 doğrulama çıktıları
│   ├── 02-experiment-design.md
│   └── gunluk.md             # Türkçe öğrenme günlüğü
├── src/
│   ├── eval/                 # Faz 1 harness
│   ├── data/                 # Faz 2 K-shot alt-küme üretimi
│   ├── train/                # Faz 3 fine-tune sarmalayıcıları
│   └── analysis/             # Faz 4 grafik ve hata analizi
├── configs/                  # LoRA ve eğitim konfigürasyonları
├── results/                  # JSON sonuçlar (git'e giriyor)
├── checkpoints/              # .gitignore
└── assets/                   # grafikler, videolar
```

## 7. Git disiplini

- Her faz kapısında commit. Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`).
- Commit mesajları İngilizce.
- `checkpoints/`, dataset cache'i ve büyük video dosyaları `.gitignore`'da.
- `results/*.json` git'e **giriyor** — tekrar üretilebilirliğin kanıtı bu.
