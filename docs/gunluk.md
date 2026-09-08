# Günlük

Bu dosya projenin öğrenme defteri. Her iş bloğundan sonra dört başlık:
ne yaptım / neden böyle / yeni kavram / sırada ne var.

---

## 2026-09-08 — Faz 0: Ortam kurulumu ve doğrulama

### Blok 1 — Doküman doğrulaması (kod yazmadan önce)

**Ne yaptım.** LeRobot'un installation, LIBERO, SmolVLA ve PEFT sayfalarını
okudum, sonra repo'nun `pyproject.toml`'unu doğrudan çekip komutları ve extra
adlarını karşılaştırdım. CLAUDE.md'deki komutların hangileri hâlâ geçerli,
hangileri değişmiş çıkardım.

**Neden böyle.** Dokümana bakmadan kuruluma başlamak, sonra "acaba komut mu
yanlış, ortam mı bozuk" belirsizliğini üç hafta boyunca taşımak demekti.
Dokümanın kendisiyle de yetinmedim — dokümantasyon sayfaları repo'nun gerisinde
kalabiliyor, o yüzden `pyproject.toml`'u ham haliyle okudum. İyi ki de okudum:
`[libero]` extra'sının artık `hf-libero` PyPI paketini çektiğini ancak orada
gördüm.

Bulunan farklar özetle: base kurulum artık kasten *lightweight*, bu yüzden
`[libero]` ve `[peft]` yetmiyor, `[smolvla]` ve `[training]` de lazım. torch
`>=2.7,<2.12.0` pinli. Train komutuna üç yeni flag eklenmiş. Eval komutuna
determinizm ve action parametrizasyonu ile ilgili flag'ler gelmiş.

**Yeni kavram — `control_mode` (relative vs absolute).** LIBERO ortamı robota
iki farklı şekilde komut kabul ediyor: `relative` modda aksiyon "şu anki
pozisyondan şu kadar sapma", `absolute` modda "şu koordinata git". Bir VLA
checkpoint'i hangi parametrizasyonla eğitildiyse eval'de de o modda koşulmalı.
Tehlikeli tarafı şu: yanlış mod hata vermiyor, sadece robot saçmalıyor ve başarı
oranı düşük çıkıyor. Yani modeli suçlarsın, halbuki ortam ayarı yanlıştır.
Bu yüzden bütün koşularda sabitleyip loglayacağız.

**Sırada ne var.** Kurulum planını çıkarıp onaya sunmak.

---

### Blok 2 — Kurulum

**Ne yaptım.** Miniforge kurdum, `lerobot` conda ortamını (Python 3.12.14)
oluşturdum, torch'u **tek başına ve LeRobot'tan önce** kurdum, sm_120'yi
doğruladım, sonra LeRobot'u `[libero,peft,smolvla,training]` extra'larıyla
kaynaktan kurdum. Yolda üç şey patladı, üçünü de çözdüm (`docs/00-environment.md`
bölüm 5'te ham hatalarıyla birlikte duruyor).

**Neden böyle.**

*Sıra neden bu:* torch'u önce ve yalnız kurmak, sm_120 tuzağını 40 paket
kurulduktan sonra değil ilk iki dakikada yakalamak için. Metadata'ya bakmakla da
yetinmedim; `sm_120 in arch_list` `True` dönse bile gerçek kernel fırlamayabilir,
o yüzden gerçek bir fp32 ve bf16 matmul çalıştırdım.

*Wheel neden cu130:* Driver 595.84 / CUDA 13.2. Hem cu128 hem cu130 sm_120'yi
içeriyor ve ikisi de bizde çalışırdı. cu128'i `--index-url` ile zorlamak
"muhafazakâr" görünüyor ama torchvision ve torchcodec'i de aynı index'ten çekmeye
mecbur bırakıyor; biri PyPI'dan biri o index'ten gelirse ABI karışır ve bu tür
hatalar geç ve çirkin patlar. cu130 PyPI'ın default'u olduğu için bütün
bağımlılık ağacı kendiliğinden tutarlı geliyor. Bu bir tahmin değildi:
kurulum sırasında inen paketler `nvidia-cublas 13.1.0.3` ve `nvidia-cudnn-cu13`
oldu, yani cu130 olduğu kurulum anında kanıtlandı.

*Neden sudo yok:* Makinede `sudo` şifre istiyordu. `git`, `ffmpeg`, `cmake`,
`make` ve derleyicileri conda-forge'dan kurdum. Başta bir mecburiyetti, sonunda
avantaj oldu — kurulumun tamamı artık root istemiyor, yani başka bir makinede
tekrar üretmek daha kolay.

*constraints dosyası:* Her `pip install` `-c torch-constraints.txt` ile koştu
(`torch==2.11.0`, `torchvision==0.26.0`). Böylece hiçbir transitive bağımlılık
torch'u sessizce başka bir sürümle değiştiremedi. Kurulum sonrası tekrar
doğruladım: torch'a dokunulmamıştı.

**Yeni kavram — pip constraints dosyası.** `requirements.txt` "şunları kur"
der, constraints dosyası ise "kurarsan şu sürümü kur" der. Yani listedeki
paketi kurulum listesine *eklemez*, sadece o paket başka bir yerden gelirse
sürümünü sınırlar. sm_120 gibi tek bir wheel'e bağımlı olduğumuz durumlarda
tam olarak ihtiyacımız olan şey bu: LeRobot'un 130 bağımlılığından herhangi biri
"torch>=2.0" deyip daha eski bir tekerlek çekmeye kalkarsa constraints onu
engelliyor.

**Sırada ne var.** 6 doğrulama kontrolü.

---

### Blok 3 — Altı kontrol

**Ne yaptım.** Altı kontrolü de koştum, hepsi geçti. Ham çıktılar
`results/phase0_checks.log` dosyasında, yorumlanmış hali
`docs/00-environment.md` içinde. Tek komutla tekrarlanabiliyor:
`bash scripts/run_phase0_checks.sh`.

En kritik iki sonuç:

- **Kontrol 5 (EGL):** `GL_RENDERER = NVIDIA GeForce RTX 5060 Laptop GPU`,
  256×256'da **5539 FPS**. Kapı kriteri "yüzlerce FPS" idi, on kat üstündeyiz.
- **Kontrol 4 (VRAM):** Gerçek boş VRAM **7.359 GiB**. CLAUDE.md "8 GB"
  diyordu; kartın toplamı zaten 7.527 GiB ve Xorg 172 MiB tutuyor. CLAUDE.md'yi
  bu sayıyla güncelledim.

**Neden böyle.** Kontrol 5'i sadece FPS ölçerek yapmadım, `GL_RENDERER`
string'ini de bastırdım. Sebebi şu: FPS tek başına yanıltıcı olabilir — küçük
bir sahnede llvmpipe de fena olmayan bir sayı üretebilir, ya da GPU'daysan ama
başka bir darboğaz varsa düşük FPS görürsün ve yanlış şeyi kovalarsın.
`GL_RENDERER` ise tartışma bitiriyor: context'in sahibi kim, orada yazıyor.
Ayrıca render edilen karenin ortalama piksel değerini de bastırdım (122.39),
çünkü ikinci bir sessiz hata modu daha var: context açılır, render "başarılı"
döner ama kareler tamamen siyahtır.

Kontrol 6'da rastgele aksiyon kullandım ve `success: false` çıktı — bu beklenen
sonuç, çünkü test edilen şey politika değil, zincirin kendisi
(bddl → robosuite → MuJoCo → EGL → piksel → mp4).

**Yeni kavram — EGL ve sessiz CPU'ya düşme.** Ekransız (headless) bir makinede
OpenGL context'i açmanın yolu EGL. Sistemde birden fazla EGL "vendor" olabiliyor;
bizde hem `10_nvidia.json` hem `50_mesa.json` kayıtlı. Zincirde bir şey ters
giderse MuJoCo hata fırlatmıyor, Mesa'nın `llvmpipe` yazılımsal rasterleyicisine
düşüyor ve 10-50 kat yavaşlıyor. Bu projede render maliyeti doğrudan eval
maliyeti demek, yani fark edilmeyen bir llvmpipe Faz 4'ü günlere yayardı.
Tuzağın asıl kötülüğü sessiz olması: her şey "çalışıyor" görünüyor.

**Sırada ne var.** Faz 0 kapısı geçildi, Faz 1'e onay bekliyorum.

---

### Faz 0'da öğrenilen üç şey

1. **Sessiz hatalar, gürültülü hatalardan pahalıdır.** Bu fazın asıl işi kurulum
   değil, sessiz hata modlarını görünür kılacak kontrolleri yazmaktı: llvmpipe'a
   düşme, siyah kare üretme, metadata'da olup gerçekte olmayan sm_120 desteği,
   transitive bağımlılığın torch'u değiştirmesi. Dördü de patlamaz, sadece
   sonuçları bozar.

2. **Dokümana bakmak yetmiyor, kaynağa bakmak gerekiyor.** LeRobot dokümanı
   doğruydu ama eksikti. `pyproject.toml`'u ham haliyle okumak `hf-libero`
   geçişini, torch pinini ve gerçek extra listesini ortaya çıkardı.

3. **Kısıtı ölçtüğün an, kısıt sayı olur.** "8 GB VRAM" bir slogandı; 7.359 GiB
   bir bütçe. Aynı şekilde "EGL çalışıyor mu" bir endişeydi; 5539 FPS bir cevap.
   Faz 3'ün batch size aramasına artık gerçek bir tavanla giriyoruz.

---

## 2026-09-08 — Faz 1 başlangıcı: ölçüm aletinin içine bakmak

### Blok 4 — `lerobot-eval` kaynak okuması

**Ne yaptım.** Harness yazmadan önce `lerobot_eval.py`'yi (1115 satır) ve
`smolvla_base` checkpoint'inin `config.json`'unu okudum. Üç determinizm tuzağı ve
bir de kendi hatam çıktı.

**Neden böyle.** Faz 1'in kapı kriteri "aynı komut iki kez, birebir aynı sonuç".
Bunu sağlayacak sarmalayıcıyı, sardığım şeyin seed'i nasıl dağıttığını bilmeden
yazamazdım. Dokümanda bu bilgi yok; kodda var.

Bulunanlar:

1. **Seed formülü:** episode *i* → `start_seed + i`, batch gruplamasından
   bağımsız. Deterministik.
2. **Ama `eval_info.json`'da per-episode seed yok.** `eval_one`, `per_episode`
   sözlüğünü `TaskMetrics`'e çevirirken `seed` alanını düşürüyor. Harness'ın
   yeniden türetmesi gerekiyor — ve türetmeyi *doğrulaması* gerekiyor.
3. **Her task aynı `start_seed` ile başlıyor**, yani seed'ler global olarak
   benzersiz değil. Şemanın anahtarı `(suite, task_id, episode_ix)` olmalı.
4. **En sinsisi:** `EvalConfig.batch_size` varsayılanı `0` ve bu "CPU çekirdek
   sayısına göre otomatik" demek. Yani batch_size vermezsen sonuç makineye bağlı
   hale geliyor. Determinizm kapısı için açıkça sabitlemek şart.

**Kendi hatam.** CLAUDE.md'ye `--policy.input_features=null` için "feature'ları
checkpoint'ten devral" yazmıştım. Tam tersiymiş: `policies.py:58` ve
`factory.py:334`'e göre `null`, "checkpoint'in feature tanımını **at**,
dataset'ten türet" demek. Düzelttim. Ders: dokümanın örnek komutundaki bir
flag'in ne işe yaradığını, o flag'i kodda aramadan yazmayacağım.

**Yeni kavram — action chunking.** `smolvla_base`'de `chunk_size = 50` ve
`n_action_steps = 50`. Model her forward'da tek bir aksiyon değil, **50
aksiyonluk bir dizi** üretiyor ve hepsini sırayla uyguluyor. Sebebi: manipülasyon
görevlerinde ardışık aksiyonlar birbirine çok bağımlı, teker teker üretmek
titrek ve tutarsız hareket veriyor; ayrıca her adımda 450M parametreli bir VLM'i
çalıştırmak çok pahalı. Chunk'lamak hem hareketi yumuşatıyor hem forward sayısını
50 kat düşürüyor. Bedeli tepki hızı: model 50 adım boyunca dünyaya yeniden
bakmıyor, o yüzden beklenmedik bir değişikliğe geç tepki veriyor. LIBERO dokümanı
Pi0.5'i `n_action_steps=10` ile koşuyor — yani bu bir ayar knob'ı, sabit değil.

**Yeni kavram — feature spec uyuşmazlığı ve normalization stats.**
`smolvla_base` state=[6], 3 kamera, action=[6] beyan ediyor; LIBERO ise state=8,
2 kamera, action=7 veriyor. Model bunu içeride `max_state_dim=32` padding'iyle
kaldırabiliyor, ama config seviyesinde eşleşme zorunlu — `input_features=null`
bunun için var. Bununla bağlantılı ikinci konu: `normalization_mapping`
STATE ve ACTION için `MEAN_STD` diyor, yani modelin girdileri normalize etmek
için kullandığı ortalama/standart sapma değerleri checkpoint'in içine gömülü
(`policy_preprocessor_step_5_normalizer_processor.safetensors`). Feature seti
değişince bu istatistikler de dataset'ten yeniden hesaplanmak zorunda; yoksa
model doğru şekle sahip ama yanlış ölçekte veri görür ve bu da sessiz bir hata.

**Sırada ne var.** `src/eval/run_eval.py`: sabit seed listesi, zorunlu explicit
batch_size, `eval_info.json`'u projenin şemasına çeviren sarmalayıcı. Ardından
determinizm kapısı.

---

### Blok 5 — `n_action_steps` kararı (ölçümle)

**Ne yaptım.** 50 (SmolVLA varsayılanı) ve 10 (LIBERO dokümanının Pi0.5 için
kullandığı değer) arasında, her şeyi sabit tutup sadece bu parametreyi
değiştirerek 10'ar episode koştum. Peak VRAM'i de ölçen bir sarmalayıcı yazdım
(`scripts/run_with_gpu_peak.sh`) — Faz 3 zaten bunu her koşuda isteyecek.

**Neden böyle.** Kararı tahminle vermek yerine ölçmek istedim ve bu iyi oldu,
çünkü **tahminim yanlıştı.** 5 kat daha fazla policy forward'ın 5 kat
yavaşlatmasını bekliyordum; gerçekte 1.44 kat yavaşlattı. Sebep: darboğaz VLM
değil, MuJoCo'nun kendisi. Faz 0'da "sim darboğaz olmayacak, inference olacak"
demiştim — bu ölçüm beni yanlışladı. Öğrendiğim şey parametrenin değeri değil,
sistemin nerede zaman harcadığı.

Karar `n_action_steps=10` oldu: hız cezası bütün Faz 4 boyunca ~1.5 saat, buna
karşılık yayınlanmış Pi0.5 sonuçlarıyla aynı protokolde oluyoruz.

**Dürüstlük.** Bu deney doğruluk sorusunu **cevaplamadı** — `smolvla_base`
LIBERO'da %0 aldığı için iki kol ayırt edilemedi. `results.json` içine `caveat`
alanı olarak yazdım. Faz 3'te K=5 çıkınca teyit edilecek.

**Sırada ne var.** Harness.

---

### Blok 6 — Faz 1: ölçüm altyapısı ve determinizm kapısı

**Ne yaptım.** `src/eval/run_eval.py` (sarmalayıcı + şema),
`src/eval/check_determinism.py` (kapı), `src/eval/summarize.py` (tablo + güven
aralığı), `scripts/verify_seed_mapping.py` (seed formülü doğrulaması). Kapı
geçti, `docs/01-eval-harness.md` yazıldı.

**Neden böyle.** İki karar açıklamaya değer.

*Seed formülünü neden ayrıca doğruladım:* Kodu okuyunca `seed = start_seed +
episode_ix` olduğu anlaşılıyordu, ama `eval_info.json` seed'i kaydetmediği için
harness'ın bunu **yeniden türetmesi** gerekiyordu. Okuduğum bir satıra dayanarak
şemaya sayı yazmak, o satırı yanlış okumuşsam bütün Faz 4'ü sessizce
zehirlerdi. `eval_policy`'yi spy'layıp gerçek seed'leri bastırdım. Formül
doğrulandı — ve bonus olarak ikinci bir şey öğrendim: **bütün task'lar aynı seed
dizisini kullanıyor**, yani seed episode'u tek başına tanımlamıyor. Şemanın
anahtarını `(suite, task_id, episode_ix)` yapmam bu yüzden.

*Determinizm kapısını neden sadece `success`'e bakarak geçmedim:* CLAUDE.md
"success değerleri birebir aynı olmalı" diyor. Aynı çıktı. Ama `smolvla_base`
her episode'da başarısız olduğu için o vektör baştan sona `False` — sistem ne
kadar rastgele olursa olsun iki koşuda da aynı çıkardı. Yani kapıyı geçmiş
görünüp hiçbir şey kanıtlamamış olurduk. Onun yerine **video byte'larını**
karşılaştırdım: 6 videonun 6'sı da birebir aynı. Video bütün piksel yörüngesini
kodluyor, yani fizikte, render'da veya inference'ta en ufak bir kararsızlık
olsa değişirdi. Bir de tersini kontrol ettim: 6 video **birbirinden** farklı,
yani test "hiçbir şey olmuyor" diye geçmiyor.

**Yeni kavram — Wilson güven aralığı ve "0/6 sıfır demek değildir".**
Elimizde ikili (başarılı/başarısız) sonuçlar var ve oran 0'a çok yakın. Bu
durumda okulda öğretilen normal yaklaşım (`p ± 1.96·√(p(1-p)/n)`) çöküyor:
p=0 için genişlik sıfır çıkıyor, yani "6 denemede 0 başarı" için "%0, kesin"
diyor. Saçma. Wilson aralığı bu problemi düzeltiyor ve gerçek cevabı veriyor:

| episode | gözlenen | Wilson %95 |
|---|---|---|
| 6 | 0/6 | **[%0, %39]** |
| 18 | 0/18 | **[%0, %17.6]** |

Yani 6 episode'dan sonra söyleyebileceğimiz tek şey "başarı %39'un altında".
Faz 4'ün eğrisi bu aralıklar olmadan bir şey ifade etmez — özellikle K=5 gibi
küçük değerlerde gerçek başarı oranı düşükken, dar bir örneklemden emin
konuşmak en kolay yanılma biçimi.

İkinci aralık türü de var: seed'ler arası yayılma. Bu ikisi **farklı soruları**
cevaplıyor — Wilson "bu kadar denemeyle oranı ne kadar hassas biliyoruz", seed
yayılması "yeniden koşarsak cevap ne kadar oynar". Faz 4'ün hata çubukları
ikincisi olacak, ama en az 3 seed gerekiyor; script 2'de "sadece betimleyici"
diye etiketliyor, 1'de hiç göstermiyor.

**Sırada ne var.** Faz 2 — deney tasarımı ve veri hazırlığı. Onay bekliyorum.
