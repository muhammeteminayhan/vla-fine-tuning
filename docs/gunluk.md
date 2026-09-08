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
