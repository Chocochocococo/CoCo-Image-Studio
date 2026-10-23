**English** · [繁體中文](README.zh-TW.md)

# CoCo Image Studio

A local workbench for generating images with AI and training LoRAs.

**[Download the latest version](https://github.com/Chocochocococo/CoCo-Image-Studio/releases/latest)** · [Website and manual](https://studio.cocos-homes.com/en/)

---

## Requirements

| Item | Requirement |
|---|---|
| OS | **Windows** (`CoCo Image Studio.exe` is the only entry point) |
| Graphics card | Full install: **NVIDIA, RTX 20 series or newer** (generate, refine and train have no CPU mode; AMD, Intel and Mac aren't supported). The light install needs no graphics card |
| Driver | Full install: **580 or newer** (PyTorch is the CUDA 13 build) |
| Python | Nothing to install: the first run downloads the official Python 3.11.9 into the `data` folder |
| VRAM | 8 GB works (set VRAM to "Low" under "More" on the generate panel, or turn on "⚙ Settings → Generate → Store the main model in fp8"); 12 GB or more is comfortable |

The Anima architecture also needs **RTX 30 series or newer**: it runs in bfloat16, which Turing (20 series) doesn't
support natively, so it would be very slow. SDXL and SD 1.5 run in fp16 and are fine on the 20 series.

Models aren't included. `data\models\` has one folder per kind, named the same as in ComfyUI (`checkpoints`, `diffusion_models`, `loras`, `vae`, `text_encoders`, `embeddings`, `controlnet`, `upscale_models`, `adetailer`, `rembg`…); put each file in the folder for its use. You can also point "Settings → Resource folders" at another tool's model folder (StabilityMatrix, ComfyUI, Forge); CoCo still reads its own folders too.
If you don't have a model yet, download the trial model (Anima, about 5 GB) from "Settings → Resource folders → Download models".

### Will it touch my existing setup?

No. PyTorch and every package are installed in `data\venv\`, with versions pinned by `data\CoCo.bat` and
the lock files (`requirements-lock.txt`: torch 2.14.0+cu130 / diffusers 0.39.0; the light install uses the CPU build of torch plus
the packages of `requirements-lite.txt` at the same versions). Every download is checked against the SHA256 recorded in the lock files,
separate from the Python, A1111 and ComfyUI already on your system. Python is CoCo's own copy too (`data\python\`, the official python.org package, SHA512 checked).

Downloaded diffusers models are stored in `data\hf_cache\`, so they don't pile up in your user folder on drive C. The packages downloaded while installing are kept in `data\pip-cache\` and deleted once the install is done; the graphics driver's compile cache is kept in `data\cuda-cache\`.

---

## Getting started

1. Download `CoCo-Image-Studio.zip` from the [latest release](https://github.com/Chocochocococo/CoCo-Image-Studio/releases/latest) and extract it wherever you like, for example `D:\CoCo-Image-Studio`.
   Keep the path short, and don't put it in a synced folder such as OneDrive or Dropbox.
2. Double-click **`CoCo Image Studio.exe`**
   > The first time, Windows may show "Windows protected your PC": choose "More info", then "Run anyway".
   > CoCo's launcher has no paid code-signing certificate, so Windows doesn't recognise it. All it does is run `data\CoCo.bat`
   > (source in `data\tools\launcher.cs`). It won't ask again.
3. The first run asks which install you want; type 1 or 2:
   - **1 Full** (about 4 GB, needs an NVIDIA graphics card): every feature.
   - **2 Light** (about 1 GB): the gallery, explore, compare, recombine, output, the library, and the prompts and tags on the generate panel. It can't generate images, and refine and training are preview only. Explore runs on the CPU.

   If no NVIDIA graphics card is found, Light is recommended. It then asks whether to add a CoCo shortcut to the desktop and the Start menu (Y/N).
   After that it sets up its environment and installs the packages, then opens the Studio.
4. From then on, it refreshes the gallery index first, then opens the Studio.

> To generate after a light install: in "⚙ Settings → Other", press "Switch to full install", then close and reopen CoCo; it installs the rest automatically.

> The first run downloads the official Python 3.11.9 (about 17 MB) into the `data` folder; any Python already on your computer isn't affected. Only without an internet connection does it use a Python 3.11–3.13 already on your computer.
> You can also pin `CoCo Image Studio.exe` to the taskbar.
> You can move or rename the whole CoCo folder without reinstalling. Recreate the shortcut afterwards: delete `data\venv\.coco-shortcut`, then open CoCo and it asks again.

### Uninstalling

Everything CoCo uses is in its own folder, so deleting the folder removes it completely:

1. Generated images are saved in `data\outputs` by default; move out the ones you want to keep.
2. If you created the shortcuts, press "Remove shortcuts" in "⚙ Settings → Other".
3. Close CoCo, then delete the whole CoCo folder.

---

## Settings live in the interface

Generation parameters (model, LoRA, prompt, steps, CFG, sampler, VRAM) are on the generate panel;
everything else (gallery folders, resource folders…) is in "⚙ Settings" at the top right.

Changes are saved to `data\settings.json`; **`data\config.py` is never rewritten**.
`config.py` holds the defaults. You can still edit it by hand; the two are merged.

---

## Folders

To open CoCo, use `CoCo Image Studio.exe`. Everything else (the program, Python, packages, your images, models, settings) is in the `data` folder next to it.

```
CoCo-Image-Studio\
  CoCo Image Studio.exe  ← the only entry point (has the CoCo icon; it only runs data\CoCo.bat)
  data\             ← everything CoCo has
    CoCo.bat          ← installs on the first run, then scans and starts (running it directly works too)
    outputs\          ← generated images
    models\           ← models, one folder per kind (checkpoints, loras, vae…); add your own
    settings.json     ← settings saved by the interface (created automatically)
    config.py         ← defaults
    variables.txt     ← variables (name = value)
    wildcards\        ← wildcard list files
    install_mode.txt  ← install type (full/lite, created on the first install)
    python\           ← CoCo's own Python (downloaded on the first install)
    venv\             ← created automatically
    requirements-lock.txt      ← packages with their SHA256 (every install is hash-checked)
    requirements-lite.txt      ← packages for the light install (installed from requirements-lite-lock.txt)
    requirements-torch-*.txt   ← torch / torchvision (CUDA or CPU build)
    core\             ← the Studio core
      conditioning.py   A1111-style prompt encoding
      template.py       template expansion
      gallery_index.py  gallery indexer
      settings.py       settings read and write
    server\           ← local server: HTTP routes, gallery, generation queue, settings
    web\              ← the interface (index.html, js\, css\, locales\)
    tools\
      studio.py         startup entry (the rest is in server\)
```

---

## Template syntax

| Syntax | What it does |
|---|---|
| `[[A\|B\|C]]` | Cartesian expansion; several of them multiply (character sheets) |
| `{A\|B\|C}` | Pick one at random |
| `{3::A\|B}` | Weighted random (A is three times as likely as B) |
| `{A\|}` | Sometimes picks nothing |
| `__pose__` | A random line from `data\wildcards\pose.txt` |
| `[[__pose__]]` | One set for every line in that file |
| `{set:c=red\|blue}` `{get:c}` | Pick once, then reuse the same value |
| `{2$$A\|B\|C}` | Pick 2 without repeats |
| `{1-3$$A\|B\|C}` | Pick 1 to 3 |
| `{2$$ and $$A\|B\|C}` | Custom separator |
| `${c={red\|blue}}` `${c}` | Same as above, Dynamic Prompts style |
| `# comment` | Ignored to the end of the line |

### Variables vs wildcards

| | What it is | Where it's stored | How to use it |
|---|---|---|---|
| **Variable** | A name for a fixed piece of text | `data\variables.txt` | `${my_oc}` |
| **Wildcard** | A file with many lines of options | `data\wildcards\*.txt` | `__pose__` picks a random line |

Add and edit variables with the form in the "Variables" tab of the generate panel's "Syntax" window; no need to edit the file by hand.
Edit wildcards in the "Wildcards" tab of the same window, one option per line.

```
# variables.txt
my_oc     = silver long hair, blue eyes, hair ribbon
quality   = masterpiece, best quality, amazing quality, newest
```

**The prompt field is itself a template**: write dynamic syntax straight into it, and "Syntax → Preview" shows the expanded prompts.

### Fixed seed (for testing LoRAs)

Press "Fixed" next to the seed on the generate panel and the whole batch shares one seed, so the only differences come from the prompt.
Add `[[<lora:A:0.8>|<lora:B:0.8>|]]` and you have a standard LoRA comparison.

With "Lock random picks" in "Syntax → Preview" turned on (the default), every combination in a batch shares the same random picks:
the pose and expression change but the character stays the same, which is what a character sheet needs.

---

## Model formats

All three are supported:

| Format | What it looks like | How to use it |
|---|---|---|
| **Single file** | `model.safetensors` | Put it in a model folder, then click its card in the library's "Models" tab |
| **diffusers folder** | A folder with `model_index.json` | Same as above |
| **Split parts** | Separate transformer / vae / text_encoder files | Pick the main model as usual, then click a card in the "VAE" and "Encoders" tabs (if you don't, CoCo finds matching ones) |

The VAE and Encoders tabs are how you override parts. Both have a "Built into the model" card;
picking it means no override, and picking any other card overrides that part. That lets several models share one VAE and saves several GB.

### Do I have to download diffusers models myself?

Put the whole folder into a model folder; no internet connection is needed.

Models that are downloaded automatically are stored in `data\hf_cache\`, not in your user folder on drive C.
To clear them, just delete that folder.

---

## Troubleshooting

**This PyTorch build doesn't support your GPU (sm_120)**
The RTX 50 series needs a build for CUDA 12.8 or newer. Delete `data\venv\.coco-ready` and reopen CoCo;
it checks the installed versions and, if they don't match, reinstalls the tested cu130 build.
(`CoCo.bat` also does this by itself when it updates the pinned versions, so you don't have to delete anything then.)

**At startup it says it "cannot see your GPU"**
The PyTorch version is right but it can't see the graphics card, so **reinstalling won't help**: the problem is the driver or the hardware.
Update the NVIDIA driver first (CUDA 13 needs 580 or newer), then check that the card looks normal in Device Manager.
In this state the gallery still works; only generation fails.

**A LoRA doesn't apply**
CoCo merges LoRAs into the model itself and supports regular LoRAs; LyCORIS LoHa and LoKr aren't supported yet.
The generation log shows how many modules each LoRA was merged into. When it says a LoRA "matches no modules", the format usually isn't supported,
or the LoRA doesn't fit the main model's architecture (for example an SDXL LoRA on Anima).

**Gallery layouts**

"Layout → Arrangement" in the wall's title bar switches between three layouts:

| Option | Also known as | Order | Gaps |
|---|---|---|---|
| Justified (default) | justified layout | kept | every row filled |
| Masonry · packed | masonry, shortest column | can jump | tight |
| Masonry · in order | masonry, round-robin | kept | uneven bottom |

Justified groups consecutive images into a row and scales the row to fill the width; masonry stacks images into columns.
They are different layout methods, not variants of one.

**The image viewer says "No generation settings."**
Those images had their metadata stripped (usually downloaded from elsewhere or edited). It isn't an error.

**At startup you see `is not recognized as an internal command`**
Make sure the whole archive was unzipped: `CoCo Image Studio.exe` should sit next to a `data` folder that holds `CoCo.bat`, `requirements-lock.txt`, `requirements-lite.txt`, `core\`,
`tools\studio.py`, `server\` and `web\`.

**V-Pred models (such as NoobAI V-Pred)**
Just pick them: CoCo recognises them from the model file and switches to the right method. The official advice is the Euler sampler, CFG 4–5 and 28–35 steps.
If you pick a sampler that doesn't fit (LMS, Heun, DPM2, DPM++ SDE, DEIS, SA-Solver), CoCo switches to Euler and says so in the generation log.

**Which VAE to use**

The library's VAE tab (click the "VAE" slot on the generate panel to open it) has three choices:

| Option | What it means |
|---|---|
| **sdxl-vae-fp16-fix** (default) | The original SDXL VAE overflows in fp16 and produces black images. diffusers works around this by decoding in fp32, and VRAM spikes at that moment. The fixed version stays in fp16 the whole way, with no spike |
| Built into the model | Exactly what the model gives, but VRAM spikes while SDXL decodes |
| Custom | Files in `data\models\vae\` and the VAE folders you set |

"Staged decode" under "More" on the generate panel is set to Auto by default; for large images it unloads the UNet before the VAE decode.

**VRAM use is higher than in Forge**
Set VRAM to "Balanced" or "Low" under "More" on the generate panel. For SDXL, also keep "Staged decode"
on "Auto"; in Balanced and Low, Anima unloads the parts it isn't using at each stage.
If that's still not enough, turn on "⚙ Settings → Generate → Store the main model in fp8": the main model takes half the graphics memory,
it runs a little slower, and images differ slightly from when it's off.

---

## Reporting problems

If you run into a problem or have a suggestion, please open an issue on GitHub.

This project **doesn't accept code contributions (pull requests)**: the author can't review other people's code themselves, so there's no way to make sure it contains nothing malicious.

---

## Sponsor

CoCo is free and has no ads. If it has helped you, you're welcome to sponsor the author's continued work on it.

| Platform | Best for | Link |
|---|---|---|
| ECPay | Taiwan: credit card, ATM, convenience stores and more | [Sponsor](https://p.ecpay.com.tw/D6622EF) |
| Ko-fi | Outside Taiwan: credit card, PayPal | [Sponsor](https://ko-fi.com/chocochocococo) |

You can also find these in CoCo: ⚙ Settings → About.

---

## License

CoCo Image Studio is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE), plus an additional permission:
**the program itself may not be used commercially; the images you generate with it may be used for anything.**
Below is a short summary; the [LICENSE](LICENSE) text is what counts.

**Allowed, no need to ask the author:**

- Use images generated with CoCo for anything: commissions, sales, licensing and commercial work are all fine
- Personal hobbies, learning, research, testing and private entertainment
- Use by schools, charities, public-interest organisations and government bodies
- Modifying the program, or sharing the original or a modified version: include the LICENSE text and the copyright notice on its first line (the `Required Notice:` line)

**Not allowed without the author's permission:**

- Selling CoCo or a modified version, or charging for copies
- Putting CoCo into a paid product, or using it to run a paid service

**Also:**

- Third-party files and code bundled with the project (fonts, tokenizers, code copied from diffusers and so on) keep their own licenses; see [THIRD_PARTY_NOTICES.md](data/THIRD_PARTY_NOTICES.md) (in Traditional Chinese).
- CoCo doesn't include any models for generation. Models you add yourself, and the images you generate with them, are also subject to each model's own license.
