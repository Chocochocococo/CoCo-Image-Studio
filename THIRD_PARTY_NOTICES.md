# 第三方聲明（Third-Party Notices）

CoCo Image Studio 本身的程式採用 PolyForm Noncommercial License 1.0.0（見 [LICENSE](LICENSE)）。
下面列出的第三方檔案與程式**各自保留原本的授權**，不適用 CoCo 的授權。

CoCo 不附帶任何生成用的模型。使用者自己放進來的模型（Anima、SDXL、Z-Image、Krea 2、Qwen Image、LoRA、找臉模型、
放大模型…），以及用它們生成的圖，受各個模型自己的授權規範，跟 CoCo 的授權無關。

---

## 一、隨 CoCo 附帶的檔案

| 檔案 | 內容 | 來源 | 授權 |
|---|---|---|---|
| `core/vendor/qwenimage21/`（三個 `.py`） | Qwen Image 2.1 的模型、VAE、管線 | [huggingface/diffusers](https://github.com/huggingface/diffusers) commit `8b33bfc`；Copyright 2026 Qwen-Image Team, The HuggingFace Team | Apache-2.0 |
| `core/vendor/prodigyopt/`（prodigy.py） | 訓練 LoRA 的 Prodigy 優化器（prodigyopt 1.1.2，原封不動） | [konstmish/prodigy](https://github.com/konstmish/prodigy)；Copyright (c) 2023 Konstantin Mishchenko | MIT（全文：`core/vendor/prodigyopt/LICENSE`） |
| `assets/anima/tokenizer/` | Qwen3 的 tokenizer（Anima 的文字編碼器用） | [Qwen](https://huggingface.co/Qwen)（Alibaba Cloud） | Apache-2.0 |
| `assets/anima/tokenizer_2/` | T5 的 tokenizer（Anima 的文字條件、FLUX.1 的 T5 用） | [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | Apache-2.0 |
| `assets/flux1/tokenizer/` | CLIP-L 的 tokenizer（FLUX.1 用） | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | MIT |
| `assets/anima/vae/config.json` | Qwen Image VAE 的設定 | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | Apache-2.0 |
| `assets/anima/scheduler/scheduler_config.json` | Anima 的排程設定（只有 shift 3.0 等幾個數值） | Anima 官方的設定 | — |
| `web/fonts/LXGWWenKaiTC-Medium.ttf` | 霞鶩文楷 TC | [lxgw/LxgwWenkaiTC](https://github.com/lxgw/LxgwWenkaiTC) | SIL Open Font License 1.1（全文：`web/fonts/LXGWWenKaiTC-OFL.txt`） |
| `web/fonts/ibm-plex-mono-*.woff2` | IBM Plex Mono | [IBM/plex](https://github.com/IBM/plex) | SIL Open Font License 1.1（全文：`web/fonts/IBMPlexMono-LICENSE.txt`） |
| `assets/tags/base.db` | 標籤庫：22 萬個標籤的名稱、分類、使用次數與別名 | 標籤名稱、網站分類、使用次數、別名來自 [DraconicDragon/dbr-e621-lists-archive](https://github.com/DraconicDragon/dbr-e621-lists-archive) 的 `danbooru_e621_merged.csv`（整理自 Danbooru 與 e621 公開的標籤資料）。細分類（髮型、服裝、表情…）、NSFW 標記與 294 筆中文翻譯由 CoCo 作者的匯入工具產生，其中約 1,400 個標籤的細分類參考了一份網路上分享、沒有署名的標籤表的分組 | 標籤清單：The Unlicense |
| `assets/tags/zh_TW.tsv` | 標籤的中文翻譯與細分類（疊在 `base.db` 上） | CoCo 自己做的：用 AI 翻譯與分類，再抽查修正 | 同 CoCo（見 LICENSE） |

`core/vendor/qwenimage21/` 改過的地方只有 import（相對匯入改成從安裝好的 diffusers 套件匯入），
各檔案開頭保留原本的授權聲明，詳見那個資料夾的 README.md。

## 二、照別人的實作改寫的程式

| 檔案 | 改寫自 | 授權 | 改了什麼 |
|---|---|---|---|
| `core/birefnet.py` | [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) 的推論部分；Copyright (c) 2024 ZhengPeng | MIT（全文見下方附錄 B） | 只留推論用的部分，不需要 timm、kornia、einops；寫法改成 CoCo 的 |
| `core/bgremove.py` 的 IS-Net 結構 | [xuebinqin/DIS](https://github.com/xuebinqin/DIS)（Xuebin Qin 等人） | Apache-2.0（全文見下方附錄 A） | 只留推論用的部分；權重名稱的前綴自動拿掉 |
| `core/qwen21.py` 的 `edit_embeds` | diffusers 的 `QwenImage21Pipeline._get_qwen_prompt_embeds` | Apache-2.0（全文見下方附錄 A） | 換成 CoCo 的元件、加上快取 |

## 三、自己寫、只是相容別人格式的程式（致謝）

下面這些程式是 CoCo 自己寫的，沒有包含對方的程式碼，只是讀得懂同樣的模型檔或照同樣的慣例。列出來表示感謝：

- **YOLOv8**（`core/yolo.py`）：讀 [ultralytics](https://github.com/ultralytics/ultralytics) 格式的找臉模型（例如 ADetailer 的模型），照檔案裡記的網路結構自己組網路。CoCo 不包含、也不需要 ultralytics。
- **ESRGAN／Real-ESRGAN 的 RRDBNet**（`core/upscale.py`）：讀 [ESRGAN](https://github.com/xinntao/ESRGAN)、[BasicSR](https://github.com/XPixelGroup/BasicSR) 兩種寫法的放大模型。
- **OpenPose 的骨架畫法**（`core/controlnet_prep.py`）：點的順序與顏色照 OpenPose 的慣例，ControlNet 的姿勢模型都是用這種圖訓練的。
- **int8 + convrot 的格式**（`core/qwen21.py`）：照 [comfy_kitchen](https://pypi.org/project/comfy-kitchen/) 的說明讀 ComfyUI 的 int8 模型檔。
- **取樣器**（`core/samplers.py`）：照論文的演算法自己實作。
- **SDXL LoRA 訓練**（`core/sdxl_train.py`）：做法照 [OneTrainer](https://github.com/Nerogar/OneTrainer) 的行為（LoRA 掛的層、分桶、存檔的名稱）自己寫，沒有包含 OneTrainer 的程式碼。
- **Anima LoRA 訓練**（`core/anima_train.py`）：做法照 [sd-scripts](https://github.com/kohya-ss/sd-scripts)（Apache 2.0）與以它為底的 Anima-Standalone-Trainer 的行為（LoRA 掛的層與名稱、時間步、caption dropout）；分桶的算法照 sd-scripts 的`BucketManager.select_bucket`（bucket_no_upscale）改寫。Copyright (c) kohya-ss，Apache License 2.0。
- **Slider LoRA 訓練**（`core/slider_train.py`）：做法照 [ai-toolkit](https://github.com/ostris/ai-toolkit)（MIT）新版的 ConceptSliderTrainer（目標的算法、norm_like、loss 的組合）自己寫，沒有包含 ai-toolkit 的程式碼。Copyright (c) Ostris, LLC，MIT License。
- **GGUF 與 JoyCaption**（`core/gguf.py`、`core/joycaption.py`）：讀 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 GGUF 格式，照 [JoyCaption](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)（LLaVA：SigLIP2＋Llama 3.1）的官方設定自己組模型。CoCo 不包含、也不需要 llama.cpp；模型檔由使用者自己準備，各依自己的授權（Llama 3.1 Community License 等）。
- **WD14 tagger**（`core/wdtagger.py`）：讀 SmilingWolf 的 [WD v3 tagger](https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3) 模型檔（ONNX 或 timm 的 safetensors），用 PyTorch 自己組 ViT。CoCo 不包含、也不需要 onnxruntime 與 timm。

## 四、安裝時才裝的套件（不隨 CoCo 附帶）

`CoCo.bat` 第一次執行時從 PyPI 與 PyTorch 官方安裝下面這些套件到專案底下的 `venv\`，版本見 `requirements-lock.txt`。
它們不在 CoCo 的檔案裡，各自依照自己的授權。

| 套件 | 授權 |
|---|---|
| PyTorch、torchvision | BSD-3-Clause |
| NVIDIA CUDA 執行階段（隨 PyTorch 安裝的 `nvidia-*` 套件） | NVIDIA 軟體授權 |
| diffusers、transformers、accelerate、safetensors、peft、tokenizers、huggingface_hub、sentencepiece、comfy-kitchen、opencv-python | Apache-2.0 |
| pillow | MIT-CMU |
| truststore | MIT |
| scipy、numpy、protobuf、nvidia-ml-py、psutil | BSD |
| 其他間接相依的套件 | MIT、BSD、Apache-2.0、ISC、PSF、MPL-2.0 |

沒有 GPL、AGPL 或禁止商用的套件。

## 五、使用者按了才下載的模型（不隨 CoCo 附帶）

CoCo 會列出下面這些模型的用途、大小與授權，使用者按了才從 Hugging Face 下載：

| 模型 | 用途 | 來源 | 授權 |
|---|---|---|---|
| anime-segmentation（IS-Net） | 去背（動漫角色） | [skytnt/anime-seg](https://huggingface.co/skytnt/anime-seg) | Apache-2.0 |
| BiRefNet lite | 去背（一般） | [ZhengPeng7/BiRefNet_lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) | MIT |
| WD ViT Large Tagger v3 | 打標台（訓練 → 資料集） | [SmilingWolf/wd-vit-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3) | Apache-2.0 |
| SigLIP2 base | 探索（找相似的圖） | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) | Apache-2.0 |
| ControlNet Union ProMax | 參考圖（SDXL） | [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) | Apache-2.0 |
| DWPose（YOLOX＋DWPose） | 參考圖的姿勢前處理 | [yzd-v/DWPose](https://huggingface.co/yzd-v/DWPose) | Apache-2.0 |
| Depth Anything V2 Small | 參考圖的深度前處理 | [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | Apache-2.0 |
| Anima Aesthetic v1.1（anima-aesthetic-v1.1） | 試用模型（設定 → 資源資料夾 → 下載模型） | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md)：模型本身禁止商用，生成的圖不限；另受 [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) 規範 |
| Qwen3 0.6B Base（qwen_3_06b_base） | Anima 的文字編碼器 | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | Apache-2.0 |
| Qwen Image VAE（qwen_image_vae） | Anima、Krea 2 的 VAE | [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) | Apache-2.0 |
| FLUX.1 VAE（ae） | Z-Image、FLUX.1 的 VAE | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) | Apache-2.0 |
| Qwen3 4B（qwen_3_4b） | Z-Image、FLUX.2 klein 4B 的文字編碼器 | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) | Apache-2.0 |
| FLUX.2 VAE（flux2-vae） | FLUX.2 klein 的 VAE | [Comfy-Org/vae-text-encorder-for-flux-klein-4b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b) | Apache-2.0 |
| Qwen3 8B fp8（qwen_3_8b_fp8mixed） | FLUX.2 klein 9B 的文字編碼器 | [Comfy-Org/vae-text-encorder-for-flux-klein-9b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b) | Apache-2.0（Qwen3-8B 的授權；這個庫整體標的 FLUX 非商用授權是給 9B 主模型的，CoCo 只下載文字編碼器） |
| Qwen3-VL 4B fp8（qwen3vl_4b_fp8_scaled） | Krea 2 的文字編碼器 | [Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL) | Apache-2.0 |
| Qwen3-VL 8B int8（qwen3vl_8b_int8_convrot） | Qwen Image 2.1 的文字編碼器 | [Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL) | Apache-2.0 |
| T5-XXL fp16（t5xxl_fp16） | FLUX.1 的文字編碼器 | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) | Apache-2.0 |
| CLIP-L（clip_l） | FLUX.1 的第二個文字編碼器（整句的向量） | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) | MIT（OpenAI CLIP） |
| Qwen Image 2.1 VAE（qwen_image_2.1_vae_bf16） | Qwen Image 2.1 的 VAE | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)：限非商用 |

Anima 由 CircleStone Labs LLC 依 CircleStone Non-Commercial License 授權；Built on NVIDIA Cosmos，Licensed by NVIDIA Corporation under the NVIDIA Open Model License。

---

## 附錄 A：Apache License 2.0

適用於第一節的 diffusers 檔案、Qwen 與 T5 的 tokenizer、Qwen Image VAE 的設定，以及第二節的 IS-Net 結構與 `edit_embeds`。

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## 附錄 B：BiRefNet 的 MIT License

適用於 `core/birefnet.py`。

```
MIT License

Copyright (c) 2024 ZhengPeng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
