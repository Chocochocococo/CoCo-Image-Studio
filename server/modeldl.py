# -*- coding: utf-8 -*-
"""下載模型：給還沒有模型的人試用的模型（Anima Aesthetic），以及拆分式模型缺少的 VAE、文字編碼器。

不會自動下載，使用者在設定的「資源資料夾」按了才下載。來源是官方的 Hugging Face（不用登入），版本固定，
下載完照公布的 SHA256 檢查。下載中斷（網路斷掉、關掉 CoCo）留著已經下載的部分，下次按下載從斷掉的地方接著下載；
取消或內容不對就刪掉（download.py，跟 ControlNet、去背、探索共用）。

放到資源資料夾裡（模型總資料夾裡對應的子資料夾，例如 StabilityMatrix 的 DiffusionModels、VAE、TextEncoders；
沒有就 CoCo 自己的 models 底下的 diffusion_models、vae、text_encoders），資源庫馬上看得到，生成時沒選的零件會自動用（assets.fill_parts）。
跟使用者手上已經測過的檔案逐位元相同（SHA256 一樣），授權見 THIRD_PARTY_NOTICES.md。
"""

from . import assets, download, presets
from .resources import res_dirs

from i18n import tr

_HF = "https://huggingface.co"
_ANIMA = ("circlestone-labs/Anima", "f973fc41ec7545364ac9776c2440285f43ff2a30")
_QIMAGE = ("Comfy-Org/Qwen-Image_ComfyUI", "1f12b17be14c89b026c51a91d67c32f84bb047bc")
_ZIMAGE = ("Comfy-Org/z_image_turbo", "6fc90a3b1b653e935a0d175e260736de25b84df5")
_QWEN3VL = ("Comfy-Org/Qwen3-VL", "02f0d3eefb59528799653d16735818801177ef1f")
_QIMAGE21 = ("Comfy-Org/Qwen-Image-2.1", "cb504a4090723e43f17ad01cec0359490e2de613")
# FLUX.2 klein 的零件（ComfyUI 官方重新包裝的）。4B 那個庫是 Apache 2.0；9B 那個庫整個標成 FLUX 非商用授權，
# 但這裡只拿它的文字編碼器（Qwen3-8B，Qwen 官方是 Apache 2.0），不拿 9B 的主模型
_KLEIN4 = ("Comfy-Org/vae-text-encorder-for-flux-klein-4b", "5f526678002e43af5551dadb73ce2e8c91b43afe")
_KLEIN9 = ("Comfy-Org/vae-text-encorder-for-flux-klein-9b", "3f62d9d8ae1fec33c6e91453d5c712855b096b55")
# FLUX.1 的文字編碼器（ComfyUI 官方的庫，Apache 2.0；CLIP-L 本身是 OpenAI 的 MIT）
_FLUXTE = ("comfyanonymous/flux_text_encoders", "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5")
_ANIMA_LICENSE = ("CircleStone Labs Non-Commercial License", f"{_HF}/circlestone-labs/Anima/blob/main/LICENSE.md")
_APACHE = ("Apache-2.0", "https://www.apache.org/licenses/LICENSE-2.0")
_MIT = ("MIT", "https://github.com/openai/CLIP/blob/main/LICENSE")
_QWEN_RESEARCH = ("Qwen Research License", f"{_HF}/Qwen/Qwen-Image-2.1/blob/main/LICENSE")


def _file(kind, repo, path, size, sha256, license_):
    name = path.rsplit("/", 1)[-1]
    return {"kind": kind, "name": name, "url": f"{_HF}/{repo[0]}/resolve/{repo[1]}/{path}", "page": f"{_HF}/{repo[0]}",
            "bytes": size, "sha256": sha256, "license": license_[0], "license_url": license_[1]}


# 可以下載的檔案（鍵＝檔名主體）
FILES = {
    "anima-aesthetic-v1.1": _file("model", _ANIMA, "split_files/diffusion_models/anima-aesthetic-v1.1.safetensors", 4182230656,
                                  "3c1868387a3a1ff504bbb87c33678321965ead381fcf87afbd0264daa600c082", _ANIMA_LICENSE),
    "qwen_3_06b_base": _file("te", _ANIMA, "split_files/text_encoders/qwen_3_06b_base.safetensors", 1192135096,
                             "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba", _APACHE),
    "qwen_image_vae": _file("vae", _QIMAGE, "split_files/vae/qwen_image_vae.safetensors", 253806246,
                            "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f", _APACHE),
    "ae": _file("vae", _ZIMAGE, "split_files/vae/ae.safetensors", 335304388,
                "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38", _APACHE),
    "qwen_3_4b": _file("te", _ZIMAGE, "split_files/text_encoders/qwen_3_4b.safetensors", 8044982048,
                       "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a", _APACHE),
    "qwen3vl_4b_fp8_scaled": _file("te", _QWEN3VL, "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 5242467968,
                                   "54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094", _APACHE),
    "qwen_image_2.1_vae_bf16": _file("vae", _QIMAGE21, "vae/qwen_image_2.1_vae_bf16.safetensors", 675509688,
                                     "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9", _QWEN_RESEARCH),
    "qwen3vl_8b_int8_convrot": _file("te", _QWEN3VL, "text_encoders/qwen3vl_8b_int8_convrot.safetensors", 9350798360,
                                     "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f", _APACHE),
    # BFL 原始命名的那份（core/flux2.convert_vae 轉；2026-10-09 核對過跟 diffusers 命名的那份權重逐位元相同）
    "flux2-vae": _file("vae", _KLEIN4, "split_files/vae/flux2-vae.safetensors", 336211292,
                       "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3", _APACHE),
    "qwen_3_8b_fp8mixed": _file("te", _KLEIN9, "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors", 8664848742,
                                "abad16806e0cbabc54e0325d6565847443fe396d5f0be38bb3cd3fe75a1201d6", _APACHE),
    "t5xxl_fp16": _file("te", _FLUXTE, "t5xxl_fp16.safetensors", 9787841024,
                        "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635", _APACHE),
    "clip_l": _file("te", _FLUXTE, "clip_l.safetensors", 246144152,
                    "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd", _MIT),
}
# 試用模型：主模型與它的零件（零件已經有相符的就不下載）
TRIAL = {"model": "anima-aesthetic-v1.1", "arch": "anima"}
# 拆分式模型的零件 → 下載哪個檔（官方工作流程用的）
PARTS = {("anima", "vae"): "qwen_image_vae", ("anima", "te"): "qwen_3_06b_base",
         ("zimage", "vae"): "ae", ("zimage", "te"): "qwen_3_4b",
         ("krea2", "vae"): "qwen_image_vae", ("krea2", "te"): "qwen3vl_4b_fp8_scaled",
         ("qwen21", "vae"): "qwen_image_2.1_vae_bf16", ("qwen21", "te"): "qwen3vl_8b_int8_convrot",
         ("klein4", "vae"): "flux2-vae", ("klein4", "te"): "qwen_3_4b",
         ("klein9", "vae"): "flux2-vae", ("klein9", "te"): "qwen_3_8b_fp8mixed",
         **{(a, k): v for a in assets.FLUX1_ARCHS for k, v in (("vae", "ae"), ("te", "t5xxl_fp16"), ("clip", "clip_l"))}}
# 零件跟另一個架構完全一樣的：下載清單只列一次，標題改成這個（FLUX.1 dev、schnell、Kontext 用同一套零件）
SAME_PARTS = {"flux1s": "flux1", "kontext": "flux1"}
GROUP_TITLE = {"flux1": "FLUX.1（dev、schnell、Kontext）"}

# 下載到哪裡：資源資料夾裡名字是這些的優先（模型的 Diffusers、StableDiffusion 是別種格式的）
PREFER = {"model": ("diffusionmodels", "diffusion_models", "unet"), "vae": ("vae",), "te": ("textencoders", "text_encoders", "clip")}
# 使用者的資料夾都沒有時用 CoCo 自己的（server/modelhome.py：拆分式主模型放 diffusion_models）
OWN = {"model": "diffusion_models", "vae": "vae", "te": "text_encoders"}

_DL = download.Jobs()            # 工作（"trial" 或檔案的鍵）→ 進度


def dest_dir(kind):
    """這一類下載到哪個資料夾：使用者的資源資料夾裡名字對的優先，再來第一個，都沒有就 CoCo 自己的"""
    from . import modelhome
    own = {str(d).lower() for d in modelhome.own_dirs(kind)}
    dirs = [d for d in res_dirs(kind) if d.is_dir() and str(d).lower() not in own]
    for d in dirs:
        if d.name.lower() in PREFER[kind]:
            return d
    return dirs[0] if dirs else modelhome.home() / OWN[kind]


def _dest(key):
    f = FILES[key]
    return dest_dir(f["kind"]) / f["name"]


def _have(key):
    """下載過了（放在下載的地方，大小對）"""
    p = _dest(key)
    return p.is_file() and p.stat().st_size == FILES[key]["bytes"]


def trial_keys():
    """試用模型要下載哪些：主模型，加上還沒有相符的零件"""
    keys = [TRIAL["model"]]
    for kind in ("vae", "te"):
        if not assets.part_candidates(kind, TRIAL["arch"]):
            keys.append(PARTS[(TRIAL["arch"], kind)])
    return [k for k in keys if not _have(k)]


def _info(key):
    """part：上次下載到一半留下的大小（關掉 CoCo 再打開，下載的進度沒了，靠這個說上次下載到哪裡）"""
    f = FILES[key]
    have = _have(key)
    return {"key": key, "name": f["name"], "kind": f["kind"], "bytes": f["bytes"], "page": f["page"],
            "license": f["license"], "license_url": f["license_url"], "have": have,
            "part": 0 if have else download.partial(_dest(key), f["bytes"])}


def status():
    keys = trial_keys()
    trial = {"model": _info(TRIAL["model"]), "files": [_info(k) for k in keys], "bytes": sum(FILES[k]["bytes"] for k in keys),
             "download": _DL.progress("trial")}
    parts = []
    for (arch, kind), key in PARTS.items():
        if arch in SAME_PARTS:
            continue
        found = assets.part_candidates(kind, arch)
        parts.append({"arch": arch, "title": GROUP_TITLE.get(arch) or assets.ARCH_TITLE[arch], "kind": kind,
                      "label": assets.part_label(kind), "found": [f["name"] for f in found], "file": _info(key),
                      "download": _DL.progress(key)})
    return {"trial": trial, "parts": parts, "dirs": {k: str(dest_dir(k)) for k in OWN}}


def _saved(f):
    """一個檔下載好了：試用模型順便設好它自己的品質標籤（官方建議 Aesthetic 不用 score 標籤）。
    放的地方一定在資源清單裡（使用者的資料夾或 CoCo 自己的，見 dest_dir），不用再登記"""
    if f["key"] == TRIAL["model"]:
        presets.set_default(str(f["dest"]), presets.AESTHETIC)


def _run(job, keys, opener=None):
    files = ({**FILES[k], "key": k, "dest": _dest(k)} for k in keys)
    download.run(_DL[job], files, opener, hint=tr("srv.dl.manualHintPage"), on_saved=_saved)


def start(job):
    """job＝"trial"（試用模型）或 FILES 的鍵（一個零件）"""
    keys = trial_keys() if job == "trial" else [job] if job in FILES else None
    if keys is None:
        return {"error": tr("srv.dl.unknownJob")}
    if not keys or all(_have(k) for k in keys):
        return {"ok": True, "already": True}
    return _DL.begin(job, sum(FILES[k]["bytes"] for k in keys), _run, (job, keys))


def cancel(job):
    return _DL.cancel(job)
