# -*- coding: utf-8 -*-
"""
samplers.py —— 自製取樣器

diffusers 沒有、但實際需要的取樣器放這裡。

**為什麼可以自己寫。** 規格 §3 原本把「自己寫取樣器 / UNet / VAE 載入」
列在同一條排除項目，理由是「每出一個新架構就要重寫」。那個理由對後兩者
成立，對取樣器不成立——取樣器只跟 sigma 與模型輸出打交道，不碰架構，
寫一次 SDXL、Anima、Krea 2 全部通用。所以取樣器屬於分層原則的第 3 層
（取樣迴圈，「用 diffusers，必要時微調」），不是第 2 層那條紅線。

**為什麼不直接移植。** ComfyUI 是 GPL-3.0、Forge 與 Neo Forge 是
AGPL-3.0，都是強制傳染的 copyleft，把它們的程式碼抄進來，整個專案就得
改用同樣的授權。這裡一律照論文的演算法自己實作——演算法本身不受著作權
保護，受保護的是特定的程式碼寫法。
"""

import numpy as np
import torch
from diffusers import (DEISMultistepScheduler, DPMSolverMultistepScheduler,
                       DPMSolverSinglestepScheduler,
                       FlowMatchEulerDiscreteScheduler, SASolverScheduler)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput)
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.utils.torch_utils import randn_tensor


def _er_phi(lam):
    """ER-SDE 的雜訊尺度函式 φ(λ) = λ·(e^(λ^0.3) + 10)

    論文把 φ 留成自由函式，不同的選擇會得到不同的解算器；
    這一個是參考實作採用的預設值。
    """
    return lam * ((lam ** 0.3).exp() + 10.0)


def _er_phi_identity(lam):
    """φ(λ) = λ —— ER-SDE 的確定性極限（ER-ODE）

    此時 r = φ(λ_t)/φ(λ_s) = λ_t/λ_s，代進變異數
    `λ_t² − (λ_s·r)²` 恰好得到 0，雜訊項自然消失，
    **而且收縮係數也一併換成確定性情況下對的那個**。

    **這才是關掉 ER-SDE 隨機性的正確方法，不是把 s_noise 調成 0。**
    r 是假設全額雜訊會被補回來才推導出來的（大步時 r 可以小到 0.08），
    只砍雜訊會留下沒有補償的收縮，高頻結構被壓掉，出圖是一片純色。
    實測（不完美的去噪器、20 步）：高頻結構相對真值

        s_noise 1.0 → 21.9%   0.5 → 11.2%   0.0 → **2.9%**

    注意這個崩潰用完美速度預測的測試**看不出來**（那種情況下 x₀ 就是
    答案，怎麼收縮都會落在對的點上），要用有誤差的模型才驗得到。
    """
    return lam


# ---------------------------------------------------------------- 自製取樣器的共用零件
#
# _FlowStepper：diffusers 那邊的手續（步數計數、型別、輸出格式）放在外殼，
# 子類別只寫「給 x 與速度 v，算出下一步」。ER-SDE 兩種寫法（flow 空間、VE 空間）共用積分與 λ 的換算。

class _FlowStepper(FlowMatchEulerDiscreteScheduler):
    """自製 flow 取樣器的外殼：輸入轉 float32、呼叫 `_advance`、轉回原本的型別、步數加一。
    每個 token 各自 timestep 的用法（per_token_timesteps）交回 diffusers 原本的 Euler"""

    def _advance(self, x, v, i, generator):
        raise NotImplementedError

    def step(self, model_output, timestep, sample, generator=None,
             return_dict=True, per_token_timesteps=None, **kw):
        if per_token_timesteps is not None:
            return super().step(model_output, timestep, sample, generator=generator,
                                return_dict=return_dict, per_token_timesteps=per_token_timesteps, **kw)
        if self.step_index is None:
            self._init_step_index(timestep)
        nxt = self._advance(sample.to(torch.float32), model_output.to(torch.float32),
                            self.step_index, generator)
        nxt = nxt.to(model_output.dtype)
        self._step_index += 1
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=nxt) if return_dict else (nxt,)


def _fresh_noise(like, generator):
    return randn_tensor(like.shape, generator=generator, device=like.device, dtype=torch.float32)


_ER_GRID = 200        # 二、三階修正的積分切成幾格
_SIGMA_CAP = 1.0 - 1e-4


def _ve_lambda(sigma):
    """flow 的 σ → VE 的 λ = σ/(1−σ)。σ=1 時無限大，先夾到 1−1e-4"""
    capped = torch.clamp(sigma, max=_SIGMA_CAP)
    return capped / (1.0 - capped)


def _er_integrals(lam_now, lam_next, device, phi_fn=_er_phi):
    """二、三階修正要的積分（左端點和，_ER_GRID 格；參考實作用 200，實測降到 50 出圖就會開始有差），
    積分區間是 [λ_next, λ_now]：
        I₁ = ∫ 1/φ(λ) dλ        I₂ = ∫ (λ − λ_now)/φ(λ) dλ
    回傳 (Δλ = λ_next − λ_now, I₁, I₂)"""
    span = lam_next - lam_now
    width = -span / _ER_GRID
    nodes = lam_next + torch.arange(_ER_GRID, device=device, dtype=torch.float32) * width
    phi = phi_fn(nodes)
    return span, (1.0 / phi).sum() * width, ((nodes - lam_now) / phi).sum() * width


class FlowMatchERSDEScheduler(_FlowStepper):
    """ER-SDE-Solver-3，flow matching 版（Anima、Krea 2 這類模型）

    Extended Reverse-Time SDE solver，論文 arXiv:2309.06169。
    照論文的演算法在 flow 空間自己推導、自己寫（2026-10-03 改寫成現在的結構，出圖逐位元不變），
    沒有沿用任何 GPL/AGPL 專案的程式碼。

    在資料預測（x̂₀ = x − σ·v）空間工作。λ = σ/α、α = 1−σ，一步（λ_s → λ_t）：

        r = φ(λ_t)/φ(λ_s)
        一階  x ← (α_t/α_s)·r·x + α_t·(1−r)·x̂₀
        二階  再加 α_t·(Δλ + I₁·φ(λ_t))·D₁           D₁ = x̂₀ 對 λ 的差分
        三階  再加 α_t·(Δλ²/2 + I₂·φ(λ_t))·D₂        D₂ = D₁ 的差分／半個區間
        雜訊  再加 α_t·ε·s_noise·√(λ_t² − (λ_s·r)²)

    I₁、I₂ 見 `_er_integrals`（沒有解析解，數值積分）。二、三階要前幾步的歷史，所以是多步解算器，
    前兩步自動降階。最後一步（σ_next = 0）直接回 x̂₀。
    """

    def __init__(self, *a, max_stage: int = 3, s_noise: float = 1.0,
                 deterministic: bool = False, **kw):
        super().__init__(*a, **kw)
        self.max_stage = int(max_stage)
        # deterministic 走 φ(λ)=λ，雜訊項會自己歸零（見 _er_phi_identity）。
        # 不要改用 s_noise=0 達成，那會壓掉高頻結構。
        self.deterministic = bool(deterministic)
        self.s_noise = 0.0 if deterministic else float(s_noise)
        self._phi = _er_phi_identity if deterministic else _er_phi
        self._past = []         # 走過的每一步：(λ_s, x̂₀, D₁ 或 None)

    # 參數要照抄父類別，不能用 *a/**kw。呼叫端（Anima 的 retrieve_timesteps）
    # 是用 inspect.signature 檢查有沒有 sigmas 這個參數的，
    # 簽名一糊掉就會被判定成「不支援自訂 sigmas」而拒絕。
    def set_timesteps(self, num_inference_steps=None, device=None,
                      sigmas=None, mu=None, timesteps=None):
        out = super().set_timesteps(num_inference_steps=num_inference_steps, device=device,
                                    sigmas=sigmas, mu=mu, timesteps=timesteps)
        self._past = []         # 換排程就把多步歷史清掉
        return out

    def _advance(self, x, v, i, generator):
        sigma, sigma_next = self.sigmas[i], self.sigmas[i + 1]
        x0 = x - sigma * v
        if sigma_next <= 0:
            return x0

        lam_s, lam_t = _ve_lambda(sigma), _ve_lambda(sigma_next)
        alpha_s = 1.0 - torch.clamp(sigma, max=_SIGMA_CAP)
        alpha_t = 1.0 - sigma_next
        phi_t = self._phi(lam_t)
        r = phi_t / self._phi(lam_s)

        out = (alpha_t / alpha_s) * r * x + alpha_t * (1.0 - r) * x0

        order = min(self.max_stage, len(self._past) + 1)
        d1 = None
        if order >= 2:
            lam_p, x0_p, d1_p = self._past[-1]
            span, i1, i2 = _er_integrals(lam_s, lam_t, x.device, self._phi)
            d1 = (x0 - x0_p) / (lam_s - lam_p)
            out = out + alpha_t * (span + i1 * phi_t) * d1
            if order >= 3 and d1_p is not None:
                d2 = (d1 - d1_p) / ((lam_s - self._past[-2][0]) / 2)
                out = out + alpha_t * (span * span / 2 + i2 * phi_t) * d2

        # 隨機項：r 已經把確定性的收縮算進去了，這裡補差額的雜訊。
        # 變異數 λ_t² − (λ_s·r)² **不能照字面平方相減**：λ 在 σ 接近 1 時可以到數百，
        # 兩個相近大數的平方相減會災難性抵消。實測用 φ(λ)=λ 時（變異數恰好是 0），
        # float32 殘留約 1e-4，開根號放大成 1e-2 再乘雜訊，出圖多一層看不出來源的擾動。
        # 寫成平方差，兩個因子都是中等大小的數，穩定得多。
        if self.s_noise > 0:
            shrunk = lam_s * r
            spread = ((lam_t - shrunk) * (lam_t + shrunk)).clamp(min=0.0).sqrt()
            out = out + alpha_t * _fresh_noise(x, generator) * self.s_noise * spread

        self._past.append((lam_s, x0, d1))      # 階數只升不降：D₁ 是 None 只有第一步或一律一階
        return out


# ---------------------------------------------------------------- R32
class FlowMatchEulerAncestralScheduler(FlowMatchEulerDiscreteScheduler):
    """可調 eta 與注入範圍的 ancestral Euler，flow matching 版

    **為什麼要自己寫。** diffusers 的 `stochastic_sampling` 是寫死的
    eta=1：每一步都把原本軌跡的雜訊**整個丟掉**重抽

        x' = (1−σ_next)·x₀ + σ_next·ε          （ε 全新）

    配上 Anima 的排程（shift 3.0、35 步時尾巴是 0.2195 → 0.1538 →
    0.0811 → 0），等於在倒數第三、第二步還把圖重新隨機化到 15% 與 8%
    的雜訊水準，然後只剩一步收尾。手指這種高頻小結構好不容易收斂，
    在最後關頭被打散就再也長不回來——這是 SDE 系列在低步數下手容易崩
    的機制。diffusers 沒有給 eta，也沒有 Karras 那套注入範圍控制。

    **做法。** 從 x₀ 反推當前軌跡的雜訊 ε̂，跟新雜訊按 eta 混合：

        ε̂  = (x − (1−σ)·x₀) / σ  =  v + x₀      （下面用右邊那式）
        x' = (1−σ_next)·x₀ + σ_next·( √(1−η²)·ε̂ + η·ε_new )

    `√(1−η²)² + η² = 1`，所以變異數守恆（前提是模型的 ε̂ 跟 ε_new 同尺度，
    這是所有 ancestral 取樣器共同的假設）。兩端接得上既有的取樣器：

        η=0  →  化簡成 x₀ + σ_next·v，**就是確定性 Euler**
        η=1  →  化簡成 diffusers 現在的 stochastic_sampling

    兩個端點都有數值測試釘住（見 tests 的 TestFlowAncestralEta）。

    ε̂ 用 `v + x₀` 算而不是照定義除以 σ，兩者代數上相同
    （x = x₀ + σ·v ⟹ x − (1−σ)·x₀ = σ·(v + x₀)），但右邊不用除法，
    σ 很小的時候穩定得多。

    **late_off** 讓最後一段比例的步驟轉成確定性，是為手部這類細結構準備的
    ——它們最後幾步才定型，不被打散就比較不會崩。

    這裡用**步數比例**而不是 sigma 門檻，是踩過坑之後改的。原本用固定的
    `s_tmin=0.15`，但它能關掉幾步完全取決於步數（shift 3.0 時最小的非零
    sigma 是 3/(steps+2)）：

        步數 ≤16  一步都關不到，整個參數是 no-op
        步數 20–35 只關到 1 步
        步數 50   才關到 2 步

    也就是同一個設定在不同步數下的意義天差地遠，低步數時根本沒作用。
    改成比例之後，`late_off=0.25` 在任何步數都是「最後四分之一走確定性」。

    **s_tmin / s_tmax** 是 Karras et al. 2022 那套 sigma 範圍限制，保留
    當低階旋鈕（門檻比對的是**落點** σ_next，也就是雜訊實際被加到的層級），
    但預設關閉，具名的取樣器一律走 late_off。

    **step_aware** 換用 k-diffusion 系（ComfyUI / A1111 / Forge 共用的
    那套）的注入量。它們的 ancestral 是「先確定性走到更低的 σ_down，
    再把雜訊加回來」，注入量是

        σ_up = η · σ_t · √(1 − (σ_t/σ_s)²)

    寫成上面那條混合式，就等於把 eta 換成

        η_eff = η · √(1 − (σ_t/σ_s)²)

    那個根號是**步長因子**：步數越多、相鄰 sigma 越接近，注入就越少。
    diffusers 的 `stochastic_sampling` 沒有這一項，注入量是固定的 η·σ_t，
    **完全不看步長**——這正是它步數越多越糊、而 k-diffusion 系不會的原因。

    這是照公開的演算法自己推的（見檔頭關於 copyleft 的說明），
    沒有沿用任何 GPL/AGPL 專案的程式碼。
    """

    def __init__(self, *a, eta: float = 1.0, late_off: float = 0.0,
                 step_aware: bool = False, s_tmin: float = 0.0,
                 s_tmax: float = float("inf"), s_noise: float = 1.0, **kw):
        super().__init__(*a, **kw)
        self.eta = float(eta)
        self.late_off = float(late_off)
        self.step_aware = bool(step_aware)
        self.s_tmin = float(s_tmin)
        self.s_tmax = float(s_tmax)
        self.s_noise = float(s_noise)

    def _eta_for(self, sigma, sigma_next):
        """這一步實際用的 eta。step_aware 會再乘上步長因子。"""
        eta = min(self.eta, 1.0)
        if not self.step_aware:
            return eta
        if sigma <= 0.0:
            return eta
        ratio = min(sigma_next / sigma, 1.0)
        return eta * (1.0 - ratio * ratio) ** 0.5

    def _is_quiet(self, i, sigma_next):
        """這一步該不該安靜（走確定性）"""
        if sigma_next <= 0.0 or self.eta <= 0.0:
            return True
        if not (self.s_tmin <= sigma_next <= self.s_tmax):
            return True
        n = self.num_inference_steps
        if self.late_off > 0.0 and n:
            return i >= int(round(n * (1.0 - self.late_off)))
        return False

    def step(self, model_output, timestep, sample, generator=None,
             return_dict=True, per_token_timesteps=None, **kw):
        # per_token 是給逐 token sigma 的模型用的，Anima 的迴圈不會傳。
        # 真的傳了就交回父類別，不要在這裡半套實作。
        if per_token_timesteps is not None:
            return super().step(model_output, timestep, sample,
                                generator=generator, return_dict=return_dict,
                                per_token_timesteps=per_token_timesteps, **kw)

        if self.step_index is None:
            self._init_step_index(timestep)

        i = self.step_index
        out_dtype = model_output.dtype
        x = sample.to(torch.float32)
        v = model_output.to(torch.float32)
        sigma = self.sigmas[i].to(device=x.device, dtype=torch.float32)
        sigma_next = self.sigmas[i + 1].to(device=x.device,
                                           dtype=torch.float32)

        x0 = x - sigma * v

        if self._is_quiet(i, float(sigma_next)):
            # η=0 的化簡式。等同父類別的 sample + dt·v，這裡照 x₀ 的形式
            # 寫，是為了跟下面那條共用同一個 x₀，讀起來看得出是同一族。
            prev = x0 + sigma_next * v
        else:
            eta = self._eta_for(float(sigma), float(sigma_next))
            eps_hat = v + x0
            noise = randn_tensor(x.shape, generator=generator,
                                 device=x.device, dtype=torch.float32)
            mixed = ((1.0 - eta * eta) ** 0.5) * eps_hat \
                + eta * self.s_noise * noise
            prev = (1.0 - sigma_next) * x0 + sigma_next * mixed

        prev = prev.to(out_dtype)
        self._step_index += 1
        if not return_dict:
            return (prev,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev)


# ---------------------------------------------------------------- R30
class _AcceptsSigmas:
    """替不吃 sigmas 的 diffusers 排程器補上這個參數（R30）

    **問題。** DPM++ / DEIS / SA-Solver 這幾支 diffusers 都支援 flow
    （`use_flow_sigmas=True` + `prediction_type="flow_prediction"`），
    解算器本身對 Anima 完全適用；卡住的只有排程的輸入介面：它們的
    `set_timesteps` 只收 `num_inference_steps` 或 `timesteps`，而 Anima 的
    `before_denoise` 一律用 `sigmas=` 呼叫，`retrieve_timesteps` 又會先
    `inspect.signature` 檢查有沒有這個參數，沒有就直接丟例外。

    **為什麼不能改用 timesteps 繞過。** 看起來 `timesteps` 是條路，其實
    不是：`use_flow_sigmas=True` 時，那幾支的 `set_timesteps` **會完全
    忽略傳進去的 timesteps**，改用 `num_inference_steps` 自己重算
    （而 timesteps 一給，num_inference_steps 就必須是 None，於是重算那行
    直接爆）。而且 timesteps 會被轉成 int64，flow 的 timestep 是
    σ×1000 的浮點數，轉完就掉精度。所以只能從 sigma 這一側接。

    **做法。** 先讓父類別照它自己的排程跑一次完整的 `set_timesteps`，
    然後只把 `sigmas` 與 `timesteps` 換成外面給的。第一次呼叫的那組 sigma
    是丟掉不用的——真正的目的是讓 `model_outputs`、`order_list`、
    `lower_order_nums`、`sample`、`step_index` 這些**每支解算器各自不同**
    的狀態，由它自己的程式碼初始化。自己逐欄位重設的話，哪天 diffusers
    某支多一個欄位就會漏掉，而且漏掉不會報錯，只會靜靜出錯圖。

    shift 用的是 `flow_shift·σ/(1+(flow_shift−1)·σ)`，跟
    `FlowMatchEulerDiscreteScheduler.set_timesteps` 第 2 步逐字相同，
    所以同一個 shift 下，這幾支跟 Euler / UniPC 的 sigma 排程完全對齊。
    """

    def set_timesteps(self, num_inference_steps=None, device=None,
                      sigmas=None, mu=None, timesteps=None):
        if sigmas is None:
            return super().set_timesteps(
                num_inference_steps=num_inference_steps, device=device)

        sigmas = np.asarray(sigmas, dtype=np.float64)
        n = len(sigmas)
        super().set_timesteps(num_inference_steps=n, device=device)

        sh = float(getattr(self.config, "flow_shift", 1.0))
        s = sh * sigmas / (1.0 + (sh - 1.0) * sigmas)

        # σ=1 會讓 α=1−σ=0，接著 log(α)=−inf 把整條 lambda 弄壞。
        # UniPC 內建就是把首項往下挪一點點，這裡照做（見它的 eps 註解）。
        if abs(s[0] - 1.0) < 1e-6:
            s[0] -= 1e-6

        train = int(self.config.num_train_timesteps)
        self.timesteps = torch.from_numpy(s * train).to(
            device=device, dtype=torch.int64)
        # 尾巴補 0：final_sigmas_type="zero"，最後一步直接落到 x₀。
        full = np.concatenate([s, [0.0]]).astype(np.float32)
        # 父類別一律把 sigmas 留在 CPU（自己的註解：少一點 CPU/GPU 往返），
        # 更新用的那幾支函式會自己 .to(device)，所以這裡跟著留 CPU。
        self.sigmas = torch.from_numpy(full)
        self.num_inference_steps = n

    def step(self, model_output, timestep, sample, generator=None,
             return_dict=True, **kw):
        """最後一步（σ_next=0）自己算，其餘照父類別。

        **為什麼要接管最後一步。** DEIS 與 SA-Solver 的一階更新是

            x_t = (α_t/α_s)·x − σ_t·(e^h − 1)·x₀,   h = λ_t − λ_s

        σ_t=0 時 λ_t=+∞、e^h=∞，於是 `σ_t·(e^h−1)` 變成 `0×∞ = NaN`。
        數學上那個極限是有限的（= α_t·σ_s/α_s），只是浮點算不出來。
        diffusers 自己不會踩到，是因為它們的 flow 排程**不以 0 收尾**
        （沒有 final_sigmas_type，最後一格是重複前一個 σ）——但那樣圖會
        停在 σ≈0.21 沒去噪完。Anima 要的是收到 x₀，所以洞得自己補。

        補法用 flow matching 的解析定義 x₀ = x − σ·v，不是近似。
        dpmsolver++ 在 σ_t=0 時化簡的結果也正是這個
        （(σ_t/σ_s)·x → 0，−α_t·(e^{−h}−1)·x₀ → x₀），所以對 DPM++ 這條
        分支不改變任何東西，只是把 DEIS / SA-Solver 算不出來的地方補上。
        `lower_order_final=True` 本來就表示最後一步要降到一階，這也一致。
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        if float(self.sigmas[self.step_index + 1]) != 0.0:
            if self._takes_generator:
                return super().step(model_output, timestep, sample,
                                    generator=generator,
                                    return_dict=return_dict, **kw)
            return super().step(model_output, timestep, sample,
                                return_dict=return_dict, **kw)

        sigma = self.sigmas[self.step_index].to(sample.device)
        prev = (sample.to(torch.float32)
                - sigma * model_output.to(torch.float32)).to(sample.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev,)
        return SchedulerOutput(prev_sample=prev)

    @property
    def _takes_generator(self):
        # DEIS 是確定性的，step() 沒有 generator，硬傳會 TypeError
        import inspect
        return "generator" in inspect.signature(super().step).parameters


class FlowDPMSolverMultistepScheduler(_AcceptsSigmas,
                                      DPMSolverMultistepScheduler):
    """DPM++ 2M / 2M SDE / 3M SDE 的 flow 版（階數與 SDE 由 config 決定）"""


class FlowDPMSolverSinglestepScheduler(_AcceptsSigmas,
                                       DPMSolverSinglestepScheduler):
    """DPM++ SDE 的 flow 版"""


class FlowDEISMultistepScheduler(_AcceptsSigmas, DEISMultistepScheduler):
    """DEIS 的 flow 版"""


class FlowSASolverScheduler(_AcceptsSigmas, SASolverScheduler):
    """SA-Solver 的 flow 版"""


# ---------------------------------------------------------------- R34 對照組
#
# 下面兩支是「同一支程式、同一個 seed」底下的 A/B 對照組：Euler a RF 與 ER-SDE ve。
# 跨程式比會混進提示詞編碼、VAE、CFG 實作等變因，回答不了「是不是解算器造成的」。
# 2026-10-03 只照公式（各自的 docstring）從頭重寫，沒有沿用任何 GPL/AGPL 專案的程式碼（見檔頭）；
# 每一步的浮點運算跟重寫前一樣，舊圖照樣重現得了（tests/test_reference_samplers.py）。
#
# ER-SDE ve 刻意在 VE 空間算，好跟 flow 空間的 FlowMatchERSDEScheduler 互相檢驗。兩個空間的橋：
# flow 是 x = (1−σ)x₀ + σε，兩邊同除 α = 1−σ 得到 x/α = x₀ + λ·ε、λ = σ/α，
# 也就是 x/α 是 VE 形式、它的 sigma 是 λ。換過去、走一步、乘 α_t 換回來，應該跟直接在 flow 空間走等價。


class ERSDEVEScheduler(_FlowStepper):
    """ER-SDE，在 VE 空間算的版本（R34 對照組）

    跟 `FlowMatchERSDEScheduler` 同一篇論文、同一個演算法，差在**在哪個空間算**：那支直接在 flow 空間
    展開（每一項自己乘 α_t），這支先除 α 換到 VE 空間（sigma 就是 λ）、走完再乘 α_t 換回來。
    兩份出圖對得上就互相驗證了推導（`test_ve_and_flow_forms_agree` 釘住）。

    一步（λ_s → λ_t，x̂₀ = x − σ·v，y = x/α_s）：

        r  = φ(λ_t)/φ(λ_s)
        y ← r·y + (1−r)·x̂₀                                   一階
        y ← y + (Δλ + I₁·φ(λ_t))·D₁                           二階，D₁ = x̂₀ 對 λ 的差分
        y ← y + (Δλ²/2 + I₂·φ(λ_t))·D₂                        三階，D₂ = D₁ 的差分／半個區間
        y ← y + ε·s_noise·√(λ_t² − (λ_s·r)²)                  補雜訊
        x_next = α_t·y

    幾階由 max_stage 與累積的步數決定（第一步只能一階）。最後一步（σ_next = 0）直接回 x̂₀。
    """

    def __init__(self, *a, max_stage: int = 3, s_noise: float = 1.0, **kw):
        super().__init__(*a, **kw)
        self.max_stage = int(max_stage)
        self.s_noise = float(s_noise)
        self._past = []         # 走過的每一步：(λ_s, x̂₀, D₁ 或 None)

    def set_timesteps(self, num_inference_steps=None, device=None,
                      sigmas=None, mu=None, timesteps=None):
        out = super().set_timesteps(num_inference_steps=num_inference_steps, device=device,
                                    sigmas=sigmas, mu=mu, timesteps=timesteps)
        self._past = []
        return out

    def _advance(self, x, v, i, generator):
        sigma, sigma_next = self.sigmas[i], self.sigmas[i + 1]
        x0 = x - sigma * v
        if sigma_next <= 0:
            return x0

        lam_s, lam_t = _ve_lambda(sigma), _ve_lambda(sigma_next)
        phi_t = _er_phi(lam_t)
        r = phi_t / _er_phi(lam_s)

        y = x / (1.0 - torch.clamp(sigma, max=_SIGMA_CAP))
        y = r * y + (1.0 - r) * x0

        order = min(self.max_stage, len(self._past) + 1)
        d1 = None
        if order >= 2:
            lam_p, x0_p, d1_p = self._past[-1]
            span, i1, i2 = _er_integrals(lam_s, lam_t, x.device)
            d1 = (x0 - x0_p) / (lam_s - lam_p)
            y = y + (span + i1 * phi_t) * d1
            if order >= 3 and d1_p is not None:
                d2 = (d1 - d1_p) / ((lam_s - self._past[-2][0]) / 2)
                y = y + (span * span / 2 + i2 * phi_t) * d2

        if self.s_noise > 0:
            # λ_t² − (λ_s·r)² 寫成平方差，避免兩個相近的數相減（理由見 FlowMatchERSDEScheduler）
            shrunk = lam_s * r
            spread = ((lam_t - shrunk) * (lam_t + shrunk)).clamp(min=0.0).sqrt()
            y = y + _fresh_noise(x, generator) * self.s_noise * spread

        self._past.append((lam_s, x0, d1))      # 階數只升不降：D₁ 是 None 只有第一步或一律一階
        return (1.0 - sigma_next) * y


# ---- Euler a RF

def _rf_plan(s, t, eta):
    """Euler a RF 一步的係數（σ_s → σ_t，η 是加噪比例）：

        σ_down = σ_t·(1 + (σ_t/σ_s − 1)·η)                    先走到比 σ_t 更低的地方
        reach  = σ_down/σ_s                                   x ← reach·x + (1−reach)·x̂₀
        lift   = α_t/α_down                                   再拉回 σ_t 的水準（α = 1−σ）
        spread = √(σ_t² − σ_down²·α_t²/α_down²)               補回的雜訊量

    η=0 時 σ_down = σ_t，第一式就是 flow 的 Euler 步，不拉回、不加噪（lift、spread 回 None）"""
    s_down = t * (1.0 + (t / s - 1.0) * eta)
    reach = s_down / s
    if eta <= 0:
        return reach, None, None
    a_t, a_down = 1.0 - t, 1.0 - s_down
    gap = t * t - s_down * s_down * (a_t * a_t) / (a_down * a_down)
    return reach, a_t / a_down, max(gap, 0.0) ** 0.5     # gap 只會因浮點誤差變負


class FlowMatchEulerAncestralRFScheduler(_FlowStepper):
    """Euler a RF：rectified flow 專用的 ancestral（R34 對照組，係數見 `_rf_plan`）

    **跟本檔的 `FlowMatchEulerAncestralScheduler` 是不同的公式。** 那支照「保留 √(1−η²) 的軌跡雜訊」推；
    這支先往下走到更低的 σ_down、再拉回 σ_t 並補雜訊。η=0 時逐位元等於確定性的 Euler（有測試釘住）。
    """

    def __init__(self, *a, eta: float = 1.0, s_noise: float = 1.0, **kw):
        super().__init__(*a, **kw)
        self.eta = float(eta)
        self.s_noise = float(s_noise)

    def _advance(self, x, v, i, generator):
        x0 = x - self.sigmas[i].to(x.device, torch.float32) * v
        s, t = float(self.sigmas[i]), float(self.sigmas[i + 1])
        if t <= 0:
            return x0
        reach, lift, spread = _rf_plan(s, t, self.eta)
        x = reach * x + (1.0 - reach) * x0
        if lift is not None:
            x = lift * x + _fresh_noise(x, generator) * self.s_noise * spread
        return x


class FlowMatchEulerChurnScheduler(FlowMatchEulerDiscreteScheduler):
    """Karras et al. 2022 Algorithm 2 的 churn（ComfyUI / A1111 的 euler）

    **這是跟 ancestral 完全不同的隨機化機制**，本專案原本沒有。
    ancestral 是「走到目標後換掉雜訊」；churn 是**先把 sigma 往上推**
    再用一個更大的確定性步走下來：

        γ      = min(s_churn/步數, √2 − 1)      （限制在 [s_tmin, s_tmax]）
        σ̂      = σ·(1 + γ)
        x      ← x + ε·s_noise·√(σ̂² − σ²)       （加噪到 σ̂）
        x      ← x + (σ_next − σ̂)·v             （從 σ̂ 走到 σ_next）

    直覺上它是「退一步再走兩步」，讓解算器有機會修正累積的誤差。
    論文建議 √2−1 的上限，超過會過度加噪。

    這裡照 flow matching 改寫：加噪那步要維持 x = (1−σ)x₀ + σε 的形式，
    所以是在 x₀ 固定的前提下把 σ 換成 σ̂——也就是

        x̂ = (1−σ̂)·x₀ + σ̂·(ε̂ 混入新雜訊後的結果)

    等價於在 VE 空間加 √(λ̂² − λ²) 的雜訊再換回來。速度要跟著重算，
    因為 σ 變了：v̂ = (x̂ − x₀)/σ̂。
    """

    def __init__(self, *a, s_churn: float = 0.0, s_tmin: float = 0.0,
                 s_tmax: float = float("inf"), s_noise: float = 1.0, **kw):
        super().__init__(*a, **kw)
        self.s_churn = float(s_churn)
        self.s_tmin = float(s_tmin)
        self.s_tmax = float(s_tmax)
        self.s_noise = float(s_noise)

    def step(self, model_output, timestep, sample, generator=None,
             return_dict=True, per_token_timesteps=None, **kw):
        if per_token_timesteps is not None:
            return super().step(model_output, timestep, sample,
                                generator=generator, return_dict=return_dict,
                                per_token_timesteps=per_token_timesteps, **kw)

        if self.step_index is None:
            self._init_step_index(timestep)

        i = self.step_index
        out_dtype = model_output.dtype
        x = sample.to(torch.float32)
        v = model_output.to(torch.float32)
        sigma = float(self.sigmas[i])
        sigma_next = self.sigmas[i + 1].to(device=x.device,
                                           dtype=torch.float32)

        n = self.num_inference_steps or 1
        gamma = 0.0
        if self.s_churn > 0 and self.s_tmin <= sigma <= self.s_tmax:
            gamma = min(self.s_churn / n, 2.0 ** 0.5 - 1.0)

        x0 = x - sigma * v

        if gamma > 0 and sigma > 0:
            # σ=1 會讓 α=0、λ=∞，第一步一定會踩到。夾法跟
            # FlowMatchERSDEScheduler._lambdas 一致。
            sigma = min(sigma, 1.0 - 1e-4)
            hat = min(sigma * (1.0 + gamma), 1.0 - 1e-4)
            # VE 空間的加噪量：√(λ̂² − λ²)，再乘回 α̂ 換到 flow 空間
            lam = sigma / (1.0 - sigma)
            lam_hat = hat / (1.0 - hat)
            var = max(lam_hat * lam_hat - lam * lam, 0.0)
            noise = randn_tensor(x.shape, generator=generator,
                                 device=x.device, dtype=torch.float32)
            a_hat = 1.0 - hat
            # 先把目前的狀態換算到 σ̂ 的水準，再加上差額雜訊
            x = (1.0 - hat) * x0 + hat * ((v + x0) * (lam / lam_hat)) \
                + a_hat * noise * self.s_noise * var ** 0.5
            sigma = hat
            v = (x - x0) / max(sigma, 1e-6)     # σ 變了，速度要重算

        prev = (x + (sigma_next - sigma) * v).to(out_dtype)
        self._step_index += 1
        if not return_dict:
            return (prev,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev)
