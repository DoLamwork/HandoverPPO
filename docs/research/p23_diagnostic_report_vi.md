# Báo cáo D0–D2 sau P2/P3: đo quá trình học PPO trên cùng state

Ngày hoàn tất pilot: 05/10/2026.

Phạm vi: D0–D2 của
[`p2_p3_diagnostic_implementation_plan_vi.md`](p2_p3_diagnostic_implementation_plan_vi.md).
Không chạy screening 100k/300k mới, không đánh giá test hoặc zero-shot và không
ghi đè kết quả E04–E10. Một smoke 2.048 macro-step riêng được dùng duy nhất để
đo gradient thật của PPO.

## 1. Kết luận chính

Gate hiện tại mở **D3-A/D3-B (đối chứng actor/critic)** trước D4.

Ba bằng chứng khớp nhau:

1. Counterfactual tìm được alternative C2/C3 hữu ích ở 6–9/80 state khi so với
   best(STAY,C1), tùy horizon. Các reward fixed nhận ra ít nhất một alternative
   hữu ích ở 100% các positive state; vì vậy chưa có bằng chứng reward xếp sai
   các action tốt đủ mạnh để ưu tiên D4 reward alignment.
2. Dù reward có tín hiệu, argmax recall của E07–E10 trên positive state là 0%.
   Xác suất conditional dành cho useful alternative chỉ 2,0–32,0% tùy model và
   horizon, không action nào trở thành argmax.
3. Smoke gradient cho thấy critic gradient lớn hơn actor 66,9 lần ở update 1 và
   37,3 lần ở update 2. Tổng gradient bị global clip còn 2,04% và 2,46%; EV đi
   từ -0,00055 xuống -0,00121 và từ -0,00052 xuống -0,00086. Shared encoder bị
   critic cập nhật mạnh trong khi actor encoder gradient ở update đầu bằng 0 do
   policy head khởi tạo zero.

Đây đúng điều kiện “hidden collapse + critic gradient chi phối” trong gate D0–D2.
Đề xuất kế tiếp là triển khai D3-A và D3-B dưới dạng hai factor riêng, pilot
10.240 step trước; chưa mở D4/D5/D6 và chưa chạy screening dài.

## 2. D0 — chuẩn hóa nền và registry

### Thay đổi code

- `summarize_ablation` lấy selected step từ checkpoint filename/SHA-256/counter,
  chỉ fallback sang metric matching cho artifact lịch sử và đánh dấu inferred.
- Registry ghi tách `selected_validation`, `last_scheduled_validation` và
  `budget_end_validation`, kèm training-last step, denominator/count, P10,
  per-speed, action distribution và provenance hash.
- Training mới tự validation đúng budget-end nếu interval không chia hết budget;
  không duplicate khi budget đã trùng scheduled validation.
- Checkpoint cũ thiếu field diagnostics vẫn load thành công; diagnostics mặc
  định tắt.

### Kết quả audit E04–E10

Tất cả checkpoint selected được truy ra bằng `checkpoint_sha256`, không còn là
suy đoán từ dòng metric giống nhau.

| Arm | Selected step | Last scheduled | Training last | Budget-end cũ |
|---|---:|---:|---:|---|
| E04 | 81.920 | 92.160 | 100.000 | không có |
| E05 | 61.440 | 92.160 | 100.000 | không có |
| E06 | 71.680 | 92.160 | 100.000 | không có |
| E07 | 40.960 | 92.160 | 100.000 | không có |
| E08 | 10.240 | 92.160 | 100.000 | không có |
| E09 | 10.240 | 92.160 | 100.000 | không có |
| E10 | 10.240 | 92.160 | 100.000 | không có |

`budget_end=null` là thông tin đúng cho run lịch sử; code không bịa validation
100k từ training checkpoint. Run mới sẽ tạo record đúng tại budget cuối.

Artifact:

- `results/analysis/p23_diagnostics/registry_v1/ablation_registry.json`
- SHA-256: `81e1be697d45d429169a498112523bf1f2a71088bee532a7ebb89ceda6c10e26`

## 3. D1 — state bank, activation, policy và gradient

### State bank TRAIN pilot

- Split: TRAIN, split seed 17, sample seed 17.
- 5 route/speed × 4 speed × decision index 0/1/2/3 = 80 state.
- Nguồn state: Always-HO-C1, đúng pilot preregistered.
- Cùng physical state lưu đồng thời `rolling_v1`, `monotonic_v1` và
  `qos_full_v1`; K=3, `ranked_v1`.
- `.npz` chỉ chứa numeric/bool/fixed Unicode arrays, load bằng
  `allow_pickle=False`.
- Manifest lưu hash từng RSRP/SINR trace, dataset provenance, protocol config,
  source policy, candidate identity, mask, feature names và NPZ hash.
- Replay D2 kiểm tra lại tic, serving cell, candidate physical ID, mask và toàn
  bộ observation với tolerance `1e-7`.

Artifact:

- `results/analysis/p23_diagnostics/state_bank_pilot_v1/state_bank_manifest.json`
- `state_bank.npz` SHA-256:
  `8a6ab44cb61212a7cc0c2aa133695e709c2a27e099640bc7602e5b40c1ea376f`
- Manifest SHA-256:
  `935a9ea742fe2b91fd41fd30c5bce22e88098d4a71f880dc391b89939fa46a32`

### Probe checkpoint trên cùng 80 state

| Checkpoint | step | Hidden-2 abs≥0,99 | Entropy chuẩn hóa | P(STAY) | P(C1) | P(C2+C3) | Argmax |
|---|---:|---:|---:|---:|---:|---:|---|
| E04 latest | 100k | 37,91% | 0,525 | 48,86% | 50,57% | 0,57% | C1 77/80 |
| E05 latest | 100k | 45,41% | 0,513 | 55,36% | 44,27% | 0,37% | STAY 80/80 |
| E06 latest | 100k | 10,10% | 0,528 | 54,42% | 44,80% | 0,78% | STAY 80/80 |
| E07 selected | 40.960 | 0,00% | 0,814 | 41,00% | 42,99% | 16,01% | STAY 13, C1 67 |
| E07 latest | 100k | 0,64% | 0,535 | 28,01% | 68,84% | 3,15% | C1 80/80 |
| E08 latest | 100k | 97,93% | 0,823 | 10,14% | 57,25% | 32,61% | C1 80/80 |
| E09 latest | 100k | 97,66% | 0,506 | 9,19% | 80,45% | 10,36% | C1 80/80 |
| E10 selected | 10.240 | 0,08% | 0,999 | 25,51% | 26,88% | 47,61% | C1 80/80 |
| E10 latest | 100k | 97,97% | 0,826 | 8,90% | 56,28% | 34,82% | C1 80/80 |

Fresh models cho cả ba schema có đúng phân phối 25% mỗi action và entropy chuẩn
hóa 1,0. Vì vậy collapse không đến từ action mask hay initialization. E08–E10
latest có hidden saturation rất nặng nhưng probability C2+C3 vẫn không bằng
argmax; chỉ nhìn entropy là không đủ phát hiện bias.

Input `qos_full_v1` có 9,74% giá trị `abs>=0,99`, nhưng phần lớn là feature nhị
phân hợp lệ (`valid`, `is_a3_target`) hoặc coverage; không được diễn giải toàn bộ
như lỗi normalization. Hidden saturation mới là dấu hiệu mạng cần xử lý.

Artifact probe:

- `results/analysis/p23_diagnostics/state_bank_pilot_v1/probe_v1/`
- `probe_summary.json` SHA-256:
  `1688a1311450fcc051f90edb1b96946c52c73ac18bc3a48eaf9bffdbb9e618ed`

### Gradient smoke 2.048 step

| Update | Actor norm | Critic norm | Critic/actor | Cosine | Total pre-clip | Clip factor | EV pre → post |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,366 | 24,483 | 66,9× | 0,0000 | 24,486 | 0,0204 | -0,00055 → -0,00121 |
| 2 | 0,546 | 20,345 | 37,3× | 0,0012 | 20,353 | 0,0246 | -0,00052 → -0,00086 |

Update 1: actor encoder norm = 0, critic encoder norm = 17,538. Update 2:
actor encoder norm = 0,00759, critic encoder norm = 14,425. Return std là
12,34/13,32 nhưng value std chỉ 0,24/0,23. Đây là smoke cơ chế, không dùng metric
validation để tuyên bố hiệu năng; model 2.048 step collapse 100% STAY và QoS fail.

Artifact:

- `results/models/diagnostics/D01_gradient_pilot_qos_rate_k3_s17/`
- `training_report.json` SHA-256:
  `fb6cdcae6d17fe8a5d85cb0134d1cc40928a4d0159fbe220d89263376a0787e4`
- `test_evaluated=false`.

Diagnostics đo actor, critic và entropy bằng `autograd.grad` trước optimizer step,
không ghi vào `.grad`; unit test xác nhận bật/tắt diagnostics với cùng RNG tạo
cùng trọng số sau update.

## 4. D2 — physical outcome, reward và action probability cùng state

Mỗi anchor được clone độc lập. Action đầu là STAY/C1/C2/C3, sau đó continuation
luôn C1. Horizon 0,5/1/2 giây dùng cùng trace và candidate identities. Reward
legacy/E07/E08/E09 được reset bằng full trace như lúc train; lưu cả raw và
discounted sum với exponent tick đầu bằng 0.

| Horizon | Positive vs C1 | Positive vs best(STAY,C1) | E07 selected P(useful\|+) | E10 selected P(useful\|+) | Argmax recall mọi checkpoint |
|---:|---:|---:|---:|---:|---:|
| 0,5 s | 10/80 | 9/80 | 8,25% | 29,44% | 0% |
| 1,0 s | 8/80 | 7/80 | 8,53% | 30,88% | 0% |
| 2,0 s | 9/80 | 6/80 | 8,76% | 31,95% | 0% |

Với định nghĩa useful nghiêm ngặt (delivered Mbit cao hơn và không tăng
HOF/RLF/PP/recovery), cả bốn fixed reward nhận ra ít nhất một useful action ở
100% positive state. Tỷ lệ từng useful action có reward tốt hơn best(STAY,C1)
là 90,91%, 88,89% và 87,50% theo ba horizon. E08 có false alternative argmax
0%; legacy/E07 là 2,5–3,75%; E09 là 1,25%.

Raw-vs-discounted argmax gần như không đổi (chỉ legacy tại 0,5 s đổi 1,25%), nên
SMDP discount không phải nguyên nhân chính trong pilot. STAY đúng là suppress
event: tại 0,5 s, 78,75% nhánh STAY chưa có decision tiếp theo, so với 60% của
C1; nhưng pilot chưa chứng minh suppression là bottleneck lớn hơn optimizer.

Artifact:

- `results/analysis/p23_diagnostics/state_bank_pilot_v1/counterfactual_v2/`
- `counterfactual_summary.json` SHA-256:
  `fd6cd84db52b9f7bcfb4d709a8b97ae8b687f9036422fa4d8e461a2de9d2d9c3`

## 5. Unit tests và tương thích

- Full suite: **173 tests, tất cả pass**.
- Có test budget 5/interval 4 tạo validation tại step 4 và 5; budget 4 chỉ tạo
  một validation tại step 4.
- Có test checkpoint cũ thiếu diagnostics field vẫn load.
- Có test bật diagnostics không đổi PPO update.
- Có test state bank TRAIN/pickle-free/hash/count.
- Có test clone không mutate engine, physical outcome khớp helper cũ, legacy
  reward identity, exponent-0 discount và terminal horizon.
- `git diff --check` phải được chạy lại ở bước bàn giao cuối.

## 6. Cách tái lập

Script:

```powershell
.\scripts\run_p23_diagnostics.ps1
```

Script mặc định tạo output mới dưới
`results/analysis/p23_diagnostics_reproduction` và từ chối ghi đè. Thêm
`-RunGradientPilot` nếu muốn chạy lại smoke 2.048 step; mặc định không tự train.
Các CLI riêng:

```text
run.py summarize_ablation
run.py audit_policy_learning build-bank
run.py audit_policy_learning probe
run.py audit_policy_learning counterfactual
run.py train_ppo --learning-diagnostics ...
```

## 7. Giới hạn và bước kế tiếp

- Pilot chỉ có 80 state từ Always-HO-C1 và bốn decision đầu; chưa phải estimate
  cuối cho toàn TRAIN distribution hoặc state distribution của E07.
- Không dùng pilot để chọn checkpoint, tune threshold hoặc claim vượt 3GPP.
- Chưa kiểm tra interaction architecture × value scaling.
- E10 dual target mismatch đã biết vẫn giữ nguyên để replay; chưa mở D5.

Phần triển khai D3-A/D3-B đã hoàn tất ngày 05/10/2026; xem
[báo cáo triển khai D3](d3_optimizer_ablation_implementation_report_vi.md).
Các pilot dưới đây chưa được tự động chạy.

Thứ tự chạy tiếp theo:

1. D3 control mới với instrumentation/budget-end validation.
2. D3-A `separate_mlp_v1`, giữ toàn bộ factor khác.
3. D3-B `fixed_scale_v1` trên shared control; fit scale từ TRAIN và serialize.
4. Chạy pilot 10.240 step cho control/A/B; gate bằng critic/actor norm,
   clipping factor, EV, hidden saturation và useful-action probability.
5. Chỉ chạy interaction D3-A×B hoặc screening 100k nếu pilot có cải thiện cơ
   chế. D4 chỉ mở nếu sau khi ổn định optimizer, fixed reward bắt đầu xếp thấp
   physical action tốt hoặc observation compact/full cần đối chứng riêng.
