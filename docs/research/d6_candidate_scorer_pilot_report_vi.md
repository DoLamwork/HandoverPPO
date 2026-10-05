# D6 candidate scorer — triển khai, pilot và quyết định gate

## 1. Câu hỏi nghiên cứu

D6 kiểm tra một factor duy nhất: thay actor MLP phẳng bằng scorer dùng chung cho
các candidate, còn K=3, thứ tự `ranked_v1`, observation `qos_full_v1`, reward
E07, optimizer, seed và split đều giữ nguyên. Mục tiêu không phải chứng minh
model cuối, mà kiểm tra liệu inductive bias đối xứng có giúp PPO nhận ra C2/C3
hữu ích trước khi tốn budget 100k hay không.

Toàn bộ diagnostic dùng split TRAIN. Pilot chỉ đánh giá validation để hoàn tất
checkpoint contract; test và zero-shot không được chạy.

## 2. Phần đã triển khai

- `candidate_scorer_v1`: một encoder/head STAY riêng và một scorer chia sẻ trọng
  số cho từng block candidate 15 feature, ghép cùng global context.
- Critic vẫn nhận toàn bộ observation; reward và protocol không đổi.
- Actor head khởi tạo zero nên policy ban đầu chính xác đều 25% cho bốn action.
- Checkpoint lưu architecture và candidate feature size; checkpoint cũ thiếu
  field mới vẫn load bằng giá trị mặc định tương thích.
- Rollout lưu metadata của đúng transition. Audit có thể gắn counterfactual
  usefulness, normalized GAE advantage và xác suất trước/sau cùng một PPO
  update mà không đổi đường update chuẩn.
- Unit test kiểm tra permutation equivariance của candidate logits, checkpoint
  round-trip và alignment metadata/rollout.

Full suite: **182 test pass**.

## 3. Observation có phân biệt được candidate không?

State bank TRAIN mở rộng có 3.117 record từ 80 route, gồm hai nguồn Always-C1
và D30. Counterfactual cho số state có C2/C3 tốt hơn cả STAY và C1 khá ổn định:

| Horizon | State dương | Tỷ lệ |
|---:|---:|---:|
| 0,5 s | 171 | 5,49% |
| 1,0 s | 167 | 5,36% |
| 2,0 s | 168 | 5,39% |

Route-holdout linear probe cho nhãn “có C2/C3 hữu ích”:

| Horizon | Positive holdout | AP | Bootstrap route 95% CI |
|---:|---:|---:|---:|
| 0,5 s | 45 | 0,25 | 0,15–0,36 |
| 1,0 s | 39 | 0,20 | 0,08–0,36 |
| 2,0 s | 38 | 0,16 | 0,08–0,24 |

Kết luận: tín hiệu không biến mất theo horizon và observation có khả năng phân
biệt lớn hơn random, nhưng positive state hiếm và độ tin cậy còn rộng. Không có
bằng chứng để quy lỗi chính cho thiếu observation và mở D4 ngay.

## 4. Pilot D60 so với control D30

Cả hai chạy đúng 10.240 macro-step, seed/split seed 17. D60 chỉ đổi
`shared_mlp_v1` thành `candidate_scorer_v1`.

### 4.1 Validation deterministic

| Model | Return | Rate Mbps | Relative rate | RLF-time | HOF | PP | STAY |
|---|---:|---:|---:|---:|---:|---:|---:|
| D30 shared MLP | 10,820 | 19,214 | 73,99% | 1,725% | 0% | 0% | 100% |
| D60 candidate scorer | 10,820 | 19,214 | 73,99% | 1,725% | 0% | 0% | 100% |

Hai policy cho trajectory deterministic giống nhau: luôn STAY. Cả hai
`QOS_INFEASIBLE`. HOF/PP bằng 0 không phải kết quả tốt vì đạt được bằng cách bỏ
toàn bộ HO và làm giảm mạnh rate.

### 4.2 Phân phối xác suất trên cùng 3.117 state TRAIN

| Model | P(STAY) | P(C1) | P(C2) | P(C3) | Normalized entropy | Argmax |
|---|---:|---:|---:|---:|---:|---|
| D30 | 37,66% | 32,37% | 15,75% | 14,22% | 0,939 | STAY 100% |
| D60 | 33,50% | 24,16% | 21,51% | 20,84% | 0,986 | STAY 100% |

D60 không triệt tiêu C2/C3 trong distribution. Scorer tăng exploration và giảm
slot bias, nhưng mức tăng chưa vượt decision boundary của STAY.

Trên 167 state có alternative hữu ích ở horizon 1 giây:

| Model | Tổng P đặt vào alternative hữu ích | Argmax recall |
|---|---:|---:|
| D30 | 18,23% | 0% |
| D60 | 25,49% | 0% |

Đây là cải thiện cơ chế thật, nhưng chưa biến thành hành vi deterministic.

## 5. Same-transition advantage và PPO update

Mỗi dòng dưới chỉ tính action thực sự được sample ở state có nhãn tương ứng.

| Policy/action | Useful chosen | Mean normalized advantage | Advantage dương | ΔP sau update trên useful state |
|---|---:|---:|---:|---:|
| Fresh C2 | 12 | -0,140 | 41,7% | -0,0058 |
| Fresh C3 | 2 | -0,076 | 50,0% | -0,0060 |
| D30 C2 | 7 | +0,050 | 57,1% | -0,0168 |
| D30 C3 | 0 | n/a | n/a | -0,0182 |
| D60 C2 | 13 | +0,010 | 53,8% | -0,0063 |
| D60 C3 | 4 | +0,044 | 50,0% | -0,0093 |

D60 tạo nhiều mẫu C2/C3 hữu ích hơn D30 và advantage trung bình không còn âm.
Tuy nhiên update vẫn giảm xác suất C2/C3 ngay cả trên useful state. Nguyên nhân
phù hợp với dữ liệu là **credit dilution + cạnh tranh softmax**: useful state chỉ
khoảng 5%, còn gradient từ phần lớn transition ưu tiên STAY đủ mạnh để kéo cả
logit candidate xuống. Chỉ tăng entropy hoặc tăng tần suất thử sai không giải
quyết được sự mất cân bằng tín hiệu này.

## 6. Quyết định gate

**Không promote D60 lên screening 100k ở trạng thái hiện tại.** Pilot đạt nửa
gate cơ chế (xác suất useful alternative tăng, entropy còn cao) nhưng thất bại ở
gate hành vi/QoS (argmax recall 0%, STAY 100%, relative rate 73,99%). Chạy dài
hơn lúc này có nguy cơ chỉ củng cố nghiệm STAY.

Bước kế tiếp nên là một factor riêng **D61 A3-gated redecision**:

1. Giữ candidate scorer, observation, reward E07 và mọi hyperparameter của D60.
2. Sau action STAY, nếu điều kiện A3/event context vẫn còn hợp lệ, mở lại decision
   sau 100 ms thay vì từ chối vĩnh viễn event hiện tại.
3. Pilot D61 10.240 step cùng seed 17; không đồng thời đổi reward/observation.
4. Gate pilot: non-STAY deterministic >0, useful-alternative argmax recall >0,
   relative rate tăng rõ so với D30/D60, và RLF/HOF không xấu hơn 3GPP một cách
   mất kiểm soát.
5. Chỉ khi D61 qua gate mới chạy multi-seed/100k. Nếu D61 vẫn STAY 100%, mở
   nhánh constraint/reward D5 có kiểm soát; không quay lại tăng entropy mù.

D61 sau đó đã được triển khai và thoát STAY collapse, nhưng chưa chọn C2/C3 và
QoS còn dưới gate 98%. Do đó không promote lên 100k; bằng chứng mới mở D62 actor
phân cấp để tách credit timing khỏi credit chọn target. Xem
[báo cáo D61](d61_a3_redecision_implementation_report_vi.md).

## 7. Artefact và tái lập

- State bank: `results/analysis/p23_diagnostics/state_bank_train_v2/`
- Multi-horizon labels: `results/analysis/p23_diagnostics/counterfactual_train_horizons_v1/`
- Supervised probes: `results/analysis/p23_diagnostics/supervised_probe_h0p5_v1/`,
  `supervised_probe_h1p0_v1/`, `supervised_probe_h2p0_v1/`
- Transition signal: `results/analysis/p23_diagnostics/transition_signal_v2/`
- D60 pilot: `results/models/d6_pilots/D60_candidate_scorer_pilot_b10240_s17/`
- D30/D60 fixed-bank probe: `results/analysis/p23_diagnostics/probe_d6_pilot_v1/`
- D30/D60 counterfactual: `results/analysis/p23_diagnostics/counterfactual_d6_pilot_v1/`
- Reproduction script: `scripts/run_d6_candidate_scorer_pilot.ps1`

Giới hạn: đây là một seed, một budget pilot và TRAIN-bank diagnostic. Không có
claim test/zero-shot hoặc kết luận cuối cùng cho bài báo.
