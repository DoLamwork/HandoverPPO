# Kế hoạch sau P2–P3: diagnostic và sửa quá trình học PPO

Ngày lập: 04/10/2026. Người thực hiện tiếp theo: Codex Sol hoặc người phát triển.
Trạng thái: **kế hoạch triển khai, chưa triển khai các command/flag mới trong tài liệu này**.

## 0. Bắt đầu ở đâu

Đọc toàn bộ tài liệu, sau đó làm **D0 → D1 → D2**. Đây là gói công việc đầu tiên:
chuẩn hóa báo cáo, tạo state bank TRAIN, kiểm tra mạng và counterfactual reward.
Chưa mở đồng thời mọi nhánh D3–D6 hoặc chạy hàng loạt 300k.

Mục tiêu nghiên cứu giữ nguyên: giữ/cải thiện QoS so với 3GPP, đồng thời giảm
RLF-time, HOF và ping-pong. Tỷ lệ chọn C2/C3 là diagnostic, không phải thành tích
cần tăng bằng mọi giá. Tiếp tục dùng feed-forward MLP PPO và A3 làm trigger.

Kết thúc mỗi gói phải có: code, test thích hợp, lệnh tái lập đã thực sự chạy,
artifact path, kết luận có bằng chứng và bước kế tiếp. Không ghi một CLI dự kiến
thành lệnh đang hoạt động khi chưa nối parser/dispatch/implementation.

Branch lúc lập kế hoạch: `codex/p3-reward-ablation`.
Source revision của screening P3: `5ecdfda559743d223630de2ac9637da644b5f87d`.
Khi bắt đầu triển khai, kiểm tra HEAD và working tree thực tế; dùng nhánh mới
`codex/p23-diagnostics` nếu người dùng yêu cầu nhánh hoặc yêu cầu triển khai kế
hoạch này trên nhánh riêng. Bảo toàn thay đổi có sẵn, không checkout/reset đè.

## 1. Các dữ kiện đã biết và giới hạn kết luận

### 1.1. Screening đã hoàn tất

E04–E10 đều đã train đủ 100.000 macro-step, seed/split seed 17,
`test_evaluated=false`. Mỗi run có 98 PPO update và 9 validation theo lịch.
Validation theo lịch cuối là 92.160 step; **không gọi đó là validation tại 100k**.

| Arm | Thay đổi | Step selected | Relative rate / 3GPP | RLF-time | HOF | PP | STAY |
|---|---|---:|---:|---:|---:|---:|---:|
| 3GPP | Reference | — | 100% | 1,313% | 10,903% | 37,057% | 0% |
| E04 | History theo physical cell | 81.920 | 100,02% | 1,273% | 10,656% | 36,916% | 0,64% |
| E05 | Monotonic normalization | 61.440 | 100% | 1,313% | 10,903% | 37,057% | 0% |
| E06 | Observation 62 chiều | 71.680 | 96,13% | 1,194% | 11,002% | 25,021% | 16,59% |
| E07 | Trajectory-normalized utility | 40.960 | 99,08% | 0,959% | 8,128% | 36,022% | 13,18% |
| E08 | Rescale event penalties | 10.240 | 100% | 1,313% | 10,903% | 37,057% | 0% |
| E09 | Thêm recovery penalty | 10.240 | 100% | 1,313% | 10,903% | 37,057% | 0% |
| E10 | Thêm adaptive QoS coefficient | 10.240 | 100% | 1,313% | 10,903% | 37,057% | 0% |

Các selected checkpoint trên không chọn C2/C3. E06 không qua QoS gate.
E05 và E06 về cuối quay lại nhiều STAY; E07 từ 81.920 step tái tạo 3GPP.
E07 selected có P10 relative rate 82,459%, thấp hơn 84,692% của 3GPP.
Không diễn giải QoS mean/per-speed pass thành đảm bảo mọi route.

Nguồn số liệu: `results/models/ablation_v2/<experiment_id>/training_report.json`,
`checkpoints/validation_metrics.json`, `training_metrics.jsonl`.
Đọc lại artifact khi triển khai; bảng này là số làm tròn, không phải ground truth
để hard-code vào test.

### 1.2. Phát hiện diagnostic trực tiếp ngày 04/10/2026

Một probe chỉ đọc đã lấy 5 route TRAIN/speed, seed 17, 4 decision đầu/route,
continuation Always-HO-C1: 20 route, 80 state. Đưa cùng state vào checkpoint:

| Checkpoint | Tỷ lệ activation tầng tanh thứ hai có abs(h)>0,99 |
|---|---:|
| E07 selected 40.960 | 0% |
| E07 latest 100k | 0,6445% |
| E08 latest 100k | 97,9297% |
| E09 latest 100k | 97,6563% |
| E10 latest 100k | 97,9688% |

E08–E10 có độ biến thiên biểu diễn tầng cuối rất nhỏ giữa các state. E10
selected 10.240 chưa bão hòa mạnh nhưng vẫn chọn C1 ở cả 80 state vì xác suất
C1 khoảng 26,9%, nhỉnh hơn các action khác. Không đồng nhất argmax C1 100%
với xác suất C1 bằng 1 hoặc entropy bằng 0.

Probe này mới có output trong chat, **chưa có script/artifact versioned**.
D1 phải tái lập và lưu provenance; không trình bày nó như audit đầy đủ dataset.

### 1.3. Các nguyên nhân cần phân biệt

| Vấn đề | Mức bằng chứng | Phép kiểm tra tiếp |
|---|---|---|
| E08–E10 latest bão hòa hidden representation | Đã đo trên mẫu 80 state | D1: tái lập, tăng mẫu, so selected/latest/initial |
| Critic làm actor mất khả năng phân biệt state | Giả thuyết; encoder và clipping đang dùng chung | D1 gradient riêng, D3 đối chứng encoder riêng |
| E10 target sai so với ý định giữ 98% 3GPP | Xác nhận từ code | D5 version constraint mới dựa TRAIN baseline |
| Alternative bị mask | Không thấy trong rollout cuối; 4 action đều mở | D1/D2 kiểm tra từng state, không suy từ một rollout ra mọi state |
| Candidate hữu ích không tồn tại | Đã có phản ví dụ P1 | D2: mở rộng positive state, đo policy/reward trên chúng |
| Observation thiếu thông tin hữu ích | Chưa kết luận | D2 learnability probe theo route |
| 100k quá ít / mạng 64x64 quá nhỏ | Chưa kết luận | Chỉ tăng sau diagnostic và learning curve |

P1 đã có 30–34/589 state có alternative tăng delivered Mbit mà không tăng
các lỗi so với best(STAY,C1), tùy horizon 0,5/1/2 s. K3 giữ được ít nhất một
alternative hữu ích trong các positive state đó. Đây là tiêu chí dominance
chặt trên mẫu, không có nghĩa mọi state còn lại đều không thể hưởng lợi.

## 2. Hợp đồng bất biến của các đối chứng

- Data: `data/uav_100m_grid100m_dev_processed`; dùng manifest/split hiện có.
- 700 train / 150 validation / 150 test ở 30/50/70/90 km/h; giữ split seed 17.
- Diagnostic/fit/normalization dùng TRAIN; validation chọn cấu hình/checkpoint.
  Không đọc test/zero-shot trong giai đoạn này.
- Runtime: 2,1 GHz, 10 MHz, dt=10 ms. Không sửa provenance generator 4 GHz/20 MHz
  hoặc tạo lại dataset để làm các diagnostic này.
- K=3, `ranked_v1`, structural mask, action STAY/C1/C2/C3.
- A3 completed-TTT trigger; STAY reject-event giữ nguyên đến D6 nếu có bằng chứng.
- E07 reward: `trajectory_qos_reward_v1`; RLF=5, PP=1, utility_scale=60.
- MLP tanh 64x64; lr=5e-5, rollout=1024, batch=128, epochs=4, entropy=0,01,
  clip=0,2, value coefficient=0,5, max grad norm=0,5, target KL=0,03.
- Gamma/second=0,99, GAE lambda/second=0,95; giữ F1 intra-macro discount.
- Mọi arm mới khởi tạo mới. Parent ID là lineage, không phải warm-start.
- Không sửa/xóa artifact E00–E10. Thư mục mới phải từ chối ghi đè nếu đã có dữ liệu.
- Không sửa reward schema `*_v1` để âm thầm thay nghĩa của checkpoint cũ.
- Không thưởng riêng cho C2/C3, phạt C1, dùng future trong actor/mask/selector,
  hoặc chuyển benchmark sang random chỉ để làm kết quả khác 3GPP.

## 3. Bản đồ code để Sol tìm đúng chỗ

| File hiện có | Vai trò |
|---|---|
| `src/ho_optim_drl/agents/ppo/network.py` | Shared encoder, actor/value heads, categorical sampling |
| `src/ho_optim_drl/agents/ppo/trainer.py` | Collector, loss, clip gradient, validation, checkpoint |
| `src/ho_optim_drl/agents/ppo/rollout.py` | Buffer, duration-aware GAE, advantage normalization |
| `src/ho_optim_drl/agents/ppo/config.py` | Cấu hình PPO đóng băng |
| `src/ho_optim_drl/envs/handover.py` | Reset prefix, step, reward/state contract |
| `src/ho_optim_drl/smdp/multicandidate.py` | Observation, history, pool/ranking, mask |
| `src/ho_optim_drl/smdp/reward.py` | Reward versioning, normalizer, dual state |
| `src/ho_optim_drl/smdp/transition.py` | Primitive-to-macro aggregation |
| `src/ho_optim_drl/protocol/engine.py` | A3, STAY suppression, target execution |
| `src/ho_optim_drl/commands/audit_candidates.py` | Clone protocol state và counterfactual physical outcome |
| `src/ho_optim_drl/commands/audit_observations.py` | Sampling route, observation audit |
| `src/ho_optim_drl/commands/summarize_ablation.py` | Registry/report artifact cũ |
| `src/ho_optim_drl/commands/train_smdp_ppo.py` | Experiment metadata, logging, train orchestration |
| `src/ho_optim_drl/cli.py` | Parser/dispatch |
| `tests/test_smdp_ppo.py` | PPO, GAE, checkpoint, integration tests |
| `tests/test_candidate_audit.py`, `tests/test_observation_audit.py` | Audit/causality tests |
| `tests/test_reward.py`, `tests/test_ablation_summary.py` | Reward và registry tests |

File mới nên tách chức năng, ví dụ `agents/ppo/diagnostics.py`,
`commands/audit_policy_learning.py`, `tests/test_policy_diagnostics.py`.
Tái sử dụng helper hiện có khi phù hợp; không copy toàn bộ trainer hoặc dựng
một protocol engine khác chỉ cho diagnostic.

## 4. D0 — khóa evidence và chuẩn hóa báo cáo

### Công việc

1. Kiểm tra experiment ID, source/data/split hash, budget, test flag và existence
   của selected/latest. Ghi selected step, last scheduled step, training-last step.
2. Dùng/mở rộng `summarize_ablation` để xuất bảng P2–P3 selected và last scheduled
   vào `results/analysis/p23_diagnostics/registry_v1/`.
3. Sửa cách xác định selected step nếu cần: ưu tiên counter/metadata/checkpoint
   source path; việc lấy dòng metric trùng đầu tiên chỉ là fallback cho artifact
   legacy và phải gắn nhãn `inferred`. Nhiều checkpoint có metric giống nhau.
4. Cập nhật trạng thái P2/P3 trong docs: screening đã chạy. Phân biệt smoke,
   screening, probe 80 state và kiểm chứng nhiều seed chưa có.
5. Với các run MỚI, thêm validation đúng budget-end khi budget không chia hết
   evaluation interval. Tránh evaluate/save trùng nếu budget-end trùng lịch.
   Cả control và treatment mới phải dùng cùng lịch này.
6. Báo ba trường riêng: `selected_validation`, `last_scheduled_validation`,
   `budget_end_validation`. Với run cũ thiếu budget-end, ghi null; không giả lập
   hoặc ghi bổ sung vào training_report cũ. Nếu evaluate latest cũ, lưu dưới
   diagnostic mới với checkpoint hash và nhãn `posthoc_validation`.

### Kiểm thử và điều kiện hoàn tất

- Report không nhầm 92.160 thành 100.000 hoặc selected thành latest.
- Test nhỏ budget=5, interval=4 có evaluation tại 4 và 5; budget=4 chỉ tại 4.
- `--skip-final-test` vẫn bảo đảm không đọc test.
- Registry giữ counts và denominator: HOF/HO-attempts, PP/HO-successes,
  recovery-time/total-time; không chỉ báo %.
- Ghi P10 và per-speed gate, không đánh đồng mean gate với QoS mọi route.

## 5. D1 — state bank và diagnostic mạng có thể tái lập

### 5.1. State bank

Giai đoạn pilot tái lập đúng probe đã có:

- Split TRAIN theo manifest, 5 route mỗi speed, sampling seed 17.
- Dùng `_sample_routes` hiện có hoặc helper tương đương đã kiểm thử deterministic.
- State source Always-HO-C1, 4 decision đầu mỗi route; dự kiến 80 state.
- Thu observation `rolling_v1`, `monotonic_v1`, `qos_full_v1` trên cùng physical
  state. Lịch sử được cập nhật từ đầu route theo cell ID.

Sau pilot, tạo bank chính 20 route/speed, nguồn state gồm Always-HO-C1 và E07
selected 40.960; lấy decision index 0/10/20/30 nếu tồn tại. Lưu số mẫu thực tế,
không ép đủ quota bằng cách lặp state hoặc bỏ route không thuận lợi.

Lưu `state_bank_manifest.json` và arrays `.npz` load với `allow_pickle=False`.
Mỗi anchor có `state_id`, route/speed, source checkpoint hash hoặc baseline ID,
decision index/tic, serving, candidates, mask và schema names. Manifest có
source/data/split/topology hash, sampling config và số route không đủ decision.

Không pickle live engine. D2 tái lập engine bằng replay source policy tới anchor
và kiểm tra ID/tic/observation rồi clone trong RAM. Với state source PPO, replay
deterministic và load đúng observation/reward state của checkpoint.

### 5.2. Forward-only diagnostic

So sánh ít nhất E04/E05/E06 latest, E07 selected/latest, E08/E09 latest,
E10 selected/latest và một model khởi tạo mới tương ứng mỗi schema.

Xuất trên cùng bank:

- Input: finite/range, mean/std từng feature, feature hằng/thưa; tách role bits.
- Mỗi tanh layer: `fraction_abs_gt_0_95`, `fraction_abs_gt_0_99`,
  `mean_feature_std_across_states`; ghi đó là tỷ lệ activation state×unit.
- Logits/probabilities theo action: mean/std/p10/p50/p90, entropy chuẩn hóa,
  top1–top2 probability margin, argmax count, mask availability.
- Value mean/std/p10/p90; chưa tính EV nếu không có return target phù hợp.
- Báo riêng state source và speed; không trộn một bank khác rồi gọi là so cùng state.

Không chỉ đo trên rollout đang thay đổi: bank cố định giúp phân biệt policy
thay đổi với việc nó đang tới những state khác.

### 5.3. Diagnostic trong quá trình update

Thêm opt-in diagnostics; default cũ không thay learning behavior.
Trên minibatch đầu mỗi update, trước backward/step, đo bằng `autograd.grad`:

- Norm gradient của `policy_loss`, `value_coefficient * value_loss`,
  `-entropy_coefficient * entropy` riêng.
- Norm trên encoder và từng head; cosine actor-vs-critic gradient trên shared
  encoder. Với architecture riêng, cosine shared encoder là null.
- Norm tổng trước clipping; hệ số clipping thực tế xấp xỉ
  `min(1, max_grad_norm / (norm + eps))`.
- Return/advantage/value distribution; log cả `explained_variance_pre_update`
  và `explained_variance_post_update` trên cùng buffer.

EV hiện tại dùng `old_values`; giữ field cũ để tương thích và ghi rõ semantics.
Loss magnitude lớn không tự chứng minh gradient critic lấn át: phải đo norm.
Norm gradient lưu mean theo update không được dùng để suy ra chính xác clipping
của từng minibatch; log clipping factor tại nơi thực sự áp dụng.

Sau mỗi 10 update và tại budget-end, chạy forward-only bank. Không dùng bank
để cập nhật policy trong D1.

### Output và test

- Command đề xuất cần xây: `run.py audit_policy_learning` với các chế độ
  `build-bank`, `probe`; các flag phải được ghi trong `--help` sau triển khai.
- Artifact: bank manifest/arrays, `network_probe.json`, `network_probe.md`,
  `policy_state_diagnostics.jsonl` của run mới.
- Test bật/tắt diagnostics trên cùng seed/buffer cho cùng parameter update,
  cùng sample order và RNG state trong tolerance CPU đã định.
- Restore model train/eval mode; không thay `.grad`, optimizer, reward/dual,
  active environment hoặc RNG Python/NumPy/Torch khi đo diagnostic.
- Test saturated synthetic layer, constant outputs, unavailable action và
  single-valid-action entropy; không hard-code tỷ lệ 98% của checkpoint thật.
- Done khi tái lập được probe và có bảng so cùng state; không đòi phải chứng minh
  nguyên nhân bão hòa ngay trong D1.

## 6. D2 — nối physical outcome, reward và khả năng chọn action

### 6.1. Counterfactual trên cùng state

Mở rộng helper `simulate_counterfactual` hoặc tách helper dùng chung, giữ API
và output P1 legacy vẫn đọc được.

Tại anchor D1, clone toàn bộ protocol state, giữ trace immutable, thử từng
action hợp lệ STAY/C1/C2/C3. Freeze mapping candidate physical ID ở decision
đầu; không lấy candidate của một branch khác. Mỗi branch sau đó cùng dùng
Always-HO-C1 làm continuation.

Horizons chính: 0,5 / 1 / 2 giây tính từ cùng decision tic. Đây không phải một
macro-step cho mỗi branch, vì macro-duration khác nhau. Route kết thúc sớm thì
ghi actual horizon/terminal; future không tạo thêm dữ liệu hay sự kiện.
Horizon tới cuối route chỉ chạy pilot hoặc subset đã chọn trước, không chỉ
chọn lại các state có kết quả thuận lợi.

Ghi physical outcome và breakdown primitive reward cho legacy, E07, E08, E09.
E10 chỉ là diagnostic bổ sung với lambda được freeze từ checkpoint chỉ định;
không cập nhật lambda trong branch và không so return có lambda khác nhau.
Reward model phải reset trên đúng full trace trước replay/branch; không reset
engine để tính normalizer, không tính normalizer từ suffix horizon.

Lưu cả raw reward và discounted reward với exponent đầu bằng 0, cùng gamma/dt.
Không thêm prefix trước decision vào reward action; full-route QoS vẫn có prefix.

### 6.2. Các câu hỏi report phải trả lời

1. Bao nhiêu state alternative strict-dominates C1 và best(STAY,C1)?
2. Trên positive state, probability mass lên useful alternatives, argmax recall
   và top1–top2 margin là bao nhiêu? So với negative state trên cùng bank.
3. Action có lợi về physical metrics có bị reward xếp thấp hơn không? Xuất
   `delta_reward`, `delta_mbit`, `delta_hof/pp/rlf/recovery` so từng reference.
4. Xếp hạng đổi ra sao khi có discount? Phân biệt nguyên nhân reward weighting
   với short-horizon/continuation assumption.
5. STAY có khoảng suppression dài rồi mất rate hoặc gặp RLF không? Lưu thời
   gian đến decision tiếp theo, event leave/target-change; đây là bằng chứng D6.

Reuse định nghĩa dominance P1 và tolerance hiện có. Khóa tolerance trước khi
đọc kết quả. Không tự cộng metric khác đơn vị thành một score bí mật.
Outcome dưới continuation cố định không phải Q* hoặc policy oracle triển khai.

### 6.3. Learnability probe tùy chọn sau khi có đủ positive state

Chia route TRAIN đã lấy thành fit/inner-holdout theo route, phân tầng speed;
mọi state source/timestamp của cùng route ở cùng fold. Khóa inner split seed 29.
Không dùng official validation/test để fit mô hình diagnostic.

Fit một mô hình nhỏ chỉ nhận observation causal, dự đoán nhãn alternative có
lợi hoặc delta outcome. Nhãn dùng future chỉ là diagnostic offline.
Báo prevalence, precision/recall/PR-AUC và interval/variation theo route;
accuracy của class majority không đủ. Nếu positive quá ít, báo insufficient
evidence rồi mở rộng sample trước khi kết luận observation thất bại.

Probe này không thay PPO, không warm-start policy và không trộn dữ liệu offline
vào PPO on-policy minibatch. Nếu sau này dùng auxiliary loss/behavior cloning,
đó là experiment riêng với disclosure rõ.

### Output và test

- Command đề xuất: mode `counterfactual` của `audit_policy_learning`.
- Output: `counterfactual_outcomes.jsonl`, `reward_action_alignment.json`,
  `diagnostic_report_vi.md`; thêm probe report nếu đã đủ dữ liệu.
- Test clone không làm thay original engine/history; equality physical outcome
  với helper P1; tick alignment, zero-rate execution/recovery, HOF/RLF overlap,
  discount và terminal-short-horizon.
- Thay future sau tic không đổi observation/mask/candidates; nó được phép đổi
  counterfactual outcome. Không đưa label/outcome vào bank feature dùng policy.

### Cổng quyết định sau D0–D2

| Evidence | Bước ưu tiên |
|---|---|
| Hidden collapse + critic gradient chi phối | D3-A/D3-B |
| Physical action tốt bị fixed reward xếp thấp | D4 reward alignment |
| Reward xếp đúng, probe phân biệt được, PPO vẫn không chọn | D3 tối ưu, rồi D6 candidate scorer nếu cần |
| Probe khó học và predictor nhiều nhiễu | D4 observation đối chứng/leave-one-out; chưa kết luận chỉ cần tăng mạng |
| STAY suppression gây mất cơ hội rõ | D6 redecision, giữ A3 gate lần đầu |
| Chưa đủ positive hoặc dữ kiện mâu thuẫn | Mở rộng diagnostic, không chọn cấu hình từ phỏng đoán |

## 7. D3 — đối chứng tối ưu actor/critic, giữ mục tiêu E07

Chỉ triển khai/train các arm đã được evidence D1/D2 hỗ trợ. Các flag trong phần
này là API dự kiến, cần thêm đồng bộ config/CLI/checkpoint/report.

### D3-A: encoder riêng

- Thêm `network_architecture`: `shared_mlp_v1` (default) và `separate_mlp_v1`.
- Separate có actor encoder 64x64 và critic encoder 64x64, cùng tanh; giữ uniform
  policy initialization, action masks, loss coefficients, optimizer/lr và global
  clipping như control. **Không đồng thời đổi thành clip riêng hoặc lr riêng.**
- Giữ interface distribution/value/act. Collector và GAE vẫn nhận value raw units.
- Log tổng và actor/critic parameter count. Separate tăng số parameter; muốn
  quy lợi ích riêng cho bỏ gradient interference cần control matched-capacity
  sau này, không khẳng định từ một so sánh duy nhất.
- `network.py`, `config.py`, `trainer.py`, CLI và experiment manifest phải cùng
  hiểu architecture. Load checkpoint thiếu field này như shared legacy.
- Lưu architecture trong model_spec; loader không đoán separate từ key encoder
  legacy. Không load shared weights vào separate hoặc resume chéo architecture.
- Test gradient actor loss không đi vào critic encoder và ngược lại; round-trip
  checkpoint và benchmark auto-reconstruct; old checkpoints vẫn load an toàn.

### D3-B: value target theo scale cố định

Đối chứng riêng với D3-A, dùng shared encoder trước. Chọn cách đơn giản, tĩnh
để kiểm chứng cơ chế trước khi triển khai PopArt/running normalization.

- API dự kiến: `value_normalization=none|fixed_scale_v1`, `value_target_scale=s`.
- Fit một scalar s từ discounted return-to-go TRAIN của baseline C1 trên bank
  calibration đã freeze: `s=max(1, sqrt(mean(G_train**2)))`.
  Với E07 dùng E07 reward và gamma đã khóa. Lưu cách fit, route/hash và s.
- Critic head trả z; value công khai cho collection/bootstrap/GAE là `V=s*z`.
- Value loss: `0.5 * mean((z - G/s)**2)`; raw reward và advantage semantics
  giữ nguyên. Không vô tình scale reward một lần và value/return lần thứ hai.
- Scale cố định suốt run, serialize trong checkpoint; `none` tương đương s=1.
- Dùng scale đã fit từ TRAIN cho validation/test sau này; không fit lại theo
  từng validation, minibatch hoặc checkpoint.
- Test bootstrap/terminal/GAE có giá trị raw đúng, gradient đúng công thức,
  evaluation predictions round-trip và control `none` tương thích legacy.
- Field log phải phân biệt `value_loss_normalized`, raw prediction MSE và EV;
  không gọi MSE nhỏ hơn do chia scale là critic tốt hơn.

### Ma trận chạy mới

| ID dự kiến | Parent | Khác biệt duy nhất |
|---|---|---|
| D30_e07_control_b100k_s17 | E07 | Instrumentation + budget-end validation, baseline mới |
| D31_e07_separate_b100k_s17 | D30 | Shared → separate encoder |
| D32_e07_vscale_b100k_s17 | D30 | Fixed value target scale |
| D33_e07_separate_vscale_b100k_s17 | D31 | Thêm scale; chỉ khi hai nhánh đáng kiểm tra tương tác |

Pilot train 10.240 step dùng ID `*_pilot_b10240_s17` và output khác. Nó đánh giá
mechanics, gradient/activation trend, không dùng để tuyên bố cải thiện QoS.
Screening 100k chạy từ đầu; không lấy pilot resume rồi gọi là matched fresh run.
Không tự chạy tất cả nhánh chỉ vì chúng được liệt kê.

## 8. D4 — tách ảnh hưởng reward và observation

### D4-A: matched-scale rate control

So E07 với một reward mới `train_fixed_rate_reward_v1`:

```text
E07:      u_j,t = 60 * dt * R_j,t / D*_j
R_scale:  u_j,t = dt * R_j,t / R_ref
R_ref = mean_TRAIN(D*_j) / 60
```

Cách chọn R_ref làm tổng oracle utility trung bình TRAIN bằng 60; khi các route
đều 60 giây, R_ref bằng mean oracle rate của các route TRAIN. Nó không làm
utility từng route bằng 60. Ghi rõ đơn vị/công thức numerical utility.

- RLF=5, PP=1 như E07; không đổi event weights cùng lúc.
- Fit R_ref từ TRAIN bằng cùng bandwidth/rate formula, cache kèm manifest hash.
- Giữ architecture/optimizer đã khóa cho cả hai arm và cùng budget.
- Schema mới + checkpoint serialization; test identity khi D* các route bằng
  nhau, và oracle mean matched scale khi D* khác nhau.
- Chỉ sau đối chứng này mới quy gain cho trajectory weighting thay vì tăng
  rate scale. Không khẳng định mọi normalization đều làm actor biết future.

### D4-B: observation gọn vs đầy đủ

So `rolling_v1` (39 chiều, E04) với `qos_full_v1` (62 chiều, E06), cùng E07
reward hoặc reward cố định đã chọn, cùng architecture và hyperparameter.
Hai run khởi tạo mới. Đây là ablation observation block, không chứng minh riêng
normalization hay một feature gây gain.

Nếu cần làm paper factorial, hoàn thành 2x2 observation×reward với cùng nền
optimizer. Bảng E11 cũ dùng E09 không tự động được đổi thành E07 rồi coi như
cùng experiment. Đặt ID mới và ghi sửa preregistration từ bằng chứng P3.

Leave-one-out projected SINR, speed, long history chỉ mở sau khi full/gọn có
tín hiệu ổn định. Không làm một vòng bỏ nhiều feature tùy cảm giác.

### D4-C: sensitivity event weights nếu counterfactual cho thấy cần

Chọn một hệ số mỗi lần, ví dụ 0,5x/1x/2x theo parent đã khóa; mọi term khác giữ
nguyên. Đánh giá chênh reward giữa action, physical metrics và tail QoS.
Không lựa chọn hệ số chỉ bằng độ lớn tổng rate reward so với tổng penalty.

## 9. D5 — sửa target của dual, chỉ mở sau fixed reward ổn định

E10 v1 phải giữ nguyên để replay. Tạo schema mới, ví dụ
`trajectory_qos_dual_3gpp_reward_v2`, với hợp đồng target rõ ràng.

### Target định nghĩa chính xác

```text
rho_j = D_policy,j / D_oracle,j
mu_ref,s = mean_{j thuộc TRAIN speed s}(rho_3gpp,j)
alpha_s = 0.98 * mu_ref,s
shortfall_s = alpha_s - mean_completed_train(rho_policy,s)
lambda_s <- clip(lambda_s + eta * shortfall_s, 0, lambda_max)
```

- Tính/cache mu_ref trên TRAIN cùng route manifest và protocol; không lấy các
  số baseline validation 91,27% trong bảng để đặt training target.
- `mean(rho_policy)/mean(rho_ref)` khác `mean(rho_policy/rho_ref)`; chọn công
  thức trên để khớp kiểu gate hiện tại theo speed, không đổi ngầm.
- Dual này chỉ điều khiển relative rate theo speed. Mean Mbps guard và aggregate
  validation gate vẫn đánh giá riêng; không gọi nó là đảm bảo mọi constraint.
- Thiếu speed/reference/hash không khớp phải fail rõ, không fallback sang 0,98 oracle.
- Giữ lambda frozen trong rollout/update; chỉ training completed episodes cập
  nhật. Không dùng validation metrics hoặc deterministic validation để update.
- Log target/reference/observed/shortfall/sample_count/lambda theo speed và
  distribution stochastic vs deterministic riêng.
- Lambda chạm trần với shortfall chỉ là dấu hiệu saturation/constraint chưa đạt
  trong run, không phải chứng minh toán học bài toán bất khả thi.
- Episode qua nhiều rollout có thể nhận nhiều lambda: ghi rõ khi báo return;
  không so learning curve return động để kết luận performance tăng.
- Không tự thêm hằng phạt target mỗi macro-step: số decision phụ thuộc action,
  làm thay mục tiêu. Mọi thiết kế Lagrangian mới cần kiểm tra discount/duration.

Test ví dụ baseline TRAIN rho=0,90 → target=0,882; policy rho=0,90 không tăng
lambda, rho=0,85 tăng; lambda min/max, sample threshold, speed thiếu và
checkpoint round-trip. Reward v1 vẫn load cho ra kết quả cũ.

Không bật dual và đổi kiến trúc/value normalization trong cùng một arm so với
E09 rồi quy tất cả khác biệt cho sửa target.

## 10. D6 — các nhánh chỉ triển khai khi diagnostic chỉ đúng nguyên nhân

### Candidate scorer dùng chung

Nếu candidate tốt thắng reward và learnability probe phân biệt được nhưng flat
MLP PPO vẫn không học, thử scorer `score(context, features_candidate)` dùng
chung weights cho C1/C2/C3, head STAY riêng; ghép 4 logits và áp mask.
Vẫn là feed-forward MLP PPO. Version architecture mới, parameter-count control,
test hoán vị candidate block+mask thì HO logits hoán vị đúng. Không bỏ role
features hoặc đổi candidate ranking đồng thời.

### Reconsider STAY trong cùng A3 event

Nếu D2 cho thấy reject-event mất QoS, thử semantics mới: A3 completed-TTT vẫn
gate đầu; STAY cho quyết định lại sau 100 ms nếu event còn hợp lệ. Trong thời
gian chờ, engine phải tiếp tục HO/RLM/RLF/tick bookkeeping bình thường.
Không thêm action ngoài A3 theo mặc định.

Version stay mode, xử lý event leave/target-change/RLF/route end; kiểm thử không
double-count TTT/event/reward và không sinh decision mỗi primitive tick.
Đây là thay action opportunity, không phải cải thiện chỉ từ reward/observation.

Entropy schedule, mạng 128x128, gamma=1 và suitability mask là các nhánh độc lập
khác. Không mở chỉ để tăng số experiment hoặc làm tỷ lệ C2/C3 lớn hơn.

## 11. Promotion sang nhiều seed và 300k

- Mọi run mới báo selected, last scheduled và budget-end; giữ checkpoint
  selection theo QoS → RLF-time → HOF → PP → HO attempts → rate.
- Gate: aggregate và từng speed giữ >=98% relative rate baseline và >=98%
  mean Mbps baseline. Báo tail/P10, không tự thêm ngưỡng tail sau khi thấy số.
- Ưu tiên cấu hình có gate pass ở ít nhất 3 validation liên tiếp, gồm budget-end,
  và cải thiện reliability có hướng nhất quán. Đây là tiêu chí promotion mới
  cho run sau kế hoạch, không sửa nhãn historical E07.
- Nếu cuối 100k còn tiến bộ rõ nhưng chưa pass ổn định, có thể đưa vào 300k
  như arm thăm dò có ghi lý do; không gọi là đã xác nhận.
- Chọn tối đa 2 treatment có bằng chứng + control phù hợp. Chạy seed17/29/43,
  split17 từ đầu; exact resume hiện chưa được hỗ trợ đầy đủ.
- Giữ control nguyên bản để so practical gain; nếu thay nền optimizer cho mọi
  treatment, cũng cần control cùng nền để tách các đóng góp.
- Báo variation theo seed và paired interval theo route, không coi ticks là
  mẫu độc lập. Test/zero-shot chỉ khi thiết kế và selection đã khóa.
- Test lịch sử đã được xem trước đây, không mô tả nó là blind holdout mới.
- Rate retention 98% không có nghĩa vượt 3GPP; claim vượt rate cần gain dương
  với uncertainty thích hợp.

## 12. Thư mục, CLI và xác minh

Output dự kiến, luôn tạo nhánh/thư mục mới nếu đã tồn tại:

```text
results/analysis/p23_diagnostics/
  registry_v1/
  state_bank_pilot_v1/
  state_bank_train_v1/
  probe_v1/
  counterfactual_v1/
results/models/ablation_v3/<new_experiment_id>/
docs/research/p23_diagnostic_report_vi.md
```

CLI nền train HIỆN CÓ (chỉ là template, cần thêm experiment ID/output riêng;
không copy chạy ngay khi đang triển khai diagnostic):

```powershell
.\.venv\Scripts\python.exe run.py train_ppo `
  --data-dir data/uav_100m_grid100m_dev_processed `
  --speeds 30 50 70 90 `
  --candidate-count 3 --candidate-order ranked_v1 `
  --observation-schema qos_full_v1 `
  --reward-schema trajectory_qos_reward_v1 `
  --total-macro-steps 100000 `
  --rollout-steps 1024 --batch-size 128 --update-epochs 4 `
  --learning-rate 0.00005 --evaluation-interval 10240 `
  --seed 17 --split-seed 17 --skip-final-test
```

Khi thêm flag mới, smoke bằng budget=2048, rollout=1024, eval interval=2048,
ID/output smoke riêng. Không tự thay split để có validation nhỏ mà không ghi
rõ trong metadata; unit tests dùng synthetic traces để chạy nhanh.

Kiểm thử theo phần vừa sửa, dùng unittest của repository:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_policy_diagnostics.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_smdp_ppo.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_candidate_audit.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_reward.py"
git diff --check
```

`test_policy_diagnostics.py` là file mới cần tạo ở D1. Chạy suite liên quan khi
thay đổi, full suite một lần trước bàn giao thay đổi xuyên trainer/checkpoint;
không chạy full suite lặp lại khi không có thay đổi mới.
Nếu sandbox không khởi động được `.venv\Scripts\python.exe`, dùng cơ chế
escalation của môi trường; không sửa venv hoặc cài lại PyTorch tùy tiện.

Kế hoạch này không tự tạo runner hoặc heartbeat. Nếu người dùng yêu cầu chạy
screening, runner tuần tự phải xác minh config/report/checkpoint, ghi stdout và
stderr riêng, không resume/ghi đè output dang dở. Background Start-Process dùng
`-WindowStyle Hidden`. Chỉ tạo theo dõi định kỳ khi có yêu cầu phù hợp.

## 13. Checklist bàn giao từng gói

- [x] D0: registry đúng selected/scheduled/budget-end; docs không còn bảo chạy lại P3.
- [x] D1: bank TRAIN tái lập, schema/hash đầy đủ, activation/probability/gradient diagnostics.
- [x] D1: bật diagnostic không thay update/RNG/collector; checkpoint cũ vẫn load.
- [x] D2: cùng state/cùng physical horizon; reward/physical outcome đúng và causal inputs.
- [x] D2: report chỉ rõ evidence, giả thuyết chưa kiểm chứng, quyết định nhánh kế tiếp.
- [x] D3 implementation: control matched, architecture/scale serialized,
  meaningful tests pass; pilot 10.240 chưa chạy.
- [ ] D4 nếu được mở: factor riêng, train-fitted normalization, không dùng test tuning.
- [ ] D5 nếu được mở: target TRAIN 3GPP có version mới; v1 không đổi nghĩa.
- [ ] Không đổi artifact cũ; mọi run mới có source/data/config provenance.
- [x] Bàn giao lệnh thực sự tồn tại, test đã chạy, artifact path và giới hạn kết luận.

Trạng thái 05/10/2026: D0–D2 và pilot diagnostic đã hoàn tất; D3-A/D3-B đã
được triển khai, kiểm thử và các pilot D30/D32 đã được phân tích. Gate sau đó
mở D6 candidate scorer; pilot D60 10.240 đã hoàn tất nhưng không qua gate QoS
do policy deterministic vẫn STAY 100%. Chưa mở D4/D5 và chưa chạy D60 100k.
Xem [báo cáo diagnostic](p23_diagnostic_report_vi.md),
[báo cáo triển khai D3](d3_optimizer_ablation_implementation_report_vi.md) và
[báo cáo D6](d6_candidate_scorer_pilot_report_vi.md).

## 14. Prompt ngắn để giao Sol gói đầu

> Đọc `docs/research/p2_p3_diagnostic_implementation_plan_vi.md` và triển khai
> D0–D2 trong project hiện tại. Ưu tiên tái sử dụng candidate/observation audit,
> tạo state bank TRAIN có provenance, đo saturation và gradient actor/critic,
> rồi đối chiếu physical outcome/reward/action probability trên cùng state.
> Làm unit tests, pilot audit và báo cáo tái lập. Giữ checkpoint cũ tương thích,
> không sửa kết quả cũ, không dùng test/zero-shot, không tự khởi động screening
> 100k/300k trong gói này. Kết thúc bằng bằng chứng và đề xuất cụ thể mở D3,
> D4 hay nhánh khác theo gate trong kế hoạch; chưa triển khai mọi nhánh cùng lúc.

## 15. Tài liệu liên quan

- [Kế hoạch ablation gốc](ppo_ablation_training_plan_vi.md)
- [P1: candidate audit và screening](p1_candidate_ablation_report_vi.md)
- [P2: thiết kế observation](p2_observation_ablation_report_vi.md)
- [P3: thiết kế reward](p3_reward_ablation_report_vi.md)
- [PPO implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/):
  shared/separate networks là một biến thể triển khai cần thử nghiệm, không là
  đảm bảo separate sẽ thắng trên dataset này.
- [Learning values across many orders of magnitude](https://arxiv.org/abs/1602.07714):
  nền tảng về value-target scaling; D3-B chỉ là fixed-scale control, không gọi
  nó là triển khai PopArt.
