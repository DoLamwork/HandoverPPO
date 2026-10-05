# Gate chẩn đoán khả năng học candidate (TRAIN-only)

## Mục tiêu

Gói này trả lời bốn câu hỏi còn thiếu sau D2/D3:

1. Kết luận trước đó có bị phụ thuộc vào state bank quá nhỏ không?
2. C2/C3 nhận advantage sai dấu, hay tín hiệu đúng nhưng quá hiếm?
3. Observation `qos_full_v1` có đủ thông tin để phân biệt candidate hữu ích trên route chưa thấy khi fit không?
4. Nên mở D6 candidate scorer hay quay lại D4 observation?

Toàn bộ dữ liệu dùng cho chẩn đoán thuộc split TRAIN. Không đọc test/zero-shot; validation chỉ được chương trình train dùng để hoàn tất contract của probe một-update D40, không dùng để fit hay quyết định nhãn supervised.

## 1. State bank mở rộng

- 80 route TRAIN: 20 route/tốc độ ở 30, 50, 70 và 90 km/h.
- Tối đa 20 decision point/route.
- 3.117 state: 1.560 state từ Always-HO-C1 và 1.557 state từ policy D30-selected.
- Phân bố theo tốc độ: 739 / 778 / 800 / 800 state.
- Candidate cố định K=3, `ranked_v1`; observation `qos_full_v1`.
- Replay kiểm tra lại tic, serving cell, candidate identity, mask và observation tại từng anchor.

Counterfactual horizon 1 giây tìm được:

- 207/3.117 state (6,64%) có C2/C3 tốt hơn C1 về physical outcome;
- 167/3.117 state (5,36%) có C2/C3 tốt hơn cả lựa chọn tốt nhất giữa STAY và C1;
- reward E07 nhận ra ít nhất một alternative hữu ích ở 94,01% state dương;
- 92,71% các action alternative hữu ích có reward E07 lớn hơn baseline STAY/C1.

Kết luận: nhãn alternative hữu ích không biến mất khi mở rộng bank, nhưng là sự kiện hiếm khoảng 1/19 state. Reward E07 nhìn chung cho đúng dấu ở các state này; lỗi chính không phải reward hoàn toàn đảo ngược mục tiêu.

## 2. Advantage và thay đổi xác suất theo action

D40 là probe đúng một rollout/update (1.024 macro-transition), khởi tạo đều 25% cho bốn action. Nó không phải model benchmark.

| Action | Lần được chọn | Mean normalized advantage | Tỷ lệ raw advantage dương | P trước | P sau | ΔP |
|---|---:|---:|---:|---:|---:|---:|
| STAY | 244 | +0,249 | 35,25% | 25,00% | 25,81% | +0,81 điểm % |
| C1 | 250 | +0,090 | 26,80% | 25,00% | 25,49% | +0,49 điểm % |
| C2 | 267 | -0,177 | 15,36% | 25,00% | 24,35% | -0,65 điểm % |
| C3 | 263 | -0,137 | 15,21% | 25,00% | 24,35% | -0,65 điểm % |

Ngay update đầu, PPO đã nhận tín hiệu tổng hợp rằng C2/C3 kém hơn và giảm xác suất của cả hai trên 100% state trong rollout. Điều này không mâu thuẫn với counterfactual: C2/C3 chỉ hữu ích ở 5,36% state, trong khi policy ngẫu nhiên thử chúng ở hơn 51% transition. Phần lớn mẫu C2/C3 là thử sai không đúng ngữ cảnh, nên gradient trung bình lấn át số ít mẫu tốt.

Do rollout chưa gắn nhãn counterfactual trực tiếp, số liệu này không khẳng định mọi action hữu ích đều có advantage dương. Kết luận được phép là: reward có dấu đúng trên phần lớn state hữu ích, nhưng tín hiệu on-policy tổng hợp cho C2/C3 âm vì positive state quá hiếm và action head dùng chung không khai thác tốt cấu trúc candidate.

## 3. Supervised probe chia theo route

Linear probe chỉ fit trên route TRAIN-fit và đánh giá trên route TRAIN-holdout (seed 29). Mọi timestamp và nguồn policy của cùng route nằm cùng fold. Holdout có 933 state.

| Nhãn | Prevalence holdout | Average precision | Lift so với random | Recall@0,5 |
|---|---:|---:|---:|---:|
| Có C2/C3 hữu ích | 4,18% | 0,204 | 4,88× | 59,0% |
| C2 hữu ích | 3,86% | 0,131 | 3,39× | 52,8% |
| C3 hữu ích | 1,18% | 0,217 | 18,4× | 45,5% |

Probe reward-best multiclass cũng vượt prevalence cho C2/C3, nhưng còn yếu: AP C2=0,143 với prevalence 0,0697; AP C3=0,0426 với prevalence 0,0161. Precision ở threshold 0,5 thấp vì mất cân bằng lớp mạnh.

Kết luận: observation hiện tại **có thông tin phân biệt có thể học được trên route chưa dùng để fit**, nên gate “observation hoàn toàn không phân biệt được” không xảy ra. Tuy nhiên tín hiệu yếu và hiếm; flat MLP phải tự học tính đối xứng giữa ba block candidate và đồng thời thắng gradient âm áp đảo.

## 4. Quyết định gate

Mở **D6 candidate scorer** trước, giữ D4 observation như đối chứng sau, không mở cả hai cùng lúc.

Lý do:

- reward E07 đã nhận đúng alternative hữu ích ở 94% positive state;
- observation tuyến tính có lift 3,4×–18,4× trên route holdout;
- PPO vẫn có argmax recall C2/C3 bằng 0 ở D30/D32;
- update đầu chủ động hạ P(C2/C3), cho thấy vấn đề là độ hiếm và inductive bias/credit assignment hơn là thiếu tuyệt đối thông tin đầu vào.

D6 tối thiểu nên dùng cùng một scorer chia sẻ trọng số cho từng candidate block, ghép context chung với feature candidate, sinh ba logit HO theo cùng hàm và một logit STAY riêng. Giữ nguyên K=3, observation, reward, optimizer, seed và budget để chỉ thay một yếu tố kiến trúc. Trước screening 100k, chạy unit test hoán vị candidate và pilot 10.240 step; gate pilot yêu cầu C2/C3 không bị triệt tiêu ngay, entropy còn đủ khám phá, và xác suất useful alternative trên bank tăng mà QoS không sụp.

## Artefact tái lập

- `results/analysis/p23_diagnostics/state_bank_train_v2/`
- `results/analysis/p23_diagnostics/counterfactual_train_v2/`
- `results/analysis/p23_diagnostics/supervised_probe_train_v2/`
- `results/models/d4_diagnostics/D40_action_signal_probe_b1024_s17/`
- `scripts/run_candidate_learning_gate.ps1`

## 5. Kết quả sau khi mở D6

D6 `candidate_scorer_v1` đã được triển khai và pilot 10.240 macro-step. Scorer
tăng tổng xác suất dành cho alternative hữu ích trên state bank từ 18,23% lên
25,49%, nhưng argmax recall vẫn 0%. D30 và D60 đều cho validation STAY 100%,
relative rate 73,99% và `QOS_INFEASIBLE`.

Same-transition audit cho thấy useful C2/C3 của D60 có normalized advantage
trung bình xấp xỉ dương, nhưng PPO update vẫn giảm xác suất của chúng. Vì vậy
gate chỉ đạt phần inductive-bias, chưa đạt hành vi/QoS; **không chạy D60 100k**.
Chi tiết, giới hạn kết luận và đề xuất D61 A3-gated redecision nằm trong
[báo cáo D6](d6_candidate_scorer_pilot_report_vi.md).
