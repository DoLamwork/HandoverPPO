# D62 — Actor phân cấp timing/candidate

## Mục tiêu và đối chứng

D62 kiểm tra duy nhất giả thuyết kiến trúc actor. Đối chứng trực tiếp là D61.
Hai arm dùng cùng TRAIN/validation split, seed 17, K=3, ranking `ranked_v1`,
observation `qos_full_v1`, reward `trajectory_qos_reward_v1`, redecision 100 ms,
optimizer và budget 10.240 macro-step. Không chạy test/zero-shot.

## Kiến trúc

Actor vẫn xuất phân phối bốn action cho PPO và environment:

- head timing: `P(STAY)` và `P(HANDOVER)`;
- candidate scorer dùng chung: `P(C1/C2/C3 | HANDOVER)`;
- `P(HO_Ci) = P(HANDOVER) * P(Ci | HANDOVER)`.

Mask candidate được áp dụng trước softmax có điều kiện. Nếu không còn candidate
hợp lệ thì HANDOVER bị mask ở head timing. Bias timing khởi tạo để xác suất
bốn action đều bằng 0,25 khi cả ba candidate hợp lệ; nhờ đó D61 và D62 không
khác nhau vì prior ban đầu.

Với policy loss của action STAY, candidate head chỉ nhận sai số số học dưới
`1e-6`; action HO truyền gradient vào cả timing và candidate head. Entropy của
phân phối joint vẫn có thể cập nhật candidate head vì đây là regularizer chung,
đúng với giả thuyết cần đo thay vì giả định tách head loại bỏ mọi ảnh hưởng.

## Tương thích

Identifier mới là `hierarchical_candidate_scorer_v1`. Checkpoint cũ và ba
kiến trúc trước vẫn giữ nguyên đường load. Checkpoint lưu rõ architecture và
không được load chéo kiến trúc.

## Tái lập

```powershell
.\scripts\run_d62_hierarchical_pilot.ps1
```

Runner từ chối ghi đè output dở dang và chỉ tái sử dụng pilot khi report đúng
experiment, budget, architecture, redecision contract và checkpoint tồn tại.

## Kết quả

Pilot hoàn tất đủ 10.240 macro-step/10 PPO update, seed/split seed 17. Không
đọc test hoặc zero-shot. Full suite sau triển khai: **194 test pass**.

### Validation trực tiếp D61–D62

| Model | Return | Rate Mbps | Relative rate | RLF-time | HOF | PP | HO | STAY |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D61 flat scorer | 31,614 | 22,440 | 88,248% | 0,387% | 4,069% | 13,237% | 15,410% | 84,590% |
| D62 hierarchical | 17,892 | 20,448 | 78,769% | 1,308% | 1,887% | 10,470% | 1,968% | 98,032% |
| 3GPP Always-C1 | -18,978 | 23,024 | 91,272% | 1,313% | 10,903% | 37,057% | 100% | 0% |

D62 giảm HOF và PP nhưng bằng cách gần như không handover. Rate giảm 1,992 Mbps,
relative rate giảm 9,479 điểm phần trăm và RLF-time tăng 0,921 điểm phần trăm so
với D61. QoS aggregate chỉ đạt 88,81% theo rate và 86,30% theo relative rate,
nên checkpoint là `QOS_INFEASIBLE`.

| Tốc độ | D61 rate | D62 rate | D61 relative rate | D62 relative rate |
|---:|---:|---:|---:|---:|
| 30 km/h | 22,254 | 20,562 | 89,674% | 81,044% |
| 50 km/h | 23,360 | 21,615 | 88,379% | 79,977% |
| 70 km/h | 21,846 | 19,341 | 86,858% | 76,042% |
| 90 km/h | 22,111 | 19,464 | 85,644% | 73,516% |

D62 kém D61 ở cả bốn tốc độ; khoảng hụt lớn nhất ở 90 km/h. Deterministic
validation của cả hai vẫn chỉ chọn C1 khi handover: D61 có 2.091 HO_C1, D62 có
483 HO_C1; C2/C3 đều bằng 0.

### Fixed TRAIN state bank

Trên cùng 3.117 state:

| Model | P(STAY) | P(C1) | P(C2) | P(C3) | Argmax STAY | Argmax C1 |
|---|---:|---:|---:|---:|---:|---:|
| D61 | 27,073% | 26,431% | 23,595% | 22,900% | 2.802 | 315 |
| D62 | 27,837% | 26,570% | 23,205% | 22,387% | 3.050 | 67 |

Một thay đổi xác suất STAY chỉ +0,764 điểm phần trăm đã làm số state argmax
STAY tăng từ 89,9% lên 97,9%, vì policy vẫn gần đồng đều và top-1/top-2 margin
rất nhỏ. Entropy chuẩn hóa D62 còn 0,9969; đây không phải collapse do thiếu
exploration.

### Same-transition signal

Ở update cuối của D62, advantage trung bình và thay đổi xác suất joint là:

| Action | Advantage chuẩn hóa | Delta xác suất sau update |
|---|---:|---:|
| STAY | +0,0799 | +0,005325 |
| C1 | +0,1524 | +0,002076 |
| C2 | -0,0424 | -0,002992 |
| C3 | -0,2235 | -0,004409 |

Audit counterfactual horizon 1 giây tìm 51 positive state cho D62. Tại state
C2 hữu ích, xác suất joint vẫn giảm 0,000665 và xác suất conditional
`P(C2|HO)` giảm 0,000532. Với C3 hữu ích, hai mức giảm lần lượt là 0,001829
và 0,002180. Xác suất tổng HANDOVER cũng giảm khoảng 0,00084 sau cùng update.

Gradient actor ở update cuối đi vào cả timing head (norm 0,0996) và candidate
head (0,2325), nên không có lỗi gradient bị ngắt. Vấn đề là tín hiệu tối ưu:
STAY/C1 nhận advantage tốt hơn trên phần lớn rollout, trong khi useful C2/C3
quá hiếm để đảo chiều conditional scorer.

## Kết luận gate

D62 **không qua gate** và không nên promote lên 100k/300k. Hai head hoạt động
đúng kỹ thuật nhưng không tự sửa imbalance dữ liệu/credit assignment; còn làm
deterministic timing nghiêng mạnh hơn về STAY.

Bước tiếp theo có giá trị nhất không phải tăng entropy hay train dài ngay. Cần
giữ D61 làm control và thử một pilot TRAIN-only có tín hiệu candidate trực tiếp
(auxiliary counterfactual ranking/classification trên positive state), với hệ số
nhỏ và không dùng future feature trong observation/inference. Chỉ promote nếu
`P(C2/C3|HO)` tăng đúng trên useful state mà QoS/HOF/RLF không xấu đi. Nếu
auxiliary probe không thay đổi được conditional scorer, quay lại gate D4 để bổ
sung observation phân biệt candidate thay vì tiếp tục sửa PPO timing.

## Artefact

- Checkpoint/report: `results/models/d6_pilots/D62_hierarchical_pilot_b10240_s17/`
- Fixed-bank probe: `results/analysis/p23_diagnostics/d62_pilot_v1/fixed_bank_probe/`
- Transition audit: `results/analysis/p23_diagnostics/d62_pilot_v1/transition_signal/`
- Runner: `scripts/run_d62_hierarchical_pilot.ps1`

Audit giải mã action sau pilot cho thấy joint argmax và hierarchical argmax
bất đồng ở 97,85% TRAIN state. Hierarchical argmax gần như tái tạo Always-C1,
còn sampling làm HOF/RLF tăng mạnh. Xem
[báo cáo action-selection audit](d62_action_selection_audit_report_vi.md).
