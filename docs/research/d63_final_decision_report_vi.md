# D63 — báo cáo kết thúc và quyết định nghiên cứu

Ngày chốt: 07/10/2026. Trạng thái: **hoàn tất gói diagnostic D63; G1 PASS,
G2 FAIL, D63-3 đóng nhánh hiện tại; không mở train PPO mới**.

## 1. Câu hỏi D63 cần trả lời

D63 không nhằm tạo thêm một checkpoint PPO. Gói này tách ba câu hỏi:

1. Trong phạm vi A3 đã hoàn tất TTT, timing và target tốt hơn Always-C1 có tồn tại không?
2. Observation nhân quả hiện tại có dự đoán được cơ hội đó đủ an toàn không?
3. Feature `execution_safety_margin_v1_db` rút ra từ failure replay có khắc
   phục được false positive trên dữ liệu chưa dùng để hình thành giả thuyết không?

Toàn bộ quá trình chỉ dùng TRAIN. Không dùng validation, test hoặc zero-shot để
tạo nhãn, chọn ngưỡng hay ra quyết định D63.

## 2. D63-0 và D63-1 — dư địa vật lý

Main bank gồm 80 TRAIN route, 20 route/tốc độ, 1.199 anchor từ Always-C1, D61
và D62. Bank thực tế dùng tối đa 5 anchor/source/route, nhỏ hơn quota tối đa 10
trong kế hoạch nhưng không lặp state để lấp quota. Full-route có đủ 320 hàng.

Finite-lookahead reference horizon 1 giây đạt G1 PASS:

| Policy | RR (%) | HOF (%) | RLF-time (%) | PP (%) |
|---|---:|---:|---:|---:|
| Always-C1 | 89,868 | 11,800 | 1,576 | 36,065 |
| D61 frozen | 85,639 | 4,882 | 0,555 | 13,729 |
| Timing-only reference | 91,294 | 3,965 | 0,321 | 19,464 |
| Timing×target reference | 91,606 | 4,431 | 0,338 | 19,405 |

So với Always-C1, timing×target tăng RR 1,738 điểm phần trăm, bootstrap CI95%
[1,569; 1,907], đồng thời các point estimate reliability đều tốt hơn. Kết luận
được phép: trong TRAIN có dư địa vật lý. Reference dùng future nên không phải
policy online hoặc kết quả PPO.

## 3. D63-2 — khả năng học từ observation hiện tại

Bộ nhãn leakage-safe có 1.199 state và 14.388 outcome timing×target. Split theo
route, stratify speed, seed 29: 48 route fit, 16 tuning, 16 holdout. Target-only
có 77 positive toàn bank nhưng chỉ 14 positive trên holdout.

Linear và MLP-32 có ranking signal, mạnh nhất là target MLP:

- prevalence holdout 1,94%;
- AP 0,182, khoảng 9,4 lần prevalence;
- nhưng threshold an toàn chọn 0 alternative;
- mọi family timing/target/joint có net gain triển khai bằng 0.

G2 FAIL: predictor chưa tạo được net physical gain dương mà giữ reliability.
Phân tích threshold 0,9 tìm thấy hai false positive nghiêm trọng, đều dự đoán
sai dấu delivered data và tạo HOF → RLF recovery.

## 4. Failure replay và giả thuyết feature

Replay SINR/RSRP/RLM trước–trong–sau HO đề xuất feature nhân quả:

```text
execution_safety_margin_v1_db
  = target SINR RLM gap to Qin tại decision
    - độ lệch chuẩn target SINR RLM trong 500 ms quá khứ
```

Hai false positive holdout có feature -7,734 và -7,950 dB; hai true-positive
tham chiếu có 2,813 và -1,263 dB. Bốn case chỉ tạo giả thuyết, không được dùng
để khóa threshold hoặc tuyên bố tổng quát hóa.

## 5. D63-3 — kiểm chứng feature trên fit/tuning

D63-3 replay lại duy nhất fold TRAIN fit và tuning; holdout D63-2 không được đọc.
Ngưỡng được khóa trên fit bằng mục tiêu tăng delivered Mbit với ràng buộc mọi
mean risk delta không dương, sau đó mới áp dụng lên tuning.

| Fold | Alternative rows | Positive | Prevalence | AP | Lift |
|---|---:|---:|---:|---:|---:|
| Fit | 1.438 | 46 | 3,199% | 0,168 | 5,25× |
| Tuning | 480 | 17 | 3,542% | 0,209 | 5,90× |

Feature có ranking signal nhưng không đạt action-selection gate. Trên fit:

- ngưỡng tăng data tốt nhất nếu bỏ reliability chọn 29 alternative;
- mean gain +0,02128 Mbit/state;
- 10 lựa chọn đúng và 19 false alternative;
- ping-pong tăng +0,00974/state;
- interruption tăng +0,00111 giây/state;
- không có bất kỳ ngưỡng non-control nào vừa có lợi vừa giữ mọi mean risk
  không xấu.

Vì vậy selector an toàn khóa `CONTROL_ONLY`. Khi áp lên tuning:

- action đổi: 0/240 state;
- mean gain: 0 Mbit/state, route-bootstrap CI95% [0; 0];
- số nhóm tốc độ có gain dương: 0/4;
- holdout rows đọc trong D63-3: 0.

Gate D63-3:
`CLOSE_D63_NO_GENERALIZING_FEATURE_EVIDENCE`.

## 6. Quyết định cuối

1. **Không mở D4 chỉ với `execution_safety_margin_v1_db`.** Feature xếp hạng
   positive tốt hơn random nhưng không đủ chính xác ở vùng quyết định an toàn.
2. **Không chạy PPO 10.240/100k/300k mới từ D63.** Điều kiện mở pilot trong kế
   hoạch không đạt; train thêm sẽ biến diagnostic âm thành sweep phỏng đoán.
3. **Đóng vòng D63 hiện tại.** Kết luận phù hợp bằng chứng là: timing×target có
   dư địa khi dùng finite future, nhưng observation/feature nhân quả hiện tại
   chưa chuyển dư địa đó thành policy online giữ QoS và reliability.
4. Nếu nghiên cứu tiếp, đó phải là phạm vi mới được phê duyệt riêng: thêm measurement
   thật sự có sẵn online trong cửa sổ HO execution, hoặc mở quyết định A3-entry/
   pre-TTT. Không được trình bày như phần tiếp tục tự động của D63.

## 7. Artifact và tái lập

- D63-1 main: `results/analysis/d63_timing_target_main_analysis_v1/`
- D63-1 verification: `results/analysis/d63_1_verification_v1/`
- D63-2 labels: `results/analysis/d63_2_labels_v1/`
- D63-2 probe: `results/analysis/d63_2_probe_v1/`
- False-positive audit: `results/analysis/d63_2_target_mlp_fp_v1/`
- Failure replay: `results/analysis/d63_2_failure_case_replay_v1/`
- D63-3 feature gate: `results/analysis/d63_3_execution_safety_v1/`

Lệnh tái lập bước cuối:

```powershell
.\scripts\validate_d63_3_feature.ps1
```

Script và command từ chối ghi đè output không rỗng. Source artifact chứa SHA-256
đầu vào; gói D63-1 đã kiểm tra công thức RR/HOF/RLF-time/PP và provenance.

## 8. Giới hạn

- G1 là TRAIN finite-lookahead reference, không phải upper bound toàn cục.
- Holdout D63-2 đã được mở một lần cho probe v1 và chỉ dùng hậu nghiệm để tạo
  giả thuyết; D63-3 cố ý không đọc lại holdout.
- D63-3 kiểm tra đúng một feature đã đăng ký, không chứng minh mọi observation
  tương lai đều bất khả thi.
- Không có claim PPO vượt 3GPP từ D63.
