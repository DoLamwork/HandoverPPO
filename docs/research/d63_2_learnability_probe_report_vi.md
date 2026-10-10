# D63-2 — probe khả năng học timing và target

Ngày chạy: 06/10/2026. Trạng thái: **đã chạy linear và MLP nhỏ trên bộ nhãn
TRAIN; G2 hiện không đạt theo action-selection gate đã khóa**.

## Hợp đồng thí nghiệm

- Split route đã khóa từ bộ nhãn: fit/tuning/holdout = 60/20/20, seed 29,
  stratify theo speed. Tất cả source và anchor của một route ở cùng fold.
- Fit chỉ dùng fold `fit`; threshold chọn bằng `tuning`; `holdout` chỉ được mở
  sau khi fit và threshold đã cố định trong cùng lượt chạy.
- Feature duy nhất: 62 chiều observation hiện tại, planned delay đã chuẩn hóa
  và one-hot planned slot. Không dùng source, speed, route ID, future outcome
  hoặc physical target tương lai làm input.
- Hai cấu hình cố định, không sweep: linear multitask và MLP một hidden layer
  32 neuron. Cả hai dự đoán `strict_useful`, delta delivered Mbit và sáu delta risk.
- Khi áp dụng predictor, mỗi state chỉ chọn tối đa một nhánh; nếu không đủ
  threshold thì dùng immediate-C1 control. Threshold tuning tối đa hóa gain với
  ràng buộc mean của mọi risk delta không dương.

## Kết quả holdout

| Family | Model | Positive prevalence | AP | Brier | Threshold | Action đổi | Mean gain |
|---|---|---:|---:|---:|---:|---:|---:|
| Timing | Linear | 5,00% | 0,084 | 0,192 | >1 | 0 | 0 Mbit |
| Timing | MLP-32 | 5,00% | 0,088 | 0,080 | >1 | 0 | 0 Mbit |
| Target | Linear | 1,94% | 0,120 | 0,150 | >1 | 0 | 0 Mbit |
| Target | MLP-32 | 1,94% | **0,182** | **0,029** | >1 | 0 | 0 Mbit |
| Joint | Linear | 4,06% | 0,085 | 0,211 | >1 | 0 | 0 Mbit |
| Joint | MLP-32 | 4,06% | 0,064 | 0,079 | >1 | 0 | 0 Mbit |

`>1` là threshold sentinel chỉ chọn control. Vì không action nào được thay đổi,
route-bootstrap CI của mean gain là [0, 0]. Đây là kết quả của gate an toàn,
không phải bằng chứng model dự đoán mọi xác suất bằng nhau.

Tín hiệu ranking có thật nhưng yếu và không đủ để ra quyết định an toàn:

- Target MLP có AP gấp khoảng 9,4 lần prevalence, cho thấy observation có khả
  năng nhận biết một phần opportunity C2/C3.
- Tuy nhiên trên tuning, threshold target-MLP cho gain cao nhất có mean gain
  +0,00742 Mbit/state nhưng làm tăng ping-pong +0,00833 và interruption
  +0,00033 s/state; vì vậy bị gate loại.
- Joint-linear có gain tuning cao nhất +0,00202 Mbit/state nhưng tăng ping-pong
  +0,00417/state, nên cũng bị loại.
- Timing không có threshold can thiệp nào cho mean gain dương trên tuning.

## Diễn giải và gate

Kết quả không ủng hộ việc đưa predictor hiện tại trực tiếp vào PPO hoặc chạy
pilot dài. G1 đã cho thấy timing/reference có dư địa vật lý, nhưng G2 yêu cầu
predictor tạo net physical gain dương mà không đổi bằng reliability. Cả linear
và MLP nhỏ đều không đạt điều kiện này.

Điểm quan trọng là target không hoàn toàn “không học được”: AP target-MLP cao
hơn prevalence rõ rệt. Nút thắt hiện tại là độ chính xác ở phần đuôi dùng để
ra action và chi phí false positive, không chỉ là thiếu positive hay thiếu
exploration. Không được diễn giải control fallback là PPO/3GPP tối ưu toàn cục.

## Artifact và tái lập

- Kết quả: `results/analysis/d63_2_probe_v1/probe_report.json`.
- Score và dự đoán continuous: `probe_scores.npz`.
- Hash package: `probe_manifest.json`.
- Script: `scripts/run_d63_2_probe.ps1` (từ chối ghi đè output cũ).
- Unit tests: feature contract, one-action-per-state, safe control fallback và
  overwrite protection.

## Hướng tiếp theo đề xuất

Không train PPO mới ngay. Nên thực hiện một diagnostic nhỏ trên TRAIN:

1. Phân tích false positive target-MLP theo speed/source và loại risk gây gate fail.
2. So sánh score với candidate geometry/trend có sẵn trong observation để biết
   lỗi do feature không đủ hay do probe multitask/class imbalance.
3. Nếu false positive tập trung ở một condition nhân quả rõ ràng, mở một D4
   observation ablation duy nhất. Nếu không có pattern ổn định, dừng D63 thay vì
   thêm model lớn/GRU hoặc sweep reward.

Holdout đã được mở cho probe v1; không được tiếp tục tuning trên chính holdout này.

Phân rã hậu nghiệm false positive được lưu tại
`d63_2_target_mlp_false_positive_report_vi.md`. Kết quả chính: tại threshold
chẩn đoán 0,9, cả hai FP đều là state D61 ở 50 km/h và đều dự đoán sai dấu
delivered data cùng HOF/RLF/recovery.

## Cập nhật đóng D63-3

Failure replay đề xuất `execution_safety_margin_v1_db`. Feature đã được kiểm
chứng lại trên TRAIN fit/tuning mà không đọc holdout. Dù AP đạt lift 5,25× trên
fit và 5,90× trên tuning, không có ngưỡng non-control nào trên fit vừa tăng data
vừa giữ mọi mean risk không xấu. Gate D63-3 đóng nhánh; xem
`d63_final_decision_report_vi.md`.
