# D63-2 — phân tích false positive của target-MLP

Ngày chạy: 06/10/2026. Đây là **phân tích hậu nghiệm không refit**, chỉ đọc
score và TRAIN-holdout đã đóng băng của D63-2 v1. Không dùng kết quả này để
tuning lại threshold trên cùng holdout.

## Operating point

Threshold an toàn đã đăng ký là `1.000001`, chọn 0 alternative nên không có
false positive để phân tích. Vì vậy báo cáo dùng thêm hai điểm chẩn đoán:

- `0,9`: threshold có mean gain target-MLP cao nhất trên **tuning** khi bỏ
  constraint reliability. Đây là operating point chính để tìm nguyên nhân.
- `0,5`: sensitivity cố định, cho thấy pattern khi model can thiệp rộng hơn.

Hai điểm này không phải policy được duyệt.

## Kết quả chính tại threshold 0,9

Trên 240 state holdout, model đổi action ở 4 state: 2 true positive và 2 false
positive. Mean gain vẫn dương `+0,01472 Mbit/state`, nhưng phát sinh trung bình
`+0,00417 HOF`, `+0,00417 RLF-start` và `+0,000833 s recovery/state`, nên không
qua gate reliability.

### Theo tốc độ

| Speed | Action đổi | TP | FP |
|---:|---:|---:|---:|
| 30 km/h | 1 | 1 | 0 |
| **50 km/h** | **2** | **0** | **2** |
| 70 km/h | 1 | 1 | 0 |
| 90 km/h | 0 | 0 | 0 |

### Theo nguồn state

| Nguồn | Action đổi | TP | FP |
|---|---:|---:|---:|
| Always-C1 | 1 | 1 | 0 |
| **D61** | **3** | **1** | **2** |
| D62 | 0 | 0 | 0 |

### Theo target slot

| Slot | Action đổi | TP | FP |
|---|---:|---:|---:|
| C2 | 3 | 2 | 1 |
| C3 | 1 | 0 | 1 |

Hai false positive đều không chỉ vi phạm một penalty nhỏ. Chúng đồng thời:

- mất tổng cộng `2,69666 Mbit`, trung bình `-1,34833 Mbit/case`;
- tăng tổng 2 HOF và 2 RLF-start;
- tăng tổng `0,4 s` recovery;
- không tăng ping-pong hoặc pure-RLF; interruption không phải nguyên nhân chính.

## Hai case sai nghiêm trọng

1. `d61:50:81:78`, chọn C2, score `0,9978`:
   model dự đoán `+0,6720 Mbit`, HOF/RLF giảm khoảng `-0,21`, recovery giảm
   `-0,042 s`; thực tế `-1,3315 Mbit`, `+1 HOF`, `+1 RLF-start`, `+0,2 s recovery`.
2. `d61:50:182:13`, chọn C3, score `0,9754`:
   model dự đoán `+1,3190 Mbit`, HOF/RLF giảm khoảng `-0,26`, recovery giảm
   `-0,050 s`; thực tế `-1,3651 Mbit`, `+1 HOF`, `+1 RLF-start`, `+0,2 s recovery`.

Đây là lỗi sai dấu ở cả throughput và execution/recovery risk, không phải chỉ
calibration hơi lệch quanh threshold.

## Sensitivity tại threshold 0,5

Model đổi 14 action: 4 TP và 10 FP, precision 28,6%, mean gain
`-0,03731 Mbit/state`. Phân bố FP:

- Speed: 30/50/70/90 km/h = 0/4/4/2.
- Source Always-C1/D61/D62 = 2/6/2.
- C2: 6 FP trên 10 lựa chọn; C3: 4 FP trên 4 lựa chọn, không có TP.
- 10/10 FP mất delivered data; 7/10 tăng HOF, RLF-start và recovery.

Pattern rộng hơn vẫn cho thấy D61 chiếm phần lớn FP và HOF→RLF recovery là lỗi
chính. Tuy nhiên số case tại threshold 0,9 chỉ là 2, nên chưa được kết luận rằng
mọi state D61 hoặc mọi route 50 km/h đều khó.

## Kết luận kỹ thuật

Target-MLP có ranking signal nhưng quá tự tin ở một số state mà hậu quả HO là
rời rạc: candidate nhìn có lợi từ observation hiện tại nhưng execution dẫn đến
HOF/RLF. Continuous head cũng dự đoán sai dấu, vì vậy chỉ tăng penalty hoặc nâng
threshold không sửa được representation error này; nó chỉ khiến model quay về C1.

Nguồn D61 và speed 50 là nơi nên kiểm tra nguyên nhân, nhưng `source` không được
đưa vào feature. Cần đối chiếu hai case với feature hiện tại và trajectory vật lý
để tìm một tín hiệu nhân quả chung, chẳng hạn target SINR margin/trend trong cửa
sổ HO execution và RLM margin. Chỉ khi tìm thấy pattern lặp lại trên fit/tuning
mới nên mở một D4 observation ablation. Không tuning theo đúng hai route holdout.

Artifact: `results/analysis/d63_2_target_mlp_fp_v1/false_positive_report.json`
và `false_positive_cases.jsonl`.
