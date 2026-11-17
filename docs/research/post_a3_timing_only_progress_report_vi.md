# Báo cáo tiến độ nhánh timing-only sau A3

**Thời điểm chốt:** 13/10/2026  
**Phạm vi:** chỉ xét quyết định sau khi Event A3 đã hoàn tất TTT.

## 1. Bài toán đang nghiên cứu

3GPP thực hiện handover ngay tới target A3 khi điều kiện A3 duy trì đủ TTT. Nhánh
nghiên cứu này giữ nguyên trigger và target đó, chỉ cho policy chọn:

- `HO_C1`: thực hiện handover ngay tới target A3;
- `STAY`: chưa handover, chờ 20 ms rồi được quyết định lại.

Vì vậy đây là bài toán **timing-only hậu A3**, không phải chọn BS đích và cũng
không phải handover trước khi A3 đủ TTT. Cấu hình radio giữ cố định ở 2,1 GHz,
10 MHz và timestep 10 ms.

## 2. Các kiểm thử đã thực hiện

### Test 1 — B0/B0A parity: kiểm tra engine quyết định hậu A3

Mục tiêu là xác nhận engine online `HO_NOW` tái tạo đúng baseline 3GPP trước khi
thử WAIT. Trên 8 TRAIN routes, B0 và B0A khớp serving-cell timeline, trạng thái
HO/RLF, counter N310/N311/T310, throughput và metric cuối route.

**Kết quả:** PASS. Sai khác ở các test sau thực sự đến từ timing action, không
phải do hai engine mô phỏng khác nhau.

### Test 2 — T2/T3: kiểm tra dư địa vật lý của việc chờ sau A3

T2 tạo state bank TRAIN tại các thời điểm hậu A3 và replay các nhánh `HO_NOW`
hoặc chờ. T3 chạy reference full-route, dùng future chỉ để xác định upper
reference chứ không đưa future vào observation.

- T2 và T3 correctness: PASS.
- G-physical: PASS.
- QoS retention của reference post-TTT so với B0: mean rate 101,94%, relative
  rate 102,23%; từng tốc độ đều lớn hơn 101%.
- Phép thử lợi ích bổ sung của hành động trước TTT không đạt gate; do đó nhánh
  hiện tại tiếp tục tập trung vào post-TTT.

**Kết luận:** tồn tại dư địa vật lý cho timing-only sau A3, nhưng reference dùng
future chưa phải policy online.

### Test 3 — T4B: observation hậu A3 mở rộng

T4B tăng observation từ 20 lên 40 feature lịch sử/protocol và huấn luyện MLP
probe dự đoán WAIT có lợi.

- 545 anchor, còn 510 unique physical state sau loại trùng.
- Holdout AP 0,1948 so với prevalence 0,1224.
- MLP chọn WAIT 8/98 state.
- Full-route RR: B0 89,7141%; reference post-TTT 91,1835%; MLP 89,8610%; luật
  đơn giản 90,0179%.
- QoS guard PASS nhưng local learnability, full-route physical và vượt simple
  rule đều FAIL.

**Kết luận:** thêm nhiều feature có tăng ranking signal nhưng chưa đủ để mở PPO.

### Test 4 — Mở rộng post-TTT TRAIN bank

Để tránh kết luận từ quá ít positive state, bank được mở rộng bằng 240 TRAIN
routes chưa dùng:

- 3.573 post-TTT anchors;
- 3.070 unique physical state;
- 363 strict WAIT-positive;
- split theo route: 160 fit, 40 tuning, 40 confirmation;
- anchor cùng source/route cách nhau tối thiểu 500 ms;
- không đọc validation/test/zero-shot.

### Test 5 — T4C: probe trên bank mở rộng

T4C dùng MLP và threshold chỉ khóa trên tuning, sau đó đánh giá confirmation:

- fit/tuning/confirmation: 2.033/523/514 state;
- positive: 228/71/64;
- MLP AP/precision/recall: 0,2168/0,3333/0,1250;
- rule `target SINR < serving SINR`: 0,2511/0,3393/0,2969.

**Kết quả:** NOT_PASSED. MLP không vượt luật SINR đơn giản và bỏ sót nhiều WAIT
hữu ích.

### Test 6 — T4D: observation compact và tách QoS/risk

T4D thu gọn observation còn 14 feature cho QoS và 19 feature cho risk. QoS head
dự đoán lượng Mbit tăng khi WAIT; risk head dự đoán WAIT có làm xấu HOF, PP,
RLF, recovery hoặc interruption hay không.

Trên 514 confirmation state:

- QoS Pearson 0,1004; Spearman 0,3235;
- meaningful AP 0,2111 với prevalence 0,0720;
- selector an toàn chọn 0 WAIT;
- luật SINR chọn 56 WAIT, gain trung bình +0,02560 Mbit/state, CI95%
  `[+0,00092; +0,05623]`.

**Kết quả:** ranking signal PASS nhưng local utility và vượt simple rule FAIL.

### Test 7 — T4D.1: material-gain ranking

T4D.1 đổi cách học sang classification + pairwise ranking, đồng thời buộc
selector không được thắng bằng cách luôn HO_NOW.

Trên 514 confirmation state:

- material AP/prevalence: 0,2344/0,0973;
- precision/recall: 18,64%/44,00%;
- selector chọn 115 WAIT nhưng chỉ 17 trường hợp material-safe;
- mean gain +0,00109 Mbit/state;
- CI95% `[-0,03582; +0,03794]`;
- kém luật SINR khoảng 0,02535 Mbit/state.

**Kết quả:** model đã chịu chọn WAIT nhưng false-positive quá nhiều; local
utility, risk calibration và beats-rule đều FAIL. Việc tăng capacity không còn
là bước hợp lý tiếp theo.

### Test 8 — Luật SINR full-route trên development confirmation

Luật được cố định:

```text
Nếu target SINR < serving SINR: WAIT 20 ms
Ngược lại: HO_NOW
```

Trên 40 TRAIN routes thuộc confirmation fold, luật WAIT 15,20% quyết định:

| Metric | B0 | Luật SINR | Chênh lệch |
|---|---:|---:|---:|
| Relative rate | 90,0838% | 90,2776% | +0,1938 điểm % |
| HOF | 11,4537% | 10,6860% | -0,7677 điểm % |
| PP | 38,2206% | 36,3166% | -1,9041 điểm % |
| RLF-time | 1,5167% | 1,3500% | -0,1667 điểm % |

RR, PP và RLF-time có CI thuận lợi; HOF CI còn cắt 0 rất nhẹ. Strict-all chưa
đạt, nên luật chưa được dùng để mở PPO tại bước này.

### Test 9 — Luật SINR trên 40 confirmation routes hoàn toàn mới

Danh sách route được khóa outcome-blind trước khi chạy và không giao với các
bank trước. Luật không đổi threshold, WAIT 264/1.797 quyết định = 14,69%.

| Metric | B0 | Luật SINR | Chênh lệch | CI95% chênh lệch |
|---|---:|---:|---:|---:|
| Relative rate | 90,2143% | 90,4168% | +0,2025 điểm % | `[+0,1109; +0,3060]` |
| HOF | 9,4176% | 8,4801% | -0,9375 điểm % | `[-1,5366; -0,3887]` |
| PP | 38,0301% | 35,4954% | -2,5347 điểm % | `[-3,2520; -1,7746]` |
| RLF-time | 1,2667% | 1,0833% | -0,1833 điểm % | `[-0,2750; -0,1000]` |

**Kết quả:** strict-all PASS; QoS retention trên 100% ở cả bốn tốc độ. Đây là
bằng chứng trực tiếp để mở PPO timing pilot.

### Test 10 — T5 PPO timing pilot 10.240 step

T5 dùng hai action `STAY/HO_C1`, observation causal T4D.1 gồm 19 feature,
reward QoS + event + recovery, redecision 20 ms.

- Relative-rate/3GPP: 94,52%, không đạt gate 98%.
- STAY deterministic: 95,33%.
- Policy stochastic vẫn gần 50/50; entropy gần `ln(2)`.
- Critic gradient update cuối 22,111 so với actor 0,0713.
- Global clipping factor chỉ 0,0226.

**Kết luận:** T5 thất bại không chỉ vì reward; critic gradient quá lớn làm
gradient clipping chung gần như triệt tiêu cập nhật actor. Deterministic argmax
khuếch đại bias rất nhỏ thành gần như luôn WAIT.

### Test 11 — T5A PPO với critic chuẩn hóa

T5A giữ nguyên mọi thành phần của T5, chỉ thêm `fixed_scale_v1`. Scale 24,5694
được fit từ 80 TRAIN routes.

| Metric validation 150 routes | 3GPP | T5 | T5A |
|---|---:|---:|---:|
| Relative rate | 91,2718% | 86,2701% | **91,3440%** |
| RR/3GPP | 100% | 94,520% | **100,079%** |
| HOF | 10,9031% | 7,7778% | **10,2440%** |
| PP | 37,0569% | 32,0209% | **33,3333%** |
| RLF-time | 1,3130% | 0,6647% | **1,1260%** |
| STAY | 0% | 95,3299% | **23,7986%** |

Critic gradient update cuối giảm xuống 0,186; actor tăng lên 0,3077; clipping
factor đạt 1,0. T5A đạt QoS ≥98% aggregate và từng tốc độ, đồng thời các metric
HOF/PP/RLF-time đều tốt hơn 3GPP.

**Giới hạn:** entropy vẫn gần cực đại và `P(HO)` trung bình khoảng 0,503. T5A
qua pilot gate nhưng chưa chứng minh policy đã hội tụ ổn định.

## 3. Trạng thái hiện tại

Đã chuẩn bị T5A 100k với validation mỗi 10.240 step:

- deterministic validation dùng để chọn checkpoint;
- stochastic validation seed 17/29/43 dùng để đo độ ổn định;
- stochastic evaluation không làm thay đổi RNG training;
- chưa chạy test hoặc zero-shot.

Mục tiêu của run 100k không chỉ là tìm một checkpoint tốt, mà kiểm tra liệu kết
quả T5A có duy trì qua nhiều mốc, xác suất action có thật sự phụ thuộc state và
deterministic–stochastic có cho cùng kết luận hay không.

## 4. Kết luận tạm thời

Nhánh timing-only hậu A3 đã đi từ reference dùng future, qua nhiều probe nhân
quả thất bại, tới một luật SINR đơn giản được xác nhận full-route và PPO T5A qua
QoS pilot. Kết quả mạnh nhất hiện tại là T5A giữ RR ngang 3GPP trong khi giảm
HOF, PP và RLF-time trên validation. Tuy nhiên claim cuối cùng phải chờ T5A
100k ổn định, xác nhận nhiều seed và held-out test.

