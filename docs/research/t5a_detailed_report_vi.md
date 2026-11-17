# Báo cáo chi tiết T5A — PPO tối ưu thời điểm HO sau A3

## 1. Phạm vi và kết luận chính

T5A kiểm tra một bài toán **timing-only sau A3**: sau khi Event A3 đã thỏa
điều kiện và TTT đã hoàn tất, PPO quyết định `HO_NOW` đến C1 hoặc
`WAIT_20_MS`. Agent không chọn cell đích khác C1 và không được handover trước
thời điểm A3+TTT.

Kết quả chính thức mạnh nhất của T5A là held-out test 100k với ba training seed
17, 29 và 43. Trung bình ba seed, PPO giữ mean relative rate ở 91,0861%, so với
91,2537% của 3GPP Always-HO, tương đương **99,8164% retention**. Đổi lại, các
point estimate trung bình cải thiện:

- HOF: 10,579% xuống 10,076%, giảm 0,503 điểm phần trăm;
- ping-pong: 37,769% xuống 33,337%, giảm 4,432 điểm phần trăm;
- RLF-time: 1,269% xuống 1,096%, giảm 0,173 điểm phần trăm;
- số HO attempt: 5.407 xuống trung bình 4.894, giảm 513 HO mỗi 150 route.

T5A vì vậy chứng minh được rằng PPO timing-only có thể đổi một phần rất nhỏ
throughput lấy mức giảm rõ hơn về ping-pong, RLF-time và số HO. Kết quả chưa
chứng minh PPO có mean relative rate cao hơn 3GPP một cách ổn định qua seed.

## 2. Câu hỏi nghiên cứu

Baseline `3gpp_always_ho` thực hiện HO đến target A3 ngay khi TTT hoàn tất. T5A
đặt câu hỏi:

> Nếu giữ nguyên target C1 và chỉ cho PPO quyền chấp nhận hoặc trì hoãn HO sau
> A3+TTT, agent có giữ được QoS đồng thời giảm HOF, ping-pong, RLF-time và số HO
> hay không?

Đây là bài toán hẹp để tách riêng khả năng học **thời điểm HO** khỏi khả năng
chọn cell đích. Nó không phải bài toán early-HO và cũng chưa thay Event A3.

## 3. Từ T5 đến T5A

### 3.1. T5 control

T5 dùng cùng action, observation, reward và PPO nhưng critic không có
`fixed_scale_v1`. Sau 10.240 macro-step, deterministic validation chọn STAY
95,33%, mean relative rate chỉ còn 86,2701%, bằng 94,52% baseline. Không
checkpoint nào qua QoS gate.

Diagnostic cho thấy critic gradient khoảng 22,11 ở update 10, actor gradient
chỉ khoảng 0,0713 và global clipping factor còn 0,0226. Gradient critic lớn đã
làm toàn bộ gradient bị co khoảng 44 lần, khiến actor gần như không cập nhật.

### 3.2. Thay đổi duy nhất ở pilot T5A

T5A giữ nguyên bài toán T5 và thay critic target bằng
`value_normalization=fixed_scale_v1`. Scale 24,569382831 được fit từ 80 TRAIN
route, 20 route cho mỗi tốc độ; validation và test không được dùng để fit.

Ở update 10 của pilot T5A:

| Diagnostic | T5 | T5A |
|---|---:|---:|
| Actor gradient norm | 0,0713 | 0,3077 |
| Critic gradient norm | 22,1110 | 0,1860 |
| Global clipping factor | 0,02261 | 1,00000 |

Thay đổi này xử lý đúng mất cân bằng gradient và là lý do T5A được mở rộng lên
100k.

## 4. Cấu hình thí nghiệm

### 4.1. Dữ liệu và split

- Dataset: `data/uav_100m_grid100m_dev_processed`.
- Tổng số route: 1.000; TRAIN/validation/test = 700/150/150.
- Split seed: 17; training seed: 17, 29, 43.
- Tốc độ train/validation/test: 30, 50, 70, 90 km/h.
- Dataset content SHA-256:
  `0c50e6cbb899a9fcef27231cc7bb508539c53e209ded2a53738fc57f2873fcb4`.
- Topology SHA-256:
  `b86745a6e75acdc04b83c67d60c9f56b750b8cb3383a102fbb3b6008451cb690`.

### 4.2. Radio/protocol

| Tham số | Giá trị |
|---|---:|
| Primitive timestep | 10 ms |
| L3 filter coefficient | 1/16 |
| A3 offset | 1 dB |
| A3 hysteresis | 1 dB |
| A3 TTT | 40 ms |
| HO preparation | 50 ms |
| HO execution | 40 ms |
| STAY redecision | 20 ms |
| RLF recovery | 200 ms |
| T310 / N310 / N311 | 1.000 ms / 10 / 3 |

Với cấu hình này, decision đầu tiên chỉ xuất hiện sau khi A3+TTT hoàn tất.
`STAY` từ chối HO hiện tại và mở lại quyết định sau 20 ms nếu cùng sự kiện A3
còn hợp lệ. `HO_C1` bắt đầu HO đến C1 ngay tại decision point.

### 4.3. Action và target

- Candidate count: K=1.
- Action: `STAY`, `HO_C1`.
- Candidate order: `ranked_v1`.
- C1 được xếp theo current SINR, projected SINR 90 ms, sau đó legacy tie-break.
- Cả hai action luôn hợp lệ tại decision point; không có action mask thiên về C1.

Do K=1, kết quả T5A chỉ phản ánh khả năng học timing. Nó không đo khả năng chọn
C2/C3 hoặc thay đổi target A3.

### 4.4. Observation `t4d1_compact_v1`

Observation có 19 feature causal:

| Nhóm | Feature |
|---|---|
| Trạng thái A3 | thời gian A3, chênh RSRP target-serving |
| Chất lượng hiện tại | serving SINR-rate, target-serving SINR gap |
| Xu hướng RSRP | delta trend 20/100/200/500 ms |
| Xu hướng SINR | delta trend 200/500 ms |
| Độ bất ổn | RSRP-delta std 100/500 ms |
| Rủi ro vô tuyến | serving/target minimum SINR 200 ms |
| RLM | N310 progress, N311 progress, T310 active/progress |
| Lịch sử HO | target có phải cell trước đó hay không |

Không có future feature trong observation. Các cửa sổ lịch sử chỉ dùng dữ liệu
tại hoặc trước decision point.

### 4.5. Reward

Schema: `trajectory_qos_recovery_reward_v1`.

Reward rate tại mỗi primitive tick được chuẩn hóa bằng lượng dữ liệu tối đa của
chính trajectory, với tổng utility scale 60. Các penalty cố định gồm:

- HOF: -0,2;
- pure RLF không trùng HOF: -0,2;
- ping-pong: -0,1;
- thời gian RLF recovery: -1,0 mỗi giây;
- HO thông thường: 0.

Reward này dùng future trajectory để tính **reward/nhãn kết quả**, không đưa
future vào observation của policy.

### 4.6. PPO

| Thành phần | Giá trị |
|---|---:|
| Network | shared MLP 64-64, tanh |
| Learning rate | 5e-5 |
| Rollout | 1.024 macro-step |
| Batch | 128 |
| Epoch/update | 4 |
| Clip range | 0,2 |
| Entropy coefficient | 0,01 |
| Value coefficient | 0,5 |
| Max grad norm | 0,5 |
| Gamma per second | 0,99 |
| Lambda per second | 0,95 |
| Advantage normalization | có |
| Critic normalization | `fixed_scale_v1` |

Validation deterministic dùng `argmax`. Stochastic validation lấy mẫu action
với các sampling seed 17, 29 và 43.

### 4.7. QoS gate và checkpoint selection 100k

Checkpoint phải đạt đồng thời rate và relative-rate retention ít nhất 98% ở
aggregate và từng tốc độ. Trong gói 100k, `best_qos` ưu tiên theo thứ tự:

1. RLF-time thấp;
2. HOF thấp;
3. ping-pong thấp;
4. số HO thấp;
5. rate cao.

Đây là policy chọn checkpoint đã được khóa trước held-out test. Quy tắc
`qos_first_v2` được tạo sau này và không thuộc kết quả xác nhận 100k.

## 5. Các thí nghiệm đã thực hiện

| Giai đoạn | Budget | Seed | Dữ liệu đánh giá | Mục đích |
|---|---:|---|---|---|
| T5 control | 10.240 | 17 | validation | Xác định lỗi học ban đầu |
| T5A pilot | 10.240 | 17 | validation | Đối chứng critic fixed-scale |
| T5A 100k | 100.000 | 17/29/43 | validation | Kiểm tra độ ổn định theo seed |
| T5A final test | checkpoint 100k đã khóa | 17/29/43 | held-out test | Đánh giá xác nhận |
| T5A 300k | 300.000 | 17 | validation + test + zero-shot | Thăm dò learning dài |

Run 300k seed 29 đã dừng ở 40.960 macro-step; seed 43 chưa chạy. Vì vậy T5A
300k không có bằng chứng multi-seed hoàn chỉnh.

## 6. Kết quả pilot 10.240 macro-step

| Metric validation | 3GPP Always-HO | T5 control | T5A fixed-scale |
|---|---:|---:|---:|
| Rate (Mbps) | 23,0245 | 21,9673 | 23,0355 |
| Relative rate (%) | 91,2718 | 86,2701 | 91,3440 |
| HOF (%) | 10,9031 | 7,7778 | 10,2440 |
| Ping-pong (%) | 37,0569 | 32,0209 | 33,3333 |
| RLF-time (%) | 1,3130 | 0,6647 | 1,1260 |
| STAY (%) | 0 | 95,3299 | 23,7986 |
| P10 relative rate (%) | — | 75,4163 | 84,8660 |

T5A qua QoS gate ở aggregate và từng tốc độ. Tuy nhiên entropy vẫn gần
`ln(2)`, KL rất nhỏ và rollout cuối có mean `P(HO)` khoảng 0,503. Pilot chứng
minh lỗi critic đã được sửa, chưa chứng minh policy timing đã hội tụ.

## 7. Kết quả validation 100k, ba seed

| Policy/seed | Selected step | Rate (Mbps) | RR (%) | HOF (%) | PP (%) | RLF-time (%) | STAY (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3GPP Always-HO | — | 23,0245 | 91,2718 | 10,9031 | 37,0569 | 1,3130 | 0 |
| PPO seed 17 | 71.680 | 23,0393 | 91,3607 | 10,0345 | 32,5518 | 1,0967 | 28,8206 |
| PPO seed 29 | 81.920 | 22,9802 | 91,0781 | 10,0707 | 35,9904 | 1,1323 | 45,3190 |
| PPO seed 43 | 61.440 | 22,9903 | 91,1436 | 9,2515 | 31,1080 | 0,9734 | 53,9169 |

Cả ba selected checkpoint qua QoS gate 98%. Learning curve không đơn điệu và
checkpoint tốt nhất xuất hiện tại các step khác nhau, nên checkpoint cuối không
được tự động dùng thay selected checkpoint.

## 8. Held-out test 100k đã khóa

Trước khi đọc test, ba checkpoint, split, seed, tốc độ và SHA-256 được khóa
trong `final_test_lock.json`. Test gồm 150 route ở 30/50/70/90 km/h; không có
zero-shot và policy được đánh giá deterministic.

### 8.1. Kết quả aggregate từng training seed

| Policy/seed | Rate (Mbps) | RR (%) | HOF (%) | PP (%) | RLF-time (%) | HO attempt | STAY (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3GPP Always-HO | 22,7242 | 91,2537 | 10,5789 | 37,7690 | 1,2693 | 5.407 | 0 |
| PPO seed 17 | 22,7247 | 91,2627 | 10,1279 | 32,7758 | 1,1071 | 4.927 | 29,6317 |
| PPO seed 29 | 22,6664 | 90,9720 | 10,6175 | 35,9643 | 1,1831 | 5.020 | 44,3497 |
| PPO seed 43 | 22,6792 | 91,0237 | 9,4826 | 31,2704 | 0,9982 | 4.735 | 56,4121 |
| PPO mean ± SD | 22,6901 ± 0,0306 | 91,0861 ± 0,1551 | 10,0760 ± 0,5693 | 33,3368 ± 2,3967 | 1,0961 ± 0,0929 | 4.894 ± 145 | 43,4645 ± 13,4121 |

### 8.2. Chênh lệch trung bình PPO − 3GPP

| Metric | Mean delta qua ba seed | Diễn giải |
|---|---:|---|
| Rate | -0,03410 Mbps | Giảm 0,1501% so với baseline |
| Relative rate | -0,16755 điểm % | Retention 99,8164% |
| HOF | -0,50289 điểm % | Tốt hơn về point estimate |
| Ping-pong | -4,43220 điểm % | Cải thiện lớn nhất và nhất quán |
| RLF-time | -0,17319 điểm % | Tốt hơn ở cả ba seed |
| HO attempt | -513 | Giảm khoảng 9,49% |
| Controlled return | +0,34580 | Cao hơn ở cả ba seed |

Seed 17 gần như giữ nguyên rate và RR. Seed 29 và 43 mất lần lượt 0,282 và
0,230 điểm phần trăm RR. HOF tốt hơn ở seed 17/43; seed 29 xấu hơn 0,039 điểm
phần trăm, gần như ngang baseline. Ping-pong và RLF-time tốt hơn ở cả ba seed.

### 8.3. Chênh lệch theo tốc độ

Bảng dưới là mean delta của ba training seed trên held-out test:

| Speed | ΔRR (điểm %) | ΔHOF (điểm %) | ΔPP (điểm %) | ΔRLF-time (điểm %) |
|---:|---:|---:|---:|---:|
| 30 | -0,1519 | -0,5040 | -4,8251 | -0,1196 |
| 50 | -0,1756 | -0,2329 | -5,5410 | -0,1491 |
| 70 | -0,1831 | -0,3429 | -3,3934 | -0,2140 |
| 90 | -0,1793 | -1,0838 | -3,6496 | -0,3182 |

Trade-off giống nhau ở cả bốn tốc độ: RR giảm khoảng 0,15–0,18 điểm phần
trăm, trong khi các metric reliability nhìn chung cải thiện. Lợi ích RLF/HOF
lớn hơn ở 90 km/h.

### 8.4. Deterministic và stochastic

Deterministic STAY thay đổi mạnh theo training seed, từ 29,63% đến 56,41%.
Trong stochastic validation, tỷ lệ STAY của cả ba model lại nằm gần 49–50% và
RR chủ yếu khoảng 90,83–90,97%. Điều này cho thấy nhiều state có hai action
probability khá sát nhau; deterministic `argmax` khuếch đại các chênh logit nhỏ
thành policy có hành vi rất khác nhau.

Vì vậy T5A đã học đủ để tạo trade-off QoS–reliability hữu ích, nhưng ranh giới
timing chưa ổn định mạnh qua seed.

## 9. Thử nghiệm 300k và `qos_first_v2`

Sau held-out test 100k, seed 17 được train mới tới 300k. Checkpoint được chọn
ban đầu theo priority thiên về reliability có STAY 76,50% trên benchmark sáu
tốc độ và làm giảm RR 0,881 điểm phần trăm. Kết quả này cho thấy train lâu hơn
không tự bảo đảm QoS tốt hơn.

Sau đó một quy tắc hậu nghiệm `qos_first_v2` được tạo:

1. mean relative rate cao nhất;
2. HOF thấp nhất;
3. ping-pong thấp nhất;
4. RLF-time thấp nhất;
5. số HO thấp nhất.

Quy tắc này chọn checkpoint step 163.840. Benchmark gộp 210 route gồm held-out
30/50/70/90 và zero-shot 120/150 cho kết quả:

| Metric | 3GPP | PPO 300k `qos_first_v2` | Delta |
|---|---:|---:|---:|
| Rate (Mbps) | 22,0958 | 22,1165 | +0,0207 |
| Relative rate (%) | 89,5993 | 89,6790 | +0,0797 điểm % |
| HOF (%) | 12,4769 | 11,8000 | -0,6769 điểm % |
| Ping-pong (%) | 37,3237 | 36,1827 | -1,1410 điểm % |
| RLF-time (%) | 1,7152 | 1,5567 | -0,1585 điểm % |
| HO attempt | 8.672 | 8.322 | -350 |
| STAY (%) | 0 | 10,8840 | — |

Theo tốc độ, ΔRR lần lượt là -0,0031, +0,0628, +0,0597, +0,0053,
+0,1571 và +0,2702 điểm phần trăm cho 30/50/70/90/120/150 km/h. HOF chỉ xấu
hơn ở 50 km/h, +0,1425 điểm phần trăm; các tốc độ khác tốt hơn.

Đây là bằng chứng thăm dò đáng chú ý, nhưng không phải kết quả xác nhận chính
thức vì:

- chỉ có training seed 17;
- priority chọn checkpoint được thay sau khi held-out test đã được mở;
- benchmark tái sử dụng held-out test và đồng thời thêm zero-shot;
- chưa có paired/bootstrap confidence interval cho claim cuối.

Không nên trình bày bảng 300k này như một final held-out result độc lập.

## 10. Những gì T5A đã đạt được

1. Xác định và sửa được lỗi kỹ thuật khiến critic gradient lấn át actor trong
   shared MLP.
2. Xây dựng được timing-only action có redecision 20 ms và checkpoint contract
   tái lập.
3. Hoàn tất 100k với ba training seed và khóa checkpoint trước held-out test.
4. Chứng minh point estimate giảm ping-pong, RLF-time và số HO ở cả ba seed,
   trong khi giữ RR trung bình ở 99,82% baseline.
5. Chỉ ra policy deterministic còn nhạy với seed/argmax, dù reward aggregate
   cao hơn baseline.
6. Chỉ ra selection priority quyết định mạnh trade-off: priority reliability
   có thể làm policy WAIT quá nhiều; priority QoS có thể giữ RR nhưng giảm mức
   cải thiện reliability.

## 11. Claim có thể và không thể dùng

### Claim được dữ liệu 100k hỗ trợ

> Trên held-out test 150 route ở 30–90 km/h, timing-only PPO sau A3 giữ trung
> bình 99,82% relative rate của 3GPP Always-HO qua ba training seed, đồng thời
> giảm point estimate ping-pong, RLF-time và số HO ở cả ba seed.

### Claim chưa được hỗ trợ

- PPO có relative rate cao hơn 3GPP một cách ổn định qua seed.
- PPO đã học một ranh giới timing chắc chắn và có confidence cao.
- Kết quả tổng quát sang topology, channel model hoặc dataset khác.
- PPO có thể chọn target tốt hơn C1.
- PPO có thể handover sớm hơn A3+TTT.
- Kết quả 300k `qos_first_v2` là held-out confirmation độc lập.

## 12. Giới hạn thực nghiệm

- Held-out test đã được mở cho T5A 100k; mọi thay đổi sau đó phải dùng tập
  confirmation mới nếu cần một final claim sạch.
- Chưa có paired route-cluster bootstrap confidence interval cho multi-seed
  final table.
- Ba model dùng cùng split seed 17; multi-seed hiện tại đo training randomness,
  chưa đo split randomness.
- Target luôn là C1 và action chỉ mở sau completed A3+TTT.
- Deterministic action chịu ảnh hưởng của logit margin nhỏ.
- Route count theo tốc độ trong split không cân bằng; aggregate là route-weighted.

## 13. Artifact tái lập

- Pilot report: `docs/research/t5a_timing_ppo_fixed_scale_pilot_report_vi.md`.
- Quyết định khóa held-out test:
  `docs/research/t5a_100k_multiseed_final_test_decision_vi.md`.
- Training runner 100k: `scripts/run_t5a_timing_ppo_fixed_scale_100k.ps1`.
- Final-test runner: `scripts/run_t5a_100k_multiseed_final_test.ps1`.
- Final-test lock:
  `results/benchmark/early_ho_timing/T5A_100k_multiseed_final_test_v1/final_test_lock.json`.
- Held-out summaries:
  `results/benchmark/early_ho_timing/T5A_100k_multiseed_final_test_v1/seed_17`,
  `seed_29`, `seed_43`.
- 300k QoS-first selection manifest:
  `results/models/early_ho_timing/T5A_timing_ppo_fixed_scale_b300000_s17/qos_first_v2_selection.json`.
- 300k all-speed benchmark:
  `results/benchmark/early_ho_timing/T5A_300k_s17_qos_first_v2_all_speeds`.

Checkpoint SHA-256 đã khóa cho final held-out test:

| Seed | SHA-256 |
|---:|---|
| 17 | `0c122a4d5531ab8fa4c4971547c1889f7244a60db94b45e5e035ad139eb6a610` |
| 29 | `a17e4239bb026c0c14090330aaff3201cffa8431a025e2094cdc8dd7f9c2b3f3` |
| 43 | `be180d640047aab2f4c3fb3e59a6c39af66e0d29018589e34efd10959bb80b06` |

## 14. Quyết định nghiên cứu sau T5A

T5A 100k nên được giữ làm baseline PPO timing-only sau A3. Trước khi thay
trigger hoặc reward, bước phân tích còn thiếu có giá trị nhất là audit timeline
trên cùng route: ghép HO của PPO với 3GPP, đo phân bố độ trễ, rồi liên hệ các
quyết định WAIT với thay đổi rate, HOF, ping-pong và RLF. Audit này xác định PPO
đã trì hoãn đúng sự kiện hay chỉ tạo trade-off bằng cách giảm số HO nói chung.

Nhánh early-HO phải được đặt thành thí nghiệm mới, đối chiếu với T5A đã khóa,
vì nó thay đổi quyền quyết định của agent từ “sau A3+TTT” sang “trước khi TTT
hoàn tất”.
