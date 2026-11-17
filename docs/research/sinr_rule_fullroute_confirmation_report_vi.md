# Báo cáo kiểm chứng full-route của luật SINR trên 40 route confirmation

## 1. Mục tiêu

Kiểm tra luật quyết định post-TTT đơn giản trên toàn bộ quỹ đạo, thay vì chỉ đánh
giá nhãn cục bộ quanh anchor:

- Khi A3 đã đủ TTT, thực hiện `WAIT 20 ms` nếu SINR của target còn thấp hơn SINR
  của serving cell.
- Ngược lại, thực hiện `HO_NOW` tới target A3.
- Đối chiếu trực tiếp với B0 (3GPP A3) và B0A (engine online luôn `HO_NOW`).

Audit chỉ dùng fold `confirmation` của post-TTT TRAIN bank, không đọc validation,
test hoặc zero-shot. Đây là tập development confirmation đã được dùng trong chuỗi
T4C/T4D, vì vậy kết quả có giá trị kiểm tra full-route nhưng chưa phải xác nhận độc
lập cuối cùng trên dữ liệu hoàn toàn chưa dùng.

## 2. Dữ liệu và khả năng tái lập

- Nguồn split: `results/analysis/early_ho_timing/post_ttt_train_bank_v1/route_split.json`.
- Tổng số route: 40, gồm đúng 10 route cho mỗi tốc độ 30, 50, 70 và 90 km/h.
- Mỗi route được chạy hết quỹ đạo cho ba policy B0, B0A và luật SINR.
- Bootstrap: 2.000 lần, phân tầng theo tốc độ và lấy cụm theo route; seed 43.
- Số dòng validation/test đã đọc: 0.
- Manifest lưu hash của split, dataset provenance, từng cặp file RSRP/SINR, mã
  nguồn và mọi artifact kết quả.
- Hash của `route_outcomes.jsonl`, `main_report.json` và `decision_gate.json` đã
  được đối chiếu lại với manifest và đều khớp.

B0 và B0A khớp toàn bộ metric trên 40/40 route. Có 39 route khớp timeline tuyệt
đối. Route 90 km/h, trajectory 97 có thêm một `ho_prep.start` của B0 ở đúng mẫu
cuối (index 5999); engine online kết thúc trace trước khi có thể phơi bày quyết
định tương ứng. Trường hợp này chỉ được chấp nhận khi mọi timeline và metric dùng
trong nghiên cứu vẫn bằng nhau; mọi sai khác khác vẫn làm audit thất bại.

## 3. Kết quả aggregate

Luật SINR đã chọn `WAIT` 272/1.790 lần quyết định post-TTT, tương đương 15,20%.

| Metric | B0 / 3GPP | Luật SINR | Chênh lệch luật - B0 |
|---|---:|---:|---:|
| Relative rate trung bình | 90,0838% | 90,2776% | +0,1938 điểm % |
| Estimated rate trung bình | 21,5050 Mbps | 21,5470 Mbps | +0,0421 Mbps |
| HO attempts | 1.589 | 1.516 | -73 |
| HOF | 11,4537% | 10,6860% | -0,7677 điểm % |
| Ping-pong | 38,2206% | 36,3166% | -1,9041 điểm % |
| RLF time | 1,5167% | 1,3500% | -0,1667 điểm % |

Khoảng tin cậy bootstrap 95% của chênh lệch:

- Relative rate: +0,1938 điểm %, CI `[+0,1138; +0,2758]`.
- HOF: -0,7677 điểm %, CI `[-1,5710; +0,0287]`.
- Ping-pong: -1,9041 điểm %, CI `[-3,0196; -0,8865]`.
- RLF time: -0,1667 điểm %, CI `[-0,2667; -0,0667]`.

Relative rate, ping-pong và RLF time đều cải thiện với CI không cắt 0. HOF giảm
về giá trị trung bình, nhưng cận trên CI còn lớn hơn 0 một lượng rất nhỏ
(+0,0287 điểm %), nên chưa đủ bằng chứng để tuyên bố HOF không xấu hơn theo gate
nghiêm ngặt đã đặt trước.

## 4. Kết quả theo tốc độ

| Tốc độ | RR B0 | RR luật SINR | RR retention | HOF B0 | HOF luật | PP B0 | PP luật | RLF-time B0 | RLF-time luật |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 91,2227% | 91,4302% | 100,2275% | 12,7273% | 11,1821% | 36,1111% | 32,7338% | 1,4000% | 1,1667% |
| 50 | 92,5542% | 92,6634% | 100,1180% | 9,3645% | 9,4737% | 33,5793% | 31,0078% | 0,9333% | 0,9000% |
| 70 | 90,8251% | 91,0193% | 100,2138% | 9,4808% | 9,3301% | 40,3509% | 37,9310% | 1,4000% | 1,3000% |
| 90 | 85,7331% | 85,9974% | 100,3082% | 13,5397% | 12,2000% | 40,4922% | 40,3189% | 2,3333% | 2,0333% |

QoS retention vượt ngưỡng 98% ở mọi tốc độ. Riêng HOF tại 50 km/h tăng nhẹ
9,3645% lên 9,4737%; điều này cùng cỡ mẫu 10 route/tốc độ giải thích vì sao gate
HOF aggregate chưa vượt qua dù trung bình toàn bộ tập tốt hơn.

## 5. Kết luận gate

- B0 parity: PASS.
- QoS retention >= 98% aggregate và từng tốc độ: PASS.
- Relative-rate gain có CI dương: PASS.
- Ping-pong non-worse: PASS.
- RLF-time non-worse: PASS.
- HOF non-worse: FAIL do CI trên bằng +0,0287 điểm %.
- Strict-all: NOT PASSED.
- Trạng thái mở PPO: CLOSED.

Kết quả chứng minh rằng quyết định chờ ngắn dựa trên quan hệ SINR có tín hiệu vật
lý có ích trên full-route: vừa giữ/tăng QoS, vừa giảm số HO, ping-pong và RLF time.
Tuy nhiên chưa nên đưa luật này vào PPO như một kết luận đã xác nhận, vì HOF chưa
qua gate và 40 route này không phải tập hoàn toàn chưa từng dùng trong nghiên cứu.

## 6. Bước tiếp theo được khuyến nghị

1. Giữ nguyên luật và toàn bộ threshold; không tuning tiếp trên 40 route này.
2. Chọn một nhóm TRAIN route mới, không giao với T2, pilot post-TTT và bank hiện
   tại, rồi khóa danh sách trước khi chạy full-route confirmation.
3. Dùng cùng bootstrap/gate để kiểm tra lại HOF. Chỉ khi HOF non-worse vượt gate
   cùng QoS/PP/RLF mới mở pilot PPO hoặc dùng luật làm teacher/guardrail.
4. Nếu không còn đủ TRAIN route độc lập, báo kết quả hiện tại là development
   evidence và chuyển bước xác nhận cuối sang validation theo protocol nghiên cứu,
   thay vì tiếp tục tối ưu trên confirmation fold hiện tại.

## 7. Artifact chính

- Command: `run.py audit_sinr_rule_fullroute`.
- Kết quả chuẩn: `results/analysis/early_ho_timing/sinr_rule_fullroute_confirmation_v2`.
- Thư mục `sinr_rule_fullroute_confirmation_v1` là lần chạy dừng ở route 36/40 do
  phát hiện terminal-censored parity và được giữ nguyên để truy vết; không dùng làm
  kết quả chính.
