# Shadow A3 và reference Rpost/Rearly — pilot TRAIN

Ngày chạy: 11/10/2026. Nhánh: `codex/early-ho-timing`.
Artifact có thẩm quyền: `results/analysis/early_ho_timing/rpost_rearly_pilot_v2`.
`rpost_rearly_pilot_v1` là lần chạy dở đã dừng vì profiler allocation gây chậm;
không dùng để phân tích và không bị ghi đè.

## Phần đã xây dựng

- Shadow A3/TTT theo dõi độc lập với action vật lý và không tự khởi động HO.
- Shadow maturity có ưu tiên nếu trùng tick với early redecision, nên không tạo
  hai action tại cùng primitive tick.
- `Rpost`: chỉ lập kế hoạch tại decision shadow A3+TTT.
- `Rearly`: lập kế hoạch từ early window `delta RSRP > 1 dB` và cả shadow A3+TTT.
- Cả hai chỉ dùng target current-best L3-RSRP; chưa mở bài toán chọn candidate.
- Mỗi decision so baseline continuation với delay `{0,20,40,80,160,320}` ms,
  horizon 1 giây. Loại nhánh tăng HOF, PP hoặc recovery cục bộ; trong phần còn
  lại chọn delivered Mbit lớn nhất. Hòa giữ baseline, sau đó ưu tiên delay ngắn.
- Chỉ thực thi action đầu `HO_NOW`/`WAIT_20`, rồi replan ở decision tiếp theo.
  Future chỉ tồn tại trong clone reference, không đi vào observation/policy.

## Correctness

- B0A shadow-immediate khớp B0 trên cả 8/8 route, 48.000 tick: RR, timeline
  serving/outage, HO/HOF/PP/RLF và các counter vật lý bằng nhau.
- Unit test bao phủ shadow parity, duplicate priority, WAIT20, delay execution,
  hủy request khi window đóng, clone isolation, risk filter và tie-break.
- Không đọc validation/test/zero-shot và không train model.

## Kết quả pilot 8 TRAIN route

| Policy | RR (%) | HOF (%) | PP (%) | RLF-time (%) |
|---|---:|---:|---:|---:|
| B0 | 90,497 | 10,671 | 43,686 | 1,458 |
| B1 | 90,504 | 10,030 | 43,919 | 1,375 |
| B2 | 91,246 | 9,420 | 34,400 | 1,083 |
| B3 | 90,150 | 13,462 | 31,556 | 1,458 |
| Rpost | 92,170 | 8,411 | 22,959 | 0,750 |
| Rearly | 92,508 | 8,257 | 23,000 | 0,750 |

So với B0, `Rpost` tăng RR `+1,673 pp`, giảm HOF `-2,260 pp`, PP
`-20,727 pp`, RLF-time `-0,708 pp`. `Rearly` tăng RR `+2,011 pp`, giảm HOF
`-2,414 pp`, PP `-20,686 pp`, RLF-time `-0,708 pp`.

Phần tăng thêm của quyền HO sớm (`Rearly - Rpost`) là RR `+0,338 pp`;
HOF `-0,154 pp`, PP `+0,041 pp`, RLF-time không đổi. RR tăng ở cả bốn speed
trong pilot (`+0,240` đến `+0,411 pp`), nhưng đây chỉ là hai route/speed.
Đặc biệt tại 50 km/h, cả Rpost/Rearly có HOF 10,870%, cao hơn B0 8,537%; do
đó pilot chưa đạt điều kiện để tuyên bố cải thiện an toàn theo từng speed.

`Rearly` thực sự dùng quyền HO sớm: tổng cộng 94 lần `HO_NOW` tại early-window
trên tám route (không chỉ tái hiện Rpost). Vì vậy chênh lệch giữa hai reference
không phải do hai đường code trùng nhau.

## Chi phí

- Runtime: 282,8 giây; median 26,2 giây/route; worst 99,0 giây/route.
- 10.542 branch và 1.052.674 simulated tick.
- Chiếu tuyến tính 80 route: khoảng 2.828 giây (47 phút), chưa cộng bootstrap,
  I/O và biến thiên decision density. Đây là số đo pilot, không phải cam kết ETA.
- `psutil` không có trong environment nên pilot không có RSS peak; artifact ghi
  rõ phép đo memory unavailable thay vì tạo số giả.

## Kết luận gate

T1 correctness/cost đủ để cân nhắc mở main audit; chưa đánh giá G-physical,
G-QoS hay G-early vì pilot không có bootstrap CI và sample quá nhỏ. Tín hiệu
ban đầu cho thấy timing sau A3 đã có phần lớn lợi ích, còn early-HO bổ sung một
mức RR nhỏ nhưng nhất quán ở bốn speed. Bước tiếp theo đúng kế hoạch là xin duyệt
T2/T3 trên 20 TRAIN route/speed, không train PPO và không đổi observation/reward.
