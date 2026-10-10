# Audit timing HO trước–trong–sau A3: kế hoạch triển khai

Ngày: 10/10/2026. Trạng thái: kế hoạch, **chưa triển khai/chạy audit mới**.
Base: `6f199a1`, chốt D62–D63 từ `codex/p3-reward-ablation`.
Nhánh: `codex/early-ho-timing`. Base đã qua 226 unit test.

## 1. Quyết định nghiên cứu và phạm vi

Ưu tiên timing-only: tại mỗi decision chỉ có `HO_NOW` hoặc `WAIT_20`.
Đích HO được xác định bằng một quy tắc cố định, không cho agent chọn target
trong gói này. Không đồng thời đổi reward, network hay radio/protocol.

Ba câu hỏi phải trả lời theo thứ tự:

1. Timing linh hoạt có tăng QoS và giảm HOF/PP trên cả route không?
2. Cho phép HO trước A3+TTT có thêm lợi ích so với chỉ trì hoãn sau A3+TTT?
3. Lợi ích có dự đoán được từ quá khứ/hiện tại, và vượt các quy tắc đơn giản không?

Chỉ dùng TRAIN tại tốc độ 30/50/70/90. Không dùng validation/test/zero-shot,
không train PPO trong gói audit. Không sửa hoặc ghi đè artifact D63, checkpoint
cũ hay cấu hình mặc định. Chạy main audit chỉ sau pilot correctness và duyệt chi phí.

## 2. Bằng chứng xuất phát, không diễn giải quá mức

Theo `d63_1_main_audit_report_vi.md`, trên 80 TRAIN route:

| Policy | RR % | HOF % | RLF-time % | PP % |
|---|---:|---:|---:|---:|
| Always-C1 | 89,868 | 11,800 | 1,576 | 36,065 |
| Timing-only reference | 91,294 | 3,965 | 0,321 | 19,464 |
| Timing×target reference | 91,606 | 4,431 | 0,338 | 19,405 |

Timing-only tăng RR 1,426 điểm phần trăm, CI95% [1,267; 1,584]. Đây là
reference dùng future 1 giây sau decision A3+TTT, **không phải policy online,
không chứng minh HO sớm có lợi và không phải upper bound toàn cục**.

D63-2 chưa tìm được ngưỡng dự đoán tạo net gain an toàn. D63-3 kiểm tra một
scalar selector cụ thể; thất bại đó không chứng minh mọi feature/observation
hay mọi cách học timing đều vô ích. Không sửa gate cũ để gọi D63 thành công.
Bank cũ tối đa 5 anchor/source/route chủ yếu sau TTT: giữ làm tham chiếu,
không đổi tên thành bank early-HO.

## 3. T0 — đóng băng contract trước khi viết engine mới

### 3.1 Baseline và cấu hình

Giữ baseline gốc: A3 offset 1 dB + hysteresis 1 dB, TTT 40 ms; với các offset
khác bằng 0, điều kiện vào là chênh lệch RSRP >2 dB liên tục đủ TTT.
Khi triển khai phải trích cấu hình thực tế và kiểm tra bất đẳng thức từ code,
không chỉ hard-code công thức rút gọn này. Radio 2,1 GHz, 10 MHz, dt 10 ms
và toàn bộ RLM/HO/recovery/filter giữ nguyên.

Tạo namespace cấu hình riêng `early_timing_v1`, không giảm A3 toàn cục.
Tham số đề xuất khóa cho vòng đầu:

- Early window mở khi best-neighbor RSRP L3 − serving RSRP L3 >1 dB.
- Đóng window khi delta <=0 dB; đây là hysteresis của adapter, không gọi là A3 chuẩn.
- Decision ngay khi mở, sau đó mỗi 20 ms khi window còn mở.
- A3+TTT shadow maturity tạo thêm decision nếu không trùng tick đã lên lịch,
  để baseline-adapter có thể tái hiện đúng thời điểm baseline gốc.
- Action WAIT phải tiến thời gian; không lặp decision/action vô hạn cùng tick.
- Không gọi agent trong HO execution, outage hoặc recovery; state machine
  protocol có ưu tiên cao hơn adapter.
- Không ép HO bằng timeout ẩn. Route end kết thúc episode; WAIT không reset
  RLM/A3 hoặc làm bỏ qua các tick vật lý.

Đích `C*` là neighbor có RSRP L3 lớn nhất tại tick decision, loại serving,
dùng cùng tie-break/điều kiện khả dụng vật lý với baseline. Trước TTT, C* không
được mô tả là cell đã đạt A3. Khi thực hiện HO, cố định target theo protocol.
Khi WAIT, lần decision sau tính lại C*. Target đổi thì đóng episode timing
cũ, hủy lịch cũ và mở episode mới nếu entry đúng; không reset timer protocol
ngoài những gì baseline thực hiện. Log nguyên nhân đổi target/đóng window.

Nếu không thể tái hiện baseline qua adapter, dừng T0, không tiếp tục lấy số liệu.

### 3.2 Baseline/đối chứng bắt buộc

| ID | Policy | Vai trò |
|---|---|---|
| B0 | 3GPP simulator gốc, ngưỡng vào hiệu dụng 2 dB, TTT 40 ms | Baseline chính |
| B0A | Adapter chỉ HO tại A3+TTT như B0 | Kiểm tra tương đương engine |
| B1 | A3 offset 0 + hysteresis 1 dB, TTT 40 ms | Giảm ngưỡng đơn thuần; lưu cả exit rule thực tế |
| B2 | Early adapter, HO ngay khi window mở | Đối chứng HO sớm đơn giản |
| B3 | Early adapter, delay cố định 40 ms | Đối chứng chờ cố định |
| Rpost | Lookahead timing, chỉ quyết định từ A3+TTT, WAIT 20 ms | Dư địa timing muộn |
| Rearly | Cùng lookahead, cho quyết định từ early window | Lợi ích bổ sung HO sớm |

B3 chỉ HO khi window/target episode còn hợp lệ; nếu mất điều kiện thì hủy
và chờ episode mới. Quy tắc này phải giống semantics delay của reference.
Không chọn B1/B2/B3 tốt nhất bằng holdout rồi coi đó là baseline đã khóa.
Các so sánh phải cùng route, initial state, seed và accounting vật lý.

## 4. T1 — correctness và pilot đo chi phí

### 4.1 Thiết kế module

Tái sử dụng clone/replay, metric và provenance của `audit_timing_target.py`,
`audit_timing_target_main.py`, `verify_d63_1.py`; không copy nguyên engine.
Thêm adapter early decision opt-in và command audit riêng. Không giả lập một
event A3 đã mature chỉ để gọi `start_handover`; cần entry point có precondition
rõ ràng cho early request, vẫn đi qua HO execution/HOF/RLM gốc.

Tên dự kiến, chưa phải lệnh có sẵn:

- `audit_early_ho_timing`: tạo bank và replay local.
- `audit_early_ho_timing_routes`: full-route references và controls.
- `verify_early_ho_timing`: metric/provenance/correctness report.

Không đổi shape observation/action của checkpoint cũ hoặc tự suy diễn schema
khi load. Các adapter/audit schema mới có version riêng; legacy path mặc định
phải giữ nguyên kết quả.

### 4.2 Tests bắt buộc

1. B0A = B0 từng tick: serving/target, HO/HOF/PP/RLF, delivered data, timer.
2. Boundary delta đúng 1/2 dB, vừa dưới/vừa trên; TTT40ms tương ứng dt10ms.
3. Window mở/đóng, target đổi, duplicate maturity tick, WAIT20 và route end.
4. WAIT không reset RLM/A3; interruption/recovery không nhận decision trái phép.
5. Delay0 khớp HO trực tiếp; delay bị hủy không bị ghi thành HO thành công.
6. Clone cô lập RNG, history, timers, cache; thứ tự chạy branch không đổi outcome.
7. Sửa future sau anchor không đổi observation/mask/decision trước anchor;
   reference được dùng future nhưng predictor không được truy cập future.
8. Zero-service trong execution/recovery và tích phân data khớp từng tick.
9. Metric raw denominators, zero HO attempts/successes; horizon truncation.
10. Hash/config mismatch, non-TRAIN input, output tồn tại đều bị từ chối.
11. Bộ unit test legacy và checkpoint compatibility tiếp tục pass.

Pilot: 2 TRAIN route/speed, tổng 8, sample seed17; local tối đa 2 anchor/source/
route, horizon1s trước. Chạy đủ B0/B0A/B1/B2/B3/Rpost/Rearly trên 8 route để
đo chi phí full-route, không suy ra từ local replay một cách máy móc.
Log wall time, decisions, simulated ticks, branches, peak memory và disk.
Ước lượng main theo phân phối độ dài/decision density, báo median và worst
pilot route; không hứa ETA trước khi đo.

Gate T1: 100% tests và B0 equivalence pass; provenance đầy đủ; không mất tick;
chi phí được báo và người dùng duyệt trước main. Pilot để bắt lỗi, không chọn
threshold dựa trên route nào cho kết quả đẹp.

## 5. T2 — TRAIN bank và counterfactual local

Main dùng 20 route/speed = 80 TRAIN route, chọn bằng seed17, không chọn theo
outcome. Có thể dùng các route D63 để paired comparison; đó là development
data đã xem, không gọi là unseen confirmation.

Nguồn state: B0, B2, B3. Mỗi source/route tối đa 5 anchor tổng cộng, phân bố
vào early-before-A3-entry, A3-pending-TTT và post-TTT nếu có. Lấy bằng hash/
reservoir với seed khóa trước outcome, không chọn theo gain. Không nhân bản
state để lấp quota. Log số state thiếu và cùng-state duplicates.

Mỗi anchor lưu route/data hash, source/version/seed, tick/time, serving/C*,
L3/filter/RLM history và timers cần replay, phase, shadow maturity, causal
observation schema và fingerprint state. Shadow maturity xuất phát từ clone
control cùng anchor, không dùng lịch B0 toàn route sau khi trajectory đã lệch.

Replay lịch delay {0,20,40,80,160,320} ms và baseline continuation tại cùng
anchor; dùng endpoint chung horizon {0,5;1;2}s, primary1s. Nếu horizon cắt
execution/recovery, đánh dấu censored và báo sensitivity2s; không coi chưa
thấy failure là an toàn. Delayed request bị hủy nếu window/target episode
mất hiệu lực; log requested/executed delay, actual target và lý do hủy.
Sau intervention, tiếp tục baseline từ **state đã thay đổi**, không chép lịch
HO tương lai của control. Mỗi branch bắt đầu từ clone giống nhau.

Nhãn lưu riêng delta Mbit, RR, HOF count, PP count, recovery seconds,
interruption seconds, HO attempts và execution status. Không chỉ lưu argmax.
Timing label HO_NOW vs WAIT20 phải ghi rõ continuation, không nhầm nhãn này
với advantage của PPO. WAIT20 nghĩa là đến decision kế tiếp; policy tiếp diễn
cần được version hóa, không tự gán future optimal vào causal predictor.

Local audit trả lời cơ chế/tín hiệu nhãn; không cộng gain của các anchor chồng
lấn để suy ra cải thiện cả route.

## 6. T3 — full-route physical opportunity audit

Chạy 7 policy trong bảng trên cùng 80 route: dự kiến 560 policy-route rows.
Rpost/Rearly dùng cùng horizon1s, delay set, target rule và comparator.
Tại mỗi decision, so sánh clone schedules; luôn có baseline-continuation làm
fallback. Shortlist không tăng HOF/PP/recovery so với control cùng horizon,
sau đó chọn delivered Mbit lớn nhất; hòa thì ưu tiên baseline rồi delay ngắn.
Chỉ thực thi action đầu và lập lại ở decision sau (receding horizon).
Reference dùng future chỉ để đo dư địa; kiểm tra rủi ro cả route vì local
ràng buộc không bảo đảm global an toàn.

Phân tích riêng Rearly−B0, Rearly−Rpost và Rearly−B1/B2/B3. Ghi tỷ lệ HO
trước A3 entry, trong TTT, sau TTT bằng shadow control tại cùng decision;
không ghép các HO có chỉ số thứ tự giống nhau giữa hai trajectory đã phân kỳ.

### Metrics và uncertainty

- RR route =100×estimated_rate/max_estimated_rate; aggregate mean theo route,
  giữ nguyên mẫu số exogenous hiện hành và kiểm chứng từ raw data.
- Delivered Mbit tích phân actual served rate theo tick; không cấp data ảo
  khi execution/recovery. Báo mean rate và QoS retention so với B0.
- HOF%=100×sum(HOF)/sum(attempt); PP%=100×sum(PP)/sum(success).
- RLF-time%=100×sum(recovery seconds)/sum(route duration).
- Thêm HOF/PP/HO mỗi phút, raw counts và route low-tail (p10 rate).
  Mẫu số bằng0 báo N/A, không giả định 0% là tốt; chống cải thiện tỷ lệ giả.
- Aggregate và từng speed; paired bootstrap theo route, stratify speed,
  2.000 resample, seed29, CI95%; recompute pooled ratios trong mỗi resample.
  Mọi source/anchor của cùng route là một cluster, không bootstrap từng row.

Gate đề xuất phải khóa trước main, đây là tiêu chí nghiên cứu không phải
ngưỡng tiêu chuẩn viễn thông:

| Gate | Điều kiện | Kết luận |
|---|---|---|
| G-physical strong | RR tăng >=0,5 pp và CI lower>0; HOF và PP giảm >=10% tương đối ở point estimate, CI upper của delta<=0; RLF-time CI upper<=0 | Có bằng chứng cải thiện đồng thời |
| G-QoS guard | Retention rate và RR >=98% aggregate và mỗi speed | Điều kiện tối thiểu cho trade-off, chưa đủ gọi QoS tăng |
| G-early | Rearly−Rpost RR CI lower>0 và HOF/PP/RLF không xấu theo CI | Có bằng chứng lợi ích bổ sung quyền HO sớm |

G-physical phải kèm G-QoS; báo per-speed deterioration kể cả aggregate pass.
Nếu chỉ giảm một trong HOF/PP hoặc chỉ giữ QoS: gọi trade-off, không tuyên bố
vượt cả ba mục tiêu. Nếu CI rộng: inconclusive, không FAIL vật lý tuyệt đối.
Cho phép đề xuất một lần mở rộng tối đa40route/speed TRAIN, phải được duyệt,
không lặp đến khi p-value đẹp. Nếu baseline risk bằng0, báo absolute delta,
không áp điều kiện relative reduction vô nghĩa.

Interruption được báo đầy đủ nhưng không là hard gate độc lập trong plan mới:
QoS đã phản ánh chi phí mất data. Đây là lựa chọn tiền đăng ký mới, không được
áp ngược vào D63 để thay kết luận cũ.

## 7. T4 — chỉ mở causal learnability khi T3 có tín hiệu

Không train PPO ngay khi reference tốt. Probe tuyến tính và MLP nhỏ trên
observation nhân quả: serving/C* RSRP L3, SINR/RLM margin, trend/variance quá
khứ, elapsed A3/early window, thời gian từ HO, trạng thái RLM. Không đưa
future trend, actual future maturity hoặc outcome reference vào input.
Feature phải có định nghĩa đo được, đơn vị, normalization fit-only và tests
causality; bắt đầu từ schema sẵn có rồi chỉ thêm timing fields cần thiết.

Route split TRAIN fit/tuning/holdout60/20/20, stratify speed seed29, tất cả
sources của cùng route chung fold. D63 holdout đã được phân tích không còn
blind: nếu dùng lại phải gọi development holdout. Nếu muốn confirmation mới,
chỉ lấy TRAIN route chưa dùng trong diagnostic và ghi provenance chứng minh;
không chuyển sang test/zero-shot vì thiếu route. Khóa split trước labels.

Fit model trên fit, chọn operating point trên tuning, đánh giá một lần trên
holdout. Báo prevalence/AP/calibration, confusion theo speed/phase/source,
delta data và các risk của positive predictions, route bootstrap. Không dùng
AP/entropy/tỷ lệ WAIT đơn lẻ làm gate thành công.

Chạy predictor frozen full-route để phát hiện distribution shift; predictor
chỉ chọn timing, C* giữ nguyên. So với B0 và rule đã khóa trên tuning.
Chỉ mở pilot PPO riêng khi predictor có physical gain an toàn, QoS guard đạt
và vượt rule đơn giản; ngân sách pilot10.240 step là gói cần duyệt sau,
không nằm trong lần triển khai audit này.

## 8. Artifact, tái lập và thứ tự giao việc

Output mới theo run ID dưới `results/analysis/early_ho_timing/`; từ chối thư
mục đã tồn tại. Không tự resume/ghi đè. Nếu sau này cần resume phải xác thực
manifest/hash và chỉ append route hoàn toàn chưa chạy bằng cơ chế riêng.

Tối thiểu: `run_manifest.json`, `protocol_contract.json`, `route_split.json`,
`bank_manifest.json`, `anchors.jsonl`, `local_outcomes.jsonl`,
`route_outcomes.jsonl`, `cost_report.json`, `verification_report.json`,
`decision_gate.json`, `report_vi.md`.
Manifest có git SHA/dirty diff hash, Python/dependencies, data/schema hashes,
seeds, full config baseline/early/reference, sample selection, metric definitions,
command thực tế, thời gian và status. Khóa các file đầu vào trước run.

Giao theo thứ tự, không mở tất cả nhánh cùng lúc:

1. T0 contract + adapter design review.
2. T1 code/tests + pilot8route + cost report; dừng xin mở main.
3. T2 bank80route + T3 paired full-route, báo gate và failure cases.
4. Nếu physical pass nhưng early không thêm lợi ích: ưu tiên timing sau A3,
   không cố bảo vệ giả thuyết early-HO.
5. Nếu đơn giản B1/B2/B3 tương đương reference: ưu tiên rule; chưa có lý do PPO.
6. Nếu reference cải thiện mà causal probe không: ưu tiên observation/đo đạc,
   không tăng budget PPO để bù thông tin thiếu.
7. Chỉ khi T4 pass mới đề xuất training timing-only có control matched.

## 9. Tiêu chí kết thúc gói

Có bằng chứng tái lập cho từng câu hỏi, kể cả kết luận âm hoặc chưa đủ dữ liệu;
không chỉ biểu đồ đẹp. Phân biệt rõ future reference, rule online, predictor
online và PPO. Báo cơ chế rủi ro, khoảng tin cậy, route/sample scope, chi phí
và quyết định tiếp tục/dừng. Không tuyên bố kết quả bài báo tổng quát từ TRAIN.
