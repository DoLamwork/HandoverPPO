# D63-0/D63-1 — báo cáo triển khai và pilot correctness

Ngày chạy: 06/10/2026. Phạm vi: TRAIN-only, không train model mới, không đọc
test/zero-shot, chưa triển khai D63-2 và chưa chạy main audit.

## 1. Kết quả bàn giao

D63-0 và phần local-counterfactual của D63-1 đã được triển khai thành command:

```powershell
.\scripts\run_d63_timing_target_pilot.ps1
```

Command tạo registry nguồn, TRAIN state bank, replay từng anchor, chạy các nhánh
delay × target và ghi physical/reward outcome. Output không được ghi đè nếu thư
mục đã có artifact. Artifact chính thức của lần chạy sau khi sửa phân loại là:

```text
results/analysis/d63_timing_target_pilot_v2/
```

Artifact `d63_timing_target_pilot_v1` được giữ nguyên như lịch sử. V1 phát hiện
một lỗi chỉ ở phần tổng hợp: nhánh C2/C3 chưa kịp thực thi trước horizon từng bị
gọi là target/joint. Raw simulation không bị thay đổi. V2 chỉ xếp target/joint
khi requested slot thực sự được thực thi; nhánh chưa thực thi được tách riêng.

## 2. Hợp đồng đã khóa

- Data: `uav_100m_grid100m_dev_processed`, split TRAIN, split seed 17.
- Tốc độ: 30/50/70/90 km/h; 5 route TRAIN mỗi tốc độ.
- Nguồn state: Always-C1, D61-selected và D62-selected.
- K=3, `ranked_v1`, observation `qos_full_v1`.
- Trigger đầu: completed A3+TTT; STAY: `a3_redecision_v1`, 100 ms.
- Counterfactual delay: 0/100/200/300 ms; target slot C1/C2/C3 được dựng lại
  tại decision thực thi, không đóng băng physical BS ở anchor.
- Horizon: 0,5/1/2 s, endpoint chung; continuation sau can thiệp là current C1.
- Reward audit: E07; tiêu chí physical không được thay bằng reward return.
- Fallback đã đăng ký: requested slot mất → current C1; không có C1 → STAY.

Checkpoint registry xác nhận:

| Source | Experiment | Architecture | Macro-step/update | Test |
|---|---|---|---:|---|
| D61 | D61_a3_redecision_pilot_b10240_s17 | candidate_scorer_v1 | 10.240 / 10 | false |
| D62 | D62_hierarchical_pilot_b10240_s17 | hierarchical_candidate_scorer_v1 | 10.240 / 10 | false |

Registry lưu checkpoint/report/config/reference hash; source hiện tại lưu HEAD,
branch, dirty status, tracked patch hash và hash cây source Python thực thi.
Checkpoint thiếu `stay_mode` không được D63 ngầm coi là redecision legacy.

## 3. State bank pilot

Bank có 300 anchor, đúng 100/source và 75/tốc độ. Mỗi source-route lấy tối đa
5 decision bằng một mẫu seed cố định trong các bin decision-index trải đều trên
route, thay vì luôn lấy vài decision đầu.

| Source | Tổng decision quan sát trên 20 route | Min–max/route | Anchor | Redecision anchor |
|---|---:|---:|---:|---:|
| Always-C1 | 844 | 17–61 | 100 | 0 |
| D61 | 2.087 | 30–175 | 100 | 63 |
| D62 | 3.682 | 47–303 | 100 | 87 |

Bank là NPZ không object/pickle. Replay kiểm tra lại tic, serving, A3 target,
A3 start, initial/redecision, candidate physical ID, mask và observation. Cả
300/300 anchor replay thành công. NPZ SHA-256:
`a8804414e8df1d9a3845a2af802fd5fb1d65ffbbb6444c921025d2348f7f9205`.

## 4. Counterfactual correctness và chi phí

- 3.600 branch simulation = 300 anchor × 4 delay × 3 target.
- 10.800 row outcome = mỗi branch tại 3 horizon.
- 709.260 primitive tick được mô phỏng.
- Thời gian phần counterfactual: 230,17 s; khoảng 3.081 tick/s.
- Ước lượng tuyến tính cho local audit 20 route/speed: khoảng 920,68 s
  (15,34 phút). Đây chưa gồm full-route lookahead reference và có thể phi tuyến.
- Fallback: 0 branch trên dataset pilot; logic fallback vẫn được unit test.
- Theo các row/horizon: 10.140 đã thực thi; 615 chưa có decision thực thi trước
  horizon; 45 kết thúc trace trước khi thực thi. Có 192 row bị cắt ngắn do route end.

Delay-0/C1 đã được unit test đối chiếu với helper counterfactual vật lý cũ.
Clone không đổi engine/builder nguồn. Unit test cũng bao phủ 100 ms redecision,
slot unavailable fallback, terminal-short-horizon, output refusal và phân loại
timing/target/joint.

## 5. Tín hiệu pilot — chỉ dùng để quyết định main audit

So với control delay=0/C1 trên cùng anchor, tỷ lệ state có ít nhất một nhánh
strict-dominant ở horizon 1 s là:

| Source state | Positive state | Positive route | Timing-only state | Target-only state | Joint state |
|---|---:|---:|---:|---:|---:|
| Always-C1 | 29/100 | 16/20 | 23 | 4 | 10 |
| D61 | 34/100 | 19/20 | 22 | 2 | 18 |
| D62 | 31/100 | 17/20 | 19 | 10 | 20 |

Một state có thể thuộc nhiều cột. `Strict-dominant` ở đây nghĩa delivered Mbit
tăng và HOF/RLF/PP/recovery/interruption không tăng trong horizon. Con số cao hơn
audit alternative cũ không mâu thuẫn: D63 thêm quyền delay/redecision và dùng
control delay=0/C1; audit cũ hỏi C2/C3 có thắng best(STAY,C1) hay không.

Pilot cũng cho thấy phần lớn lựa chọn lexicographic vẫn là delay=0/C1: 62–64%
state ở horizon 1 s tùy source. Cơ hội target-only ít hơn timing/joint ở Always-C1
và D61; D62 state source có nhiều target/joint hơn nhưng chưa phải bằng chứng
policy triển khai có thể nhận diện chúng.

Không được cộng delta của các anchor chồng lấn thành gain toàn route. Pilot chưa
có bootstrap route, paired interval hoặc full-route finite-lookahead reference.

## 6. Gate trước main audit

**Pilot correctness: PASS. Gate G1: NOT EVALUATED.**

PASS correctness vì:

- đủ 300 anchor, cân bằng source/speed và có 150 redecision anchor;
- replay 300/300 không mismatch;
- không đọc test/zero-shot, không train model;
- branch dùng engine thật, candidate được dựng lại tại execution;
- output/provenance đầy đủ và từ chối ghi đè;
- unit tests liên quan pass, không có fallback ngoài dự kiến trên dữ liệu thật.

Khuyến nghị: **mở main local audit 20 route/speed**, vì positive signal xuất hiện
trên 15–19/20 route ở horizon 1 s và chi phí local ước khoảng 15–20 phút. Tuy
nhiên chưa chạy main tự động. Trước khi đánh giá G1 đầy đủ phải bổ sung/chạy:

1. bootstrap theo route, phân tầng speed, seed 29;
2. paired delta magnitude và interval, không chỉ positive count;
3. full-route finite-lookahead reference cùng protocol cost;
4. Always-C1, D61 frozen, timing-only và joint reference trên cùng route;
5. raw count/denominator và rate/RR theo speed.

Chưa có cơ sở mở D63-2 hoặc pilot PPO. Main audit phải PASS/FAIL/INCONCLUSIVE
theo ngưỡng đã khóa trong kế hoạch D63 rồi mới thực hiện bước đó.

## 7. File và lệnh xác minh

- Code: `src/ho_optim_drl/commands/audit_timing_target.py`
- CLI: `run.py audit_timing_target`
- Runner: `scripts/run_d63_timing_target_pilot.ps1`
- Tests: `tests/test_d63_timing_target_audit.py`
- Plan: `docs/research/d63_research_decision_plan_vi.md`
- Artifact: `results/analysis/d63_timing_target_pilot_v2/`

Test liên quan:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_d63_timing_target_audit `
  tests.test_policy_learning_audit `
  tests.test_candidate_audit
```

