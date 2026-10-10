# D63 — kế hoạch quyết định tiếp tục nghiên cứu timing và target

Ngày: 06/10/2026. Cập nhật chốt: 07/10/2026. Trạng thái: **D63-0 đến D63-3
đã hoàn tất. G1 PASS; G2 FAIL; feature execution-safety không qua gate
fit/tuning, vì vậy đóng D63 và không chạy training mới**. Xem
`d63_timing_target_pilot_report_vi.md`, `d63_1_main_audit_report_vi.md` và
`d63_2_label_dataset_report_vi.md`, `d63_2_learnability_probe_report_vi.md`,
`d63_final_decision_report_vi.md`.

## 1. Quyết định đề xuất

**Chưa dừng nghiên cứu; dừng mở thêm biến thể PPO theo phỏng đoán.**
Tiếp tục D63 dưới dạng gói diagnostic hữu hạn: xác định dư địa cải thiện
timing × target, khả năng học từ observation nhân quả, rồi mới chọn một can thiệp.
Không chạy ngay D63-A prior 50/50, không promote D62 lên 100k/300k.

Tài liệu này thay thế thứ tự triển khai trong phần “Hướng triển khai tiếp” của
`d62_action_selection_audit_report_vi.md`. Giữ nguyên báo cáo đó như lịch sử;
không sửa kết quả hoặc diễn giải rằng kế hoạch mới đã được thực hiện.

D63 gồm D63-0/1/2 là chẩn đoán. D63-3 là quyết định gate. Pilot training là gói
sau, chỉ triển khai nhánh được chọn. Yêu cầu lập plan không tự cho phép chạy pilot.

## 2. Bằng chứng và điều chưa biết

| Dữ kiện hiện có | Kết luận được phép |
|---|---|
| D61 RR 88,248%, RLF-time 0,387%, HOF 4,069%, PP 13,237%; 3GPP lần lượt 91,272%, 1,313%, 10,903%, 37,057% | Timing/redecision tạo đánh đổi có thật; chưa giữ được QoS gate 98% |
| D62 joint argmax RR 78,769%; hierarchical argmax RR 91,289%, gần Always-C1 | Đổi selector không chứng minh policy tốt hơn |
| D62 sampling HOF khoảng 58%, RLF-time khoảng 6,1% | Tăng tần suất thử alternative không tương đương chọn đúng ngữ cảnh |
| 167/3.117 state cũ có alternative thắng best(STAY,C1) ở horizon 1 s | Có cơ hội cục bộ, không suy ra mức tăng throughput toàn route |
| Probe route-holdout AP 0,204, prevalence 4,18% | Có tín hiệu nhưng chưa chứng minh đủ độ chính xác/giá trị khi triển khai |
| D61/D62 mới 10.240 macro-step, 10 PPO update, seed 17 | Không đủ để tuyên bố hội tụ hoặc train dài hơn vô ích |

Những điều cần tránh:

- Không bắt buộc C2/C3 phải được chọn mới gọi là thành công. Timing tốt với C1
  vẫn có giá trị nếu physical metrics tốt.
- Không gọi xác suất gần đồng đều là thiếu exploration hoặc hội tụ chắc chắn.
- Khoảng P(HO) hẹp chưa chứng minh không phân biệt state; phải đo ranking với nhãn.
- Không quy toàn bộ lỗi cho reward: audit cũ cho thấy E07 nhận đúng dấu ở phần
  lớn alternative hữu ích, nhưng weighted reward không bảo đảm QoS constraint.
- Không dùng nhãn STAY reject-event cũ để kết luận trực tiếp cho redecision 100 ms.
- Không xem tối ưu hữu hạn bằng future là upper bound toàn cục hoặc policy triển khai.

Nguồn: báo cáo D61, D62, action-selection audit, candidate-learning gate và
artifact JSON tương ứng. Số làm tròn ở đây không phải ground truth cho unit test.

## 3. Mục tiêu và hợp đồng thí nghiệm

### 3.1. Hai mức mục tiêu, không được đánh đồng

1. **Mục tiêu khả thi trước mắt:** giữ ít nhất 98% mean Mbps và relative rate
   của 3GPP, aggregate và từng speed, đồng thời giảm lỗi HO/recovery.
2. **Mục tiêu mạnh cho bài báo:** vượt relative rate 3GPP có bằng chứng uncertainty,
   đồng thời không đánh đổi bằng HOF/PP/RLF-time tăng ngoài giới hạn đã đăng ký.

Đạt mục tiêu 1 không được viết thành “PPO vượt 3GPP về rate”. Không gọi mean
gate là bảo đảm QoS mọi route. Báo P10, phân bố paired delta và route suy giảm.

### 3.2. Giữ nguyên trong D63

- Dataset hiện tại, manifest/split seed 17; TRAIN-only để tạo nhãn, fit probe,
  đặt threshold/normalizer. Không tạo lại data, không dùng test/zero-shot.
- Runtime 2,1 GHz, 10 MHz, dt 10 ms; K=3, ranked_v1, structural mask.
- Observation qos_full_v1; reward E07 chỉ để đối chiếu, không dùng return làm
  tiêu chí chính chọn nhãn physical.
- Gate đầu: A3 hoàn tất TTT. STAY: a3_redecision_v1, 100 ms; event leave,
  đổi target, RLF, HO execution, terminal đều qua engine thật.
- Checkpoint D61 và D62 đóng băng. Không sửa selector mặc định của benchmark.
- Không tự mở nhánh quyết định trước TTT. Đó là thay đổi phạm vi nghiên cứu,
  cần quyết định riêng nếu D63 cho thấy không gian timing hiện tại quá hạn chế.

### 3.3. Baseline và đơn vị đo

Luôn phân biệt:

```text
rho_j = D_policy,j / D_best,j
relative rate (%) = 100 * rho_j
QoS retention theo speed = mean(rho_policy) / mean(rho_3gpp)
```

Không thay bằng mean(rho_policy/rho_3gpp) mà không version metric.
E10 cũ dùng target 0,98 của rho, không phải 98% baseline 3GPP; giữ v1 để replay.
Nếu mở nhánh constraint mới, reference phải fit từ TRAIN cùng speed, không lấy
baseline validation làm training target. Lambda chạm trần không chứng minh bất khả thi.

## 4. D63-0 — khóa evidence và chuẩn hóa replay

### Công việc

1. Kiểm tra HEAD, working tree, cấu hình, checkpoint hash và artifact gốc.
   Tại lúc lập plan: branch codex/p3-reward-ablation, HEAD a3a5eb1; D62/audit
   còn thay đổi chưa commit. Không reset, checkout đè hoặc tự commit thay người dùng.
2. Lưu source provenance gồm commit và hash nội dung/diff của code thực sự chạy;
   chỉ commit hash không đủ khi working tree dirty.
3. Trích selected step, budget-end, objective, architecture, decoder, stay mode
   và validation của D61/D62/3GPP vào registry mới. Kiểm tra metric từ artifact,
   không copy bảng Markdown vào dữ liệu đầu vào.
4. Tái sử dụng audit_candidates/audit_policy_learning và evaluator hiện có.
   Kiểm tra replay anchor khớp tic, serving, physical candidate IDs, observation,
   mask, suppression/redecision state và lịch sử measurement.

### Acceptance

- Clone/replay không làm thay đổi engine gốc hoặc RNG của nguồn state.
- Outcome C1 và STAY đối chiếu với engine/evaluator trên synthetic trace.
- Metadata thiếu stay mode phải báo legacy rõ; không âm thầm dùng default sai.
- Checkpoint cũ vẫn load và cho cùng action/probability trên fixed input.

## 5. D63-1 — đo dư địa timing × target

### 5.1. Bank mới tương thích cơ chế hiện tại

- Pilot correctness: 5 route TRAIN/speed = 20 route, tối đa 5 anchor/source/route.
- Main bank: 20 route TRAIN/speed = 80 route, ưu tiên tái sử dụng route bank cũ
  nhưng tạo artifact mới và replay đúng cơ chế; tối đa 10 anchor/source/route.
- Ba nguồn: 3GPP C1, D61-selected joint deterministic, D62-selected joint
  deterministic. Các nguồn PPO chạy đúng objective của checkpoint.
- Lấy anchor phân bố trên route bằng sampling seed cố định, không chỉ decision đầu.
  Báo riêng mỗi nguồn; không xem anchor trùng/nguồn khác là mẫu độc lập.
- Lưu source, route, tic, decision index, initial/redecision, thời gian từ A3/HO,
  candidate physical ID, mask, observation và provenance nguồn/dataset/topology.

Main bank tối đa 2.400 anchor; số thực tế có thể ít hơn. Không lặp state để đủ quota.
Nếu thiếu positive, chỉ được mở rộng một lần lên 40 route/speed (160 route),
giữ quy tắc sampling. Không tìm route “đẹp” bằng outcome để đưa vào tập đánh giá.

### 5.2. Các phương án counterfactual

Tại cùng anchor, so sánh các policy can thiệp ngắn:

- Delay đề xuất: 0/100/200/300 ms.
- Target slot: C1/C2/C3 **tại decision thực thi**, cộng continuation control.
- Với delay d: chọn STAY tại các decision hợp lệ trước anchor_time+d; tại
  decision hợp lệ đầu tiên sau mốc đó chọn slot k nếu tồn tại, rồi trả điều
  khiển cho continuation đã khóa. Không ép engine HO đúng deadline khi không
  có decision; không đóng băng BS ID cũ rồi gọi đó là candidate k mới.
- Slot không tồn tại: ghi unavailable; áp fallback đã đăng ký (C1 hợp lệ, nếu
  không có thì STAY), lưu cả requested/executed action. Không loại route xấu.
- Nếu terminal/horizon đến trước khi thực thi, ghi no_execution/censored;
  vẫn tính dữ liệu và lỗi đến thời điểm kết thúc thực tế.

Mọi nhánh giữ dt/RLM/HO execution/recovery chuẩn. Candidate và observation phải
được dựng lại nhân quả ở mỗi decision, không chỉ chỉnh tic trên snapshot.

Continuation chính: Always-C1 sau can thiệp. Sensitivity: D61 frozen trên cùng
một tập con chọn trước. Không so nhánh A dùng C1 continuation với nhánh B dùng
D61 rồi quy khác biệt cho action đầu.

Horizon chính 1 s; sensitivity 0,5/2 s từ cùng anchor. Các nhánh cùng anchor
phải có cùng endpoint min(anchor+horizon, route_end). Horizon cắt ngắn phải
báo riêng; không kéo dài cửa sổ chỉ để nhánh delay có đủ thời gian nhận lợi ích.

### 5.3. Outcome và cách tổng hợp

Lưu từng nhánh, không chỉ nhãn argmax:

- Delivered Mbit, elapsed/connected/HO-interruption/recovery seconds.
- HO attempts/successes, HOF, pure RLF và RLF-start gồm HOF nếu protocol đếm như vậy,
  ping-pong; tách counts để tránh double-count khi diễn giải.
- Physical target thực thi, thời điểm, execution status và fallback.
- Reward từng thành phần, discount; chỉ dùng để đo reward/physical alignment.

Hai phân tích tách biệt:

1. Strict dominance: delivered data tăng mà các event/recovery không tăng.
2. Pareto frontier: rate cao hơn nhưng đổi bằng rủi ro, không ép mọi state có
   một nhãn duy nhất hoặc coi nhánh không strict-dominant là vô ích.

Báo timing-only (C1), target-only (delay=0), joint timing×target; paired theo
anchor nhưng uncertainty bootstrap theo route, phân tầng speed (2.000 lần,
seed 29). Không lấy hàng nghìn ticks làm hàng nghìn mẫu độc lập.

Không cộng delta của các anchor chồng lấn để suy ra throughput gain toàn route.
Đánh giá thêm trên route đầy đủ một **lookahead reference hữu hạn**: quyết định
bằng future horizon đã khóa ở mỗi decision, thực thi action đầu rồi tính lại.
Reference này phải chịu đúng protocol cost, không phải upper bound toàn cục.
Giữ C1, D61, timing-only, joint reference cùng route và cùng outcome definitions.

### Gate G1 — có đáng học không?

- Dư địa có ích: cải thiện xuất hiện trên nhiều route/speed và full-route reference
  cho Pareto trade-off tốt hơn control, không chỉ một vài delta ngắn hạn.
- Ngưỡng ưu tiên nghiên cứu đề xuất, phải khóa trước khi chạy: RR tăng ít nhất
  0,5 điểm phần trăm so với control tương ứng mà reliability không xấu theo point
  estimate; hoặc giữ QoS retention 98% so 3GPP và giảm ít nhất 10% tương đối một
  metric reliability, các metric còn lại không xấu theo point estimate.
- Các ngưỡng trên là tiêu chí sàng lọc thực dụng, không phải ngưỡng từ 3GPP hoặc
  tiêu chuẩn chứng minh paper. Luôn kèm interval và raw counts.
- Nếu CI còn rộng/positive quá ít: kết luận INCONCLUSIVE, mở rộng bank một lần.
- Nếu không đạt sau mở rộng: không mở pilot kiến trúc; ghi “không có bằng chứng
  đủ mạnh trong phạm vi đã khảo sát”, không kết luận mọi policy đều bất khả thi.

## 6. D63-2 — observation có học được cơ hội đó không?

### Thiết kế

- Chia route TRAIN thành fit/tuning/holdout 60/20/20, stratify speed, split seed29.
  Tất cả source/anchor của cùng route thuộc cùng fold; lưu ID/hash.
- Fit/tuning chọn model/threshold; holdout chỉ kiểm tra sau khi đã khóa.
  Nếu mở rộng bank theo gate, giữ fold bằng mapping deterministic đã đăng ký.
- Đầu vào chỉ observation hiện tại và planned action/delay được biết lúc quyết định.
  Không dùng measured outcome, future candidate identity, source label hoặc route ID.
- Probe tối giản: linear/ridge control và một MLP nhỏ, cùng folds. Giới hạn một
  cấu hình chính mỗi loại, không sweep lớn để tối ưu holdout.
- Dự đoán gain delivered data và risk theo timing/target; báo thêm nhãn hữu ích.
  Candidate comparison cần biết false-positive cost, không chỉ chọn lớp hiếm.
- Tách timing-only, target-only và joint để xác định chính xác nút thắt.

### Metric

- AP/prevalence, precision/recall; calibration nếu báo xác suất.
- Gain/loss khi thực sự dùng predictor chọn action, regret so reference hữu hạn.
- Mbit mất do false alternative; missed gain; confusion STAY/C1/C2/C3.
- So với Always-C1, D61 frozen và quy tắc nhân quả đơn giản được khóa từ TRAIN-tuning.
- Bootstrap route và báo theo speed/source, horizon 0,5/1/2 s.
- Đo AUC/AP của P(HO) đối với nhãn timing, conditional scores đối với nhãn target;
  không chỉ đếm bao nhiêu probability vượt 0,5.

Gate G2 không dựa vào AP tốt hơn random một mình. Predictor phải cho net physical
gain dương trên holdout mà không tạo reliability trade-off trái G1. Nếu CI chứa 0,
coi là tentative/inconclusive, không tuyên bố causal learnability đã được xác nhận.
Mốc adequacy đề xuất: ít nhất 50 positive anchor trên 20 route trong toàn bank;
holdout vẫn phải báo số positive/route riêng. Đây không thay thế power analysis.

## 7. D63-3 — chọn đúng một hướng

| Kết quả G1/G2 | Quyết định tiếp |
|---|---|
| Reference có ích, predictor cũng có ích, PPO chưa tận dụng | Pilot tín hiệu học phụ từ counterfactual TRAIN, giữ D61 làm control |
| Timing dự đoán được từ full observation nhưng head D62 không tận dụng | Đối chứng timing nhận pooled candidate context; giữ prior/decoder/reward như control D62 |
| Reference có ích, predictor hiện tại không đủ, positive đủ rộng | Mở D4 observation có giả thuyết cụ thể; không mặc định tăng chiều hoặc GRU |
| Target không có lợi đáng kể nhưng timing có lợi | Thu hẹp nghiên cứu timing; C1 là target vẫn hợp lệ |
| G1 yếu do chỉ được quyết định sau TTT | Đề xuất riêng nhánh A3-entry/pre-TTT; cần người dùng duyệt đổi phạm vi |
| Reward xếp các lựa chọn QoS tốt thấp một cách có hệ thống | Mở D5 target/constraint mới dựa TRAIN 3GPP, giữ kiến trúc đối chứng |
| Bằng chứng thiếu sau một lần mở rộng | Dừng vòng D63; lưu kết luận giới hạn, không tiếp tục sweep vô hạn |

Không triển khai đồng thời pooling + auxiliary + reward + early-trigger.
Prior 50/50 chỉ là ablation khởi tạo tùy chọn sau này; thay prior và decoder
cùng lúc không được trình bày là một-factor experiment.

## 8. Gói pilot sau D63 — chỉ khi được mở

### Ngân sách

- Tối đa một treatment và một matched control, seed17, 10.240 macro-step mỗi arm.
- D61 frozen là reference thực dụng, nhưng cần fresh control cùng instrumentation,
  budget, decoder, optimizer khi so nhân quả treatment/control.
- 10 PPO update chỉ là kiểm tra cơ chế. Nếu tín hiệu có hướng và chưa ổn định,
  được đề xuất một cặp fresh run 30.720 step; không tự resume hoặc gọi run nối
  checkpoint là fresh matched control.
- Không tự chạy 100k/300k. Sau pilot phải báo gate và xin quyết định gói tiếp.

### Nếu chọn auxiliary supervision

Nhãn counterfactual chỉ từ TRAIN-fit/tuning, không official validation hoặc
TRAIN-holdout. Tách batch/loss supervised với on-policy PPO; không đưa transition
offline vào surrogate PPO như thể có on-policy old log-probability. Version loss,
coefficient, sampling positives/negatives và normalization. Không thưởng C2/C3
theo tên action; bao gồm C1/STAY tốt và alternative xấu.

Future dùng để tạo teacher label trong mô phỏng, không được đi vào actor input,
mask hay selector. Gọi đúng là PPO + auxiliary supervision, không “vanilla PPO”.
Mọi pretraining là factor có đăng ký, không âm thầm warm-start treatment.

### Gate pilot

- Mechanism: gain/advantage/probability trên cùng transition phù hợp hơn với
  nhãn hữu ích; không tăng false-alternative cost. Log actor/critic gradient,
  effective parameter update, KL, entropy, saturation và value quality.
- Outcome: validation QoS aggregate/per-speed, reliability và tail so matched
  control; không quyết định chỉ bằng return, entropy hoặc tỷ lệ alternative.
- Fail rõ/deterioration: dừng treatment. Inconclusive nhưng learning curve còn
  tiến bộ: đề xuất cặp budget 30.720, không gọi model đã hội tụ.
- Muốn promote screening: retention gate >=98% mean Mbps và RR aggregate/per-speed,
  reliability có lợi nhất quán, không chỉ một selected checkpoint may mắn.

## 9. Xác nhận nghiên cứu nếu pilot thành công

Gói sau cần phê duyệt riêng: treatment + control, seed17/29/43, 100k fresh mỗi
run, cùng validation schedule và budget-end. Giữ selected/last riêng, báo biến
thiên seed và paired route uncertainty; tránh coi ba seed là đủ chứng minh mọi
kịch bản. Chỉ cân nhắc 300k khi learning curve và độ ổn định hỗ trợ.

Chốt hypothesis, metric primary, selector, checkpoint-selection và ngưỡng
non-inferiority trước final evaluation. Validation đã được dùng nhiều lần để
thiết kế nên không coi nó là bằng chứng xác nhận độc lập.
Không dùng test/zero-shot trong D63. Test lịch sử từng được xem không được
quảng bá là blind holdout mới; xác nhận paper cần mô tả giới hạn đó hoặc xin
gói dữ liệu/scenario độc lập sau khi khóa thiết kế.

## 10. Điều kiện dừng nghiên cứu nhánh hiện tại

1. Không tìm được dư địa thực dụng trong scope sau một lần mở rộng bank.
2. Có dư địa theo future nhưng probe nhân quả và một đối chứng observation có
   cơ sở vẫn không đạt net benefit đủ tin cậy: dừng claim timing+target hiện tại.
3. Sau một treatment có cơ sở và tối đa một budget extension, không cải thiện
   cơ chế hoặc QoS/reliability so matched control: đóng nhánh, không sweep tiếp.
4. Chỉ đạt rate bằng Always-C1, hoặc chỉ giảm HOF/PP bằng bỏ HO và mất QoS:
   không promote và không coi đó là đóng góp vượt baseline.

Dừng nhánh không đồng nghĩa bỏ project. Có thể chốt nghiên cứu timing-only,
đánh đổi QoS/reliability và giới hạn A3-gated control, với claim đúng bằng chứng.
Không cam kết khả năng xuất bản chỉ từ kết quả negative/pilot.

## 11. Deliverable, test và tái lập

Output đề xuất, **chưa có và chưa phải CLI đã triển khai**:

```text
results/analysis/d63_timing_target_v1/
  registry.json
  state_bank_manifest.json
  state_bank.npz
  timing_target_outcomes.jsonl
  route_reference_outcomes.jsonl
  probe_split_manifest.json
  probe_metrics.json
  decision_gate.json
docs/research/d63_timing_target_report_vi.md
```

Runner/command mới chỉ viết khi nhận yêu cầu triển khai. Refuse overwrite;
ghi args, seed, source/data/config/checkpoint hashes, Python/dependency versions,
stdout/stderr và elapsed time. Chạy pilot correctness trước để đo chi phí;
ước lượng main bank từ số branch × tick thực tế, không hứa thời gian chưa đo.

Test cần có: replay deterministic; event leave/target-change trong delay;
RLF/HO execution trong lúc chờ; terminal/horizon ngắn; candidate slot biến mất;
requested vs executed ID; clone isolation; candidate/history causality bằng
thay future sau anchor; delay0/C1 equality; fold không rò route/source; mask;
checkpoint legacy; helper audit không đổi collector/RNG khi tắt diagnostic.

Acceptance report phải nêu: đã chạy gì, lệnh thực tế, test pass/fail, artifact,
giới hạn, G1/G2 PASS/FAIL/INCONCLUSIVE và chỉ một nhánh được đề nghị.

## 12. Prompt giao gói đầu

> Đọc docs/research/d63_research_decision_plan_vi.md. Triển khai D63-0 và
> D63-1 trước, tái sử dụng engine/candidate audit; tạo TRAIN bank đúng
> a3_redecision_v1 và counterfactual timing×target có provenance. Viết unit tests,
> chạy pilot correctness, đo chi phí rồi báo gate trước main audit lớn. Không
> thay policy/reward/trigger, không chạy train mới, không dùng test/zero-shot,
> không ghi đè artifact cũ. Chưa triển khai D63-2 hoặc nhánh pilot đồng thời.
