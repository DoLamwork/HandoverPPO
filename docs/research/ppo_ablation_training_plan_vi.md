# Kế hoạch kiểm thử các bản PPO tiếp theo

Ngày lập: 02/10/2026.
Trạng thái: P0 đã triển khai; P1 đã hoàn thành code, audit train-only và
screening E00-E03 ở 100k seed17. K*=3 được khóa làm nền gọn cho P2, nhưng P1
không tạo cải thiện vận hành: checkpoint feasible của E00/E02/E03 đều collapse
về 100% HO_C1 và tái tạo đúng 3GPP. P2-P5 vẫn là kế hoạch. Báo cáo:
`docs/research/p0_experiment_baseline_report_vi.md` và
`docs/research/p1_candidate_ablation_report_vi.md`.
Tài liệu thiết kế: docs/research/ppo_qos_observation_reward_design_vi.md.

## 1. Mục tiêu và nguyên tắc so sánh

Mục tiêu: giữ hoặc cải thiện relative rate so với 3GPP, đồng thời giảm HOF, ping-pong và RLF-time. Tỷ lệ chọn C2/C3 là chỉ số chẩn đoán, không phải mục tiêu tối ưu.

Mỗi thí nghiệm có một parent đối chứng, một thay đổi chính và một giả thuyết. Không thay candidate, reward, observation, entropy, learning rate và STAY cùng một lúc. So sánh cumulative chỉ kết luận lợi ích có điều kiện trên parent; cần thêm factorial/leave-one-out để kiểm tra tương tác trước khi viết paper.

Tập test đã từng được xem nhiều lần chỉ còn là bộ kiểm tra tương thích/replication. Không mô tả nó là blind holdout hoàn toàn. Nếu cần kết quả xác nhận cuối độc lập, dành route/seed mới chưa dùng tune; không cần tạo lại toàn bộ 1.060 route để làm các thí nghiệm phát triển.

Không mở test/zero-shot sau mỗi trial để chọn thiết kế. Dùng train cho fit/diagnostic/dual update; validation cho chọn cấu hình/checkpoint; final test sau khi freeze thiết kế.

## 2. Thiết lập cố định

| Thành phần | Cấu hình kế hoạch |
|---|---|
| Dataset | data/uav_100m_grid100m_dev_processed |
| Train / validation / test đã lưu | 700 / 150 / 150 route ở 30,50,70,90 km/h |
| Zero-shot | 120 và 150 km/h; 30 route mỗi tốc độ |
| Dataset split | Giữ manifest hiện có, split-seed 17 |
| Seed screening | 17 |
| Seed xác nhận | 17, 29, 43; nếu ngân sách cho phép thêm 71, 101 |
| MLP cơ sở | 64x64, tanh |
| Learning rate | 5e-5 |
| Rollout | 1024 macro-transition |
| Batch size | 128 |
| Epoch/update | 4 |
| Entropy coefficient | 0,01 |
| PPO clip | 0,2 |
| Value coefficient | 0,5 |
| Max grad norm | 0,5 |
| Gamma/second | 0,99, với aggregation nhất quán ở nền F1 |
| GAE lambda/second | 0,95 |
| Target KL | 0,03 |
| Validation interval | 10.240 macro-step, tức 10 rollout |
| Screening budget | 100.000 macro-step/arm/seed |
| Confirmation budget | 300.000 macro-step/arm/seed |
| Baseline protocol | A3/TTT, preparation/execution, RLM, recovery giữ nguyên |
| STAY | Giữ semantics reject-event đến giai đoạn riêng |

Đây là cấu hình gần bản 300k đã train, không dùng các default CLI khác (512 rollout, batch 64, lr 3e-4, 10 epoch) một cách vô tình.

Giữ cùng sampling distribution của route. Ghi số episode/primitive tick/giây mô phỏng bên cạnh macro-step vì các policy có macro-duration khác nhau. Chưa đổi sang speed-balanced sampler trong các arm chính; nếu đổi sampler, tạo arm riêng.

Các arm khác K/input/action schema được khởi tạo mới, không lấy checkpoint 109/11 để warm-start. Các arm đổi reward cũng khởi tạo mới để tránh carry-over policy. Chỉ resume cùng arm/cùng schema/cùng hyperparameter.

## 3. P0 — chuẩn bị nền thí nghiệm, chưa train dài

### 3.1. Khóa baseline và hợp đồng cấu hình

- Lưu source revision + patch/hash của working tree thực tế (hiện có nhiều thay đổi chưa commit), dataset manifest/hash, topology hash.
- Giữ nguyên checkpoint cũ làm historical reference; smdp_ppo_final.pt hiện là best_qos step 280.576, không phải weights step 300.000.
- Thêm experiment config có các field: experiment_id, parent_id, K, candidate_pool, candidate_order, mask_mode, observation_schema, history_windows, normalization, reward_schema/weights, discount_mode, stay_mode, model, seeds, budget.
- Lưu config hoàn chỉnh trong checkpoint và kết quả. Dùng các field trên là đặc tả triển khai, CHƯA phải CLI flags đang tồn tại.
- Khi đổi K phải đổi đồng bộ action enum/mapping, observation size, mask, collector, update, evaluation, checkpoint compatibility.

### 3.2. Bảo vệ quá trình đánh giá

Train command hiện tự evaluate test sau train ở train_smdp_ppo.py. Cần thêm lựa chọn skip final-test cho các arm phát triển; khi skip không sinh/đọc metric test trong selection.

QoS gate hiện dùng mean Mbps ratio. Thêm bộ đánh giá thống nhất dùng mean trajectory-relative-rate ratio theo speed, đồng thời giữ mean Mbps guard. Gate mới chỉ chọn model, không thay PPO loss ở những arm reward cũ.

Nếu không có checkpoint feasible, ghi rõ INFEASIBLE/best-effort. Không ghi fallback best-return như một nghiệm bảo đảm QoS.

### 3.3. Checkpoint và resume

Checkpoint hiện có model/optimizer/config/counter nhưng chưa đủ exact resume toàn bộ RNG, env, collector và lịch validation.

Trước khi dùng chiến lược 100k rồi nối lên 300k, cần lưu/restore RNG Python/NumPy/Torch, sampling state, collector observation + pending env/trace state, counters, checkpoint-selection records, schedule, và rollout nếu save giữa rollout. Ưu tiên save tại update boundary; có thể dùng cơ chế reset episode có công bố thay vì gọi đó là exact resume.

Nếu chưa triển khai và kiểm chứng resume, chạy xác nhận mới từ đầu đủ 300k. Không tính một run nối không tương đương thành replication chính xác.

### 3.4. Discount consistency

Giữ historical F0 đúng như code hiện tại. Xây nền F1:

    gamma_tick = gamma_per_second ** dt_seconds
    R_macro = sum_l gamma_tick ** l * r_l
    bootstrap = gamma_per_second ** macro_duration_seconds

Quy ước tick đầu exponent 0. Lưu cả raw undiscounted breakdown để báo cáo và discounted reward để update.

Kiểm tra cùng một chuỗi primitive reward được chia thành các macro boundary khác nhau cho cùng discounted return khi ghép đúng. Gamma=1 là ablation riêng sau này, không đổi cùng observation/reward.

Chạy smoke 2.048 macro-step để xác nhận mechanics; F0/F1 matched 100k seed17 nếu muốn định lượng tác động sửa aggregation. Tất cả arm chính dùng F1. So sánh trực tiếp checkpoint cũ với F1 mà không công bố thay đổi discount là không hợp lệ.

### 3.5. Kiểm tra bắt buộc trước train dài

- Always HO-C1 cho cùng serving timeline và protocol metrics với baseline, bất kể K hay slot order.
- Candidate IDs duy nhất, không serving; không drift giữa observation và action.
- Mask dùng nhất quán khi sample, lưu rollout, PPO update, deterministic evaluation.
- Thay phần trace sau decision t không làm đổi observation/candidate/mask tại t (causality).
- Lịch sử theo physical cell ID, không reset chỉ vì A3 mới bắt đầu.
- Reward/event counts khớp engine; phân biệt HO preparation start và execution attempt.
- Prefix trước action đầu không được gán cho action; full-trace QoS vẫn tính prefix.
- Checkpoint khác schema bị từ chối rõ ràng.
- Smoke không NaN, shape lỗi, action sai mapping hoặc reward bị đếm trùng.

Đầu ra P0: baseline config tái lập, smoke report, logger, F1 và checkpoint contract hoạt động. Chưa chạy train dài nếu mechanics chưa đạt.

## 4. P1 — chọn số lượng và cách tạo candidate

Trạng thái hoàn tất ngày 03/10/2026: selector ranked_v1, K=3/5/10, audit
counterfactual, checkpoint/benchmark schema động và registry screening đã hoạt
động. Audit 80 route train, 589 state cho coverage K=3 bằng 100% trên 38-39
positive state so với C1 và 30-34 positive state so với best(STAY,C1).
E00-E03 đã chạy đủ 100k seed17, không dùng test. E00/E02/E03 chỉ đạt QoS bằng
cách chọn HO_C1 100%; E01 collapse về STAY và infeasible. K*=3 được chọn theo
parsimony và coverage để tiếp tục P2, không phải vì đã chứng minh tốt hơn K=5/10.
Xem báo cáo P1 để biết checkpoint-selection effect và caveat một seed.

### 4.1. Audit offline trước

Mở rộng mẫu TRAIN: ví dụ 20 route/speed = 80 route, tăng tiếp nếu số cơ hội alternative quá ít. Giới hạn số decision lấy từ mỗi route để route có nhiều HO không chi phối toàn bộ mẫu.

Lấy state cả từ baseline và policy hiện có. Counterfactual trên cùng complete protocol state, same physical horizons 0,5/1/2 s và cùng continuation policy. Đo delivered bits, HOF, PP, RLF count và recovery time.

Future chỉ dùng làm nhãn đánh giá khả năng giữ cơ hội; không dùng để chọn shortlist/observation/mask.

### 4.2. Pool và thứ tự chung

Pool ban đầu giữ đúng 10 cell của candidate builder hiện tại để không đồng thời đổi topology coverage.

- Giữ C1 là A3 target.
- Rank một alternative theo chất lượng hiện tại, một alternative theo causal projected quality +90 ms.
- Loại trùng; append các cell còn lại theo một quy tắc deterministic đã freeze.
- Bộ K=3 là prefix của K=5, K=5 là prefix của K=10 trong cùng một ranked list.
- Giai đoạn này không dùng hard suitability mask và không bỏ cell vì Qin; chỉ dùng structural mask. Chất lượng kém bị xếp thấp nhưng vẫn tồn tại trong list K lớn.
- Causal rolling history có thể dùng trong selector ngay từ đây nhưng chưa đưa vào actor; công bố rằng agent đã được trợ giúp bởi selector. Mỗi so sánh chỉ gọi là thay selector, không phải giữ nguyên mọi thông tin gián tiếp.

### 4.3. Ma trận candidate

| ID | Parent | Thay đổi | K/action | Observation | Reward |
|---|---|---|---:|---|---|
| E00 | F1 | Baseline mới trên nền nhất quán | 10/11 | Old 109 | Legacy |
| E01 | E00 | Chỉ reorder cùng pool theo quality/trend | 10/11 | Old per-cell features | Legacy |
| E02 | E01 | Lấy prefix K=5 | 5/6 | 9+10x5=59 | Legacy |
| E03 | E02 | Lấy prefix K=3 | 3/4 | 9+10x3=39 | Legacy |

Mỗi arm 100k seed17. Cùng feature semantics/normalization, cùng mask và hyperparameter. E00 nếu đã train matched trong P0 thì tái sử dụng, không chạy lại vô ích.

Coverage_K = tỷ lệ state có alternative hữu ích trong full pool mà shortlist giữ lại ít nhất một. “Hữu ích” có tiêu chí freeze: vượt delivered data của C1 với không tăng HOF/PP/recovery time; cũng báo đối chiếu với best(STAY,C1).

Mốc định hướng 95% coverage chỉ là mục tiêu thiết kế, không là guarantee. Nếu sample chỉ có vài positive state, chưa được kết luận K=3 đủ chỉ vì coverage=100%.

Chọn K* dựa validation QoS, reliability, convergence, coverage và chi phí; khi hiệu năng chưa phân biệt rõ, ưu tiên K nhỏ hơn. Không chọn K vì entropy lớn hay non-C1 count lớn.

Đối chứng tùy chọn: old-order K=3 nếu cần đo nguy cơ “cắt C1-C3 máy móc”. Không phải cấu hình đề xuất mặc định.

## 5. P2 — observation, reward vẫn legacy

Khóa K*, ranked selector và structural mask từ P1. Train mới từ đầu mỗi arm; không tiếp tục weights của parent.

| ID | Parent | Chỉ thay đổi |
|---|---|---|
| E04 | Candidate arm được chọn | Rolling history theo cell ID: statistics 500 ms, SINR short slope 100 ms thay event-only 40 ms; giữ số feature cũ |
| E05 | E04 | SE không clip SINR sớm về [-10,10]; normalization monotonic dùng cùng scale giữa candidate |
| E06 | E05 | Observation đầy đủ 17 global + 15 feature/candidate: thêm N311, suitability persistence, p10, projected +90 ms, context |

Nếu K*=3: E04/E05 vẫn 39 chiều, E06 có 62.
Nếu K*=5: E04/E05 59 chiều, E06 có 92.
Nếu K*=10: E04/E05 109 chiều, E06 có 167.

Các stage là chuỗi ablation có điều kiện. E06 thêm một nhóm feature, không cho phép kết luận riêng từng feature gây cải thiện. Các ablation leave-one-out sau xác nhận:
- Bỏ projected +90 ms.
- Bỏ long history 500 ms.
- Bỏ speed.
- Bỏ p10/suitability fraction.
- Bỏ topology role flags nếu cần.

Cần đo sai số dự báo +90 ms và độ bão hòa feature trên train/validation; slope/prediction không là giá trị future thật. Fit normalization chỉ train, freeze vào checkpoint. Reward giữ legacy trong toàn bộ E04-E06 để tách tác động observation.

Không loại một bộ feature chỉ vì chưa có lợi ở một seed/100k. Chọn/promote dựa cả learning curve, per-speed QoS và uncertainty.

## 6. P3 — reward, khóa K* và observation E06

Khóa architecture/sampler/entropy/action trigger. Các run khởi tạo mới.

Định nghĩa QoS:
    rho_j = sum dt*R_j(t) / D*_j
    D*_j = sum dt*R_best,j(t)
    u_j,t = 60*dt*R_j(t)/D*_j

Normalizer full-trace chỉ thuộc simulator reward, không là actor/critic input. Công bố privileged training normalization. Nếu không chấp nhận normalizer này, chọn fixed train-fitted R_ref và ghi rõ objective khác; không trộn hai định nghĩa giữa các arm.

| ID | Parent | Chỉ thay đổi |
|---|---|---|
| E07 | E06 | Thay term rate bằng u; giữ event penalties legacy |
| E08 | E07 | Rescale event cost: HOF=0,2; RLF thuần=0,2; PP=0,1; HO cost=0 |
| E09 | E08 | Thêm RLF recovery-time cost 1 x dt mỗi tick recovery |
| E10 | E09 | QoS dual coefficient theo speed, điều chỉnh chỉ bằng training completed episodes |

E07 đổi cả trajectory weighting và scale reward rate. Nếu muốn tuyên bố lợi ích đến từ relative normalization thay vì chỉ tăng rate weight, phải có arm phụ R_scale: dt*rate/R_ref, với R_ref chọn từ TRAIN để matched scale, events giữ legacy.

HOF và RLF thuần là hai nhóm event không chồng; duration cost là tác hại thời gian bổ sung có chủ ý. Không thêm penalty target-ID hoặc bonus non-C1.

Trọng số trên là starting settings cần preregister; không tối ưu theo test đã xem. Nếu sensitivity cần thiết, mỗi lần chỉ thay một hệ số, ví dụ 0,5x/1x/2x, không quét mọi tổ hợp.

E10: cập nhật lambda theo speed khi đủ completed episodes trong một khoảng đã freeze, giữ cố định trong rollout/update; log lambda, rate shortfall, sample count từng speed. Nếu lambda chạm trần mà vẫn fail, ghi infeasible. Việc training stochastic đạt threshold không thay thế kiểm tra validation deterministic.

## 7. Kiểm tra tương tác observation và reward

Các bước tuần tự trên chưa đủ khẳng định đóng góp độc lập. Với K* và selector cố định, hoàn thành bảng 2x2:

| | Reward legacy | Reward mới cố định E09 |
|---|---|---|
| Observation old | Candidate baseline đã có | E11: arm bổ sung |
| Observation E06 | E06 đã có | E09 đã có |

Chỉ cần E11 nếu các ba arm còn lại đã chạy đúng setup. E10 adaptive là extension, không thay ô E09 rồi gọi toàn bảng là fixed reward ablation.

Đối chiếu gain observation khi reward legacy vs khi reward mới; gain reward khi observation old vs mới. Báo tương tác, không cộng cơ học các gain.

## 8. P4 — xác nhận nhiều seed và 300k

Chọn tối đa ba cấu hình để tránh tốn ngân sách:
1. Baseline chung E00.
2. Bản gọn tốt nhất (candidate/observation thay đổi, chưa cần dual nếu không có lợi).
3. Bản đầy đủ tốt nhất trên validation.

Train 300k với seed 17/29/43, cùng split17. Tối đa 9 run. Nếu đã có run 100k và exact resume đã kiểm chứng, nối SAME arm lên 300k; nếu không, chạy lại từ đầu và tính ngân sách tương ứng.

Không kết luận từ một seed. So sánh matched seed và cùng validation route, report variation across seeds. Một cấu hình được coi có triển vọng khi lợi ích có hướng nhất quán, không chỉ tốt ở một checkpoint của seed17.

Gate validation đề xuất:
- Mọi speed 30/50/70/90 có mean-relative-rate ratio >=0,98 baseline.
- Mọi speed có mean-Mbps ratio >=0,98 như guard bổ sung.
- So sánh RLF-time, HOF% và PP% riêng; không đổi thứ tự ưu tiên sau khi thấy kết quả.
- Trong feasible set: ưu tiên RLF-time, rồi HOF%, PP%, HO attempts/time, rồi rate.
- Báo Pareto set nếu có trade-off; không gắn nhãn “mọi chỉ số tốt hơn” khi chỉ count giảm.
- Nếu không feasible sau 300k, giữ là negative result và quay lại diagnostic, không tự cộng thêm budget vô hạn.

Đạt 98% là rate retention; muốn viết “vượt 3GPP” cần rate improvement dương có CI phù hợp. Threshold 98% mean không bảo đảm QoS mọi route; báo tail/low-decile route quality.

## 9. P5 — chỉ thử những thay đổi bổ sung khi chẩn đoán cho thấy cần

Mỗi arm so với cùng bản parent đã chọn, không gộp:

| Arm | Thay đổi | Khi nào nên chạy |
|---|---|---|
| M1 | Suitability-aware mask hoặc để slot trống | Nếu PPO còn thử quá nhiều candidate rõ ràng yếu |
| N1 | MLP 128x128 | Nếu value fitting và learning curve gợi ý thiếu capacity |
| N2 | Shared MLP candidate scorer | Nếu mapping/slot bias và generalization là vấn đề |
| H1 | Entropy coefficient/schedule | Nếu probability mass trên candidate hữu ích biến mất quá sớm |
| T1 | STAY cho redecision sau 100 ms khi A3 còn active | Nếu reject-event gây mất QoS trong cùng event |
| G1 | Gamma=1 finite horizon | Kiểm tra objective full-route so với discounted F1 |

M1 là đổi admissible action set, không chỉ giảm K. Không bắt alternative phải đạt A3 và không dùng future thật. Giữ C1 để policy vẫn có lựa chọn baseline.

N2 vẫn là feed-forward MLP PPO nhưng đổi kiến trúc; thêm parameter-budget-matched control nếu tuyên bố lợi ích từ weight sharing.

H1 không giữ uniform policy mãi. So sánh normalized entropy và probability dành cho alternative tốt, không so raw entropy giữa 4 và 11 action.

T1 là đổi semantics action/trigger. A3 vẫn gate lần đầu; thêm action opportunity khi event còn active. Báo riêng, không quy lợi ích này cho observation/reward.

Nếu chưa chứng minh lợi ích P1-P4, chưa cần chạy đồng thời tất cả P5.

## 10. Logging tối thiểu cho từng run

Training mỗi update:
- macro-step, primitive ticks/time, episodes, route/speed usage;
- policy/value loss, KL, clip fraction, grad norm, explained variance;
- raw entropy, normalized entropy H/log(n_valid) khi n_valid>1;
- counts STAY/C1/C2+, per-slot probability mean/quantiles;
- conditional candidate entropy khi chọn HO;
- mask availability và number valid candidates;
- reward rate/event/time từng thành phần, raw + discounted;
- duration macro p50/p90/p99 theo STAY/HO;
- lambda và QoS shortfall nếu có.

Action/diagnostic trace trên một bộ TRAIN cố định:
- route/speed/timestamp/serving ID;
- candidate physical IDs, slot mapping, valid mask;
- observation trước action, logits/probabilities;
- selected action, subsequent HOF/RLF/PP và duration;
- so sánh probability lúc alternative suitable/beneficial và lúc không.
Không cần ghi full tick log cho toàn training nếu quá tốn dung lượng.

Validation mỗi 10.240 step:
- full-trace estimated rate + relative rate aggregate/per-speed;
- QoS ratio tới 3GPP cho cả hai metric;
- HOF count, attempt count, percentage;
- PP count, successful HO count, percentage;
- RLF count, RLF seconds/percentage;
- HO attempts/time, connected time;
- p10/low-decile relative rate đã chọn trước;
- deterministic action distribution.
Reward giữa các schema khác nhau không so mean-return trực tiếp; so operational metrics.

## 11. Chọn checkpoint và quản lý đầu ra

Mỗi run thư mục riêng:
    results/models/ablation_v2/E03_ranked_k3_s17/
    results/benchmark/ablation_v2/E03_ranked_k3_s17_validation/

Cần lưu:
- experiment_config.json và schema version;
- source revision + dirty diff hash/patch;
- dataset/split/topology fingerprints;
- training_metrics.jsonl;
- validation metrics và action diagnostic summary;
- all validation checkpoints hoặc retention policy preregister;
- latest training checkpoint;
- best_qos_selected checkpoint kèm source step và selection reason;
- report với gate PASS/FAIL và parent deltas.

Không dùng tên final để ngầm chỉ cả selected model và weights cuối training. Nếu tiếp tục dùng tên tương thích, metadata phải nói rõ selected_from/selected_step/training_last_step.

Preregister cùng checkpoint rule cho mọi arm. Ưu tiên report cả budget-end checkpoint và QoS-selected checkpoint để phát hiện selection effect. Không chọn checkpoint bằng test.

Một model registry dạng Markdown hoặc JSON có cột:
ID | parent | changed factor | seed | budget | schema | status | QoS worst-speed | RR | RLF-time | HOF | PP | selected step | decision.
Các ID ở tài liệu là tên thí nghiệm, không phải các job đã được tạo.

## 12. Final test và viết bài

Khi đã khóa thiết kế, hyperparameter, normalization và selection:
- Evaluate mỗi seed đã chọn trên cùng test 150 route.
- Evaluate riêng zero-shot 120/150, không dùng để chọn model.
- So với 3GPP và các baseline causal đơn giản (best-current suitable target, trend heuristic).
- Paired bootstrap theo route; giữ phụ thuộc các policy trên cùng route.
- Báo seed variability riêng hoặc hierarchical bootstrap seed/route. Không coi primitive ticks hay cùng route lặp lại ở nhiều seed là toàn bộ mẫu độc lập.
- Báo CI và phân biệt primary endpoint đã chọn (relative rate, RLF-time) với secondary HOF/PP; không claim mọi metric cải thiện nếu không có bằng chứng.
- Báo K/candidate coverage, computation, số decision/episode, training budget và source/data provenance.
- Current legacy trace được sinh theo provenance 4 GHz/20 MHz, runtime đang 2,1 GHz/10 MHz; ghi hai cấu hình đúng, không đổi mô tả để che khác biệt.

## 13. Ước lượng ngân sách

Run lịch sử 300k mất 12.570,9 giây ~3,49 giờ, gồm validation. Suy tuyến tính:
- 100k ~1,16 giờ.
- 300k ~3,49 giờ.

Đây chỉ là mốc máy/cấu hình cũ. Candidate count, số quyết định, validation interval, history và logging làm runtime thay đổi.

- Vòng đầu candidate E00-E03: 4 x100k ~4,7 giờ.
- Nếu chạy đủ screening E00-E11: 12 x100k ~14 giờ, chưa tính audit, F0 comparison và optional arms.
- Confirmation ba cấu hình x ba seed x300k: ~31,4 giờ nếu chạy lại từ đầu.
- Tổng full core screening + confirmation ~45 giờ máy; không phải phải chạy tất cả ngay.
- Nhánh tiết kiệm: chỉ P0/P1 trước, quyết định sau số liệu; rồi P2/P3 từng arm. Có thể dừng arm bị lỗi hoặc rõ ràng dominated, nhưng báo early-stop reason và không so nó như run đủ budget.

100k là screening, không là chứng minh hội tụ. Nếu tại 100k model chưa feasible nhưng curve tiếp tục cải thiện và cơ hội candidate còn được khám phá, có thể promote để xác nhận 300k. Không loại máy móc chỉ vì chưa đạt threshold sớm.

## 14. Trình tự thực hiện ngay khi triển khai

1. P0: experiment config + log + skip-test + checkpoint contract + SMDP consistency.
2. Mở rộng train-only candidate audit và freeze ranked selector.
3. ~~Train E00, E01, E02, E03 ở 100k seed17.~~ Hoàn tất 03/10/2026.
4. ~~Chốt K* trên validation/coverage.~~ K*=3 làm nền P2; chưa phải kết luận
   hiệu năng cuối.
5. Train E04-E06 với K=3/ranked_v1, giữ reward cũ.
6. Train E07-E10, giữ K*/observation/mạng.
7. Bổ sung E11 để hoàn thành interaction 2x2.
8. Xác nhận tối đa ba cấu hình với ba seed/300k.
9. Chỉ chạy P5 nếu còn một vấn đề cụ thể cần giải thích.
10. Freeze, final test/zero-shot và viết kết luận.

Điểm quyết định đầu tiên là dữ liệu từ bốn candidate arm, không phải chuyển ngay sang bản đầy đủ đổi tất cả thành phần.
