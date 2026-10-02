# Phân tích PPO QoS và thiết kế observation/reward cho nghiên cứu

Ngày kiểm tra: 02/10/2026. Phạm vi: source hiện tại, benchmark topology109, log training và chẩn đoán chỉ đọc trên một mẫu train. Chưa thay đổi environment, reward, mạng hay checkpoint.

## 1. Bằng chứng và kết luận chính

Không đủ bằng chứng để kết luận PPO có lỗi chọn action. Mask hiện tại chỉ chặn slot không tồn tại. Trong kiểm tra 24 route train, toàn bộ 11 action đều được mở ở mọi decision. Policy học cách gần như loại bỏ C2–C10; phần lớn target ngoài C1 thực sự rất yếu trong các trạng thái được quan sát.

Checkpoint benchmark là smdp_ppo_final.pt, nhưng metadata và loader cho thấy đây là best_qos ở **280.576 macro-step**, được chọn sau training 300.000 step. Nó không phải weights cuối cùng tại step 300.000.

Benchmark hiện có gộp 150 test route tốc độ 30/50/70/90 và 60 zero-shot route 120/150, tổng 210 route:

| Chỉ số | 3GPP | PPO |
|---|---:|---:|
| Mean rate, Mbps | 22,0958 | 21,7444 |
| Mean relative rate, % | 89,5993 | 87,6750 |
| HOF, % số HO attempt | 12,4769 | 9,4564 |
| RLF-time, % | 1,7152 | 1,4321 |
| PP, % số HO thành công | 37,3237 | 37,1758 |
| HO attempt | 8.672 | 6.990 |
| HOF count | 1.082 | 661 |
| RLF count | 1.082 | 903 |
| PP count | 2.831 | 2.351 |
| STAY action | 0 | 1.588 |
| HO-C1 action | 8.677 | 7.000 |
| HO-C2 đến HO-C10 action | 0 | 0 |

PPO giảm HOF/RLF-time và số PP, nhưng PP percentage gần như đứng yên. Giảm PP count phần lớn đi cùng giảm tổng số HO. Relative rate giảm **1,9243 điểm phần trăm**, không phải 1,9243% tương đối.

Tốc độ 70 km/h chịu giảm mean relative rate lớn nhất: 87,9305% xuống 84,4451%, giảm 3,4854 điểm phần trăm. Mean rate ở nhóm này chỉ bằng 96,8294% baseline. Không nên chỉ dùng aggregate để tuyên bố QoS được giữ.

Nguồn benchmark: results/benchmark/uav_100m_grid100m_topology109_v1_s17/summary.json.

## 2. Chẩn đoán có tái lập trên train

Script: docs/research/audit_observation_reward_2026_10_02.py.
Kết quả: docs/research/observation_reward_audit_2026-10-02.json.

Chạy từ root repository:

    .\.venv\Scripts\python.exe docs/research/audit_observation_reward_2026_10_02.py

Chọn ngẫu nhiên 6 route mỗi tốc độ 30/50/70/90 bằng numpy default_rng(17), theo manifest train đã lưu, tổng 24 route. Replay hai policy: luôn HO-C1 và PPO deterministic.

Trên 939 decision do policy 3GPP đi qua:

- 100% decision mở đủ 11 action.
- Lịch sử dùng để tính feature event luôn chỉ kéo dài 40 ms.
- C1 có SINR > Qin tại 95,4207% decision.
- 97,0891% lượt candidate C2–C10 có SINR <= Qin ngay lúc decision.
- Bình quân chỉ có 0,262 candidate ngoài C1 suitable tại một decision.
- Xét SINR sau 90 ms: C1 suitable ở 88,1663% decision còn đủ dữ liệu; các lượt candidate khác suitable khoảng 2,8903%.
- 8,5197% decision có ít nhất một candidate khác có rate tức thời cao hơn C1.
- C1 có SINR > 10 dB ở 3,9404% decision; serving ở 0,1065%.

Trên 912 decision do PPO đi qua:

- PPO chọn C1 753 lần và STAY 159 lần.
- Tổng xác suất trung bình C2–C10 chỉ khoảng 0,564%; đây là tổng 9 action.
- Xác suất trung bình STAY khoảng 46,94%, C1 khoảng 52,49%. Argmax biến chênh lệch xác suất này thành 82,6% action C1.
- Thời gian tới decision tiếp theo sau STAY: median 0,62 s; p95 3,081 s; max 6,65 s.
- Thời gian macro-step sau HO: median 0,78 s; p95 5,65 s; max 22,16 s.

Xác suất thấp khác với bị mask. Deterministic không chọn C2–C10 không có nghĩa stochastic training tuyệt đối không sample chúng.

Counterfactual nhỏ: chọn tối đa bốn trạng thái mỗi route tại decision thứ 0/10/20/30 của baseline, còn ít nhất 1 giây dữ liệu. Có 86 trạng thái. Clone toàn bộ state protocol, thử cả STAY và 10 target đã freeze. Mỗi nhánh chạy đúng 100 tick = 1 giây; nếu có decision mới thì mọi nhánh cùng dùng HO-C1 làm continuation policy.

- 12/86 trạng thái có candidate ngoài C1 truyền nhiều dữ liệu hơn C1 trong 1 giây.
- Chỉ 2/86 có candidate khác vừa truyền nhiều dữ liệu hơn vừa không tăng HOF/RLF/PP count so với C1.
- 8/86 có candidate khác tốt hơn C1 theo reward cũ.
- Action thắng về dữ liệu: STAY 23, C1 56, C2 5, C7 1, C10 1.
- Action thắng theo reward cũ: STAY 48, C1 34, C5 1, C6 1, C7 1, C10 1.

Đây không phải chứng minh tối ưu toàn dataset. State được lấy trên quỹ đạo baseline, horizon chỉ 1 giây, continuation cố định; điều kiện “không tăng lỗi” chưa xét RLF-duration và không so với cả STAY. Kết quả future +90 ms chỉ dùng trong chẩn đoán, không đưa vào observation.

## 3. Nguyên nhân policy thu về C1/STAY

### 3.1. Candidate rộng nhưng phần lớn chưa suitable

Candidate builder lấy target A3, sector cùng site, mọi sector của hai site lân cận mạnh nhất, sau đó điền theo RSRP. Gần về vị trí không đồng nghĩa antenna sector đang hướng tốt tới UAV.

C1 đã được sàng lọc bởi strongest L3-RSRP và A3/TTT. Những candidate khác không được bảo đảm có SINR đủ tốt. Với xác suất đều ban đầu, C2–C10 chiếm 9/11 xác suất action. Mẫu train cho thấy phần lớn những lựa chọn này sẽ bắt đầu từ một target yếu.

SINR dưới Qin ngay lúc decision không chứng minh chắc chắn HOF sau 90 ms; có thể target đang tốt lên. Vì vậy không nên thay mọi mask thành hard SINR > Qin nếu muốn nghiên cứu predictive target selection. Nhưng cũng không nên ép policy tiếp tục chọn target yếu để tạo biểu đồ action đa dạng.

Generator hiện tại dùng reuse-1: SINR_i = P_i / (sum_j P_j - P_i + N). Nếu cùng noise/resource scaling, thứ hạng SINR tức thời đơn điệu theo received power. Metadata legacy cũng mô tả reuse-1. Điều này giải thích vì sao strongest-power là đối thủ mạnh; khác biệt còn đến từ L3 filtering, diễn biến tương lai và chi phí HO. Không nên kỳ vọng sector RSRP rất yếu đồng thời có rate tức thời rất cao trong một mô hình như vậy.

### 3.2. Lịch sử danh nghĩa dài nhưng feature chỉ dùng 40 ms

observation.py lọc pair_history từ a3_start_tic tới decision.tic. multicandidate.py làm tương tự với event_history. Sau đó trend_window_ticks=10 chỉ lấy phần cuối của chính lịch sử này.

Với TTT 40 ms, actual slope dựa trên khoảng 5 mẫu/40 ms, không phải 100 ms và càng không phải 1024 tick. 5 thống kê margin cho mỗi candidate có thể trùng thông tin đáng kể trên cửa sổ ngắn. Không có lịch sử đủ dài để phân biệt fading tức thời và xu hướng do chuyển động.

HO completion xảy ra sau khoảng 90 ms, còn PP dùng cửa sổ 1 giây. Observation đang nhìn quá ngắn so với hậu quả cần quyết định.

### 3.3. C1 không có nhiều feature riêng hơn candidate khác

Cần sửa cách diễn giải trước đó: 109 = 9 global + 10 x 10 candidate features. Khối “19 feature gốc” trộn 9 global với 10 feature C1. C2–C10 có cùng 10 loại feature riêng.

Chỉ đổi thứ tự vector thành global + các block đối xứng không tạo thêm thông tin cho MLP. Lợi ích là hợp đồng rõ hơn, dễ dùng shared scorer và kiểm thử hoán vị. Bias có thể đến từ thứ hạng/role C1, ý nghĩa slot cố định và dữ liệu, không phải do C1 được cấp 19 feature riêng.

### 3.4. Clipping làm mất độ phân giải nhưng chưa phải nguyên nhân lớn nhất

SE trong observation lấy SINR clip [-10,10] dB; rate trong reward dùng SINR thật. Candidate 12 dB và 20 dB có cùng SE feature. Delta-SE giữa hai cell có thể bị triệt tiêu.

Tuy nhiên mẫu train chỉ có khoảng 4% C1 vượt 10 dB, nên không đủ bằng chứng cho rằng đây là nguyên nhân chính gây collapse. Qin gap còn giữ một phần thông tin, nên cũng không đúng nếu nói toàn observation của hai SINR này hoàn toàn giống nhau.

### 3.5. STAY là từ chối cả event

engine.stay() latch cặp serving/target. Cùng event không được gọi lại cho tới khi rời A3, thay strongest neighbor hoặc serving thay đổi. STAY không có nghĩa “chờ 50 ms rồi xem lại”.

Việc target vẫn tốt hơn trong khi serving xấu dần có thể không tạo thêm action opportunity. Observation/reward mới có thể giúp tránh từ chối sai, nhưng không thể cấp quyền hành động mà protocol chưa cho phép.

### 3.6. Dữ liệu học của critic khó và policy mất khám phá hữu ích

Entropy log: 2,3979 tại 1.024 step; 1,6188 tại 25.600; 0,5575 tại 51.200; 0,7202 tại 300.000. ln(11)=2,3979 và ln(2)=0,6931. Entropy gần ln(2) có thể chủ yếu là random giữa C1/STAY.

Explained variance cuối training khoảng 0,0847: critic giải thích ít biến thiên target trên rollout đó. Đây là triệu chứng, không chứng minh lỗi network hay thiếu hidden units. Feature history ngắn, macro-duration biến thiên, reward event lớn và phần trạng thái không quan sát được đều có thể góp phần.

Tăng entropy chung không bảo đảm khám phá đúng candidate có triển vọng. Cần log xác suất có điều kiện: khi C2–C10 an toàn/có lợi thì policy có tăng xác suất cho chúng không.

## 4. Reward đang khuyến khích chính hành vi được quan sát

Reward cũ mỗi tick:

    r_t = dt * R_delivered(t)/100 - 5 I_RLF-start - 1 I_PP

Ở 20 Mbps, 1 giây cho +0,2. Một RLF -5 tương đương 25 giây dịch vụ 20 Mbps; một PP -1 tương đương 5 giây.

Để cân nhắc C2 thay C1, điều quan trọng là lợi ích chênh lệch, không phải tổng rate. Tăng thêm 2 Mbps trong 1 giây chỉ cho +0,02; tăng 1% xác suất RLF gây expected cost 0,05. Reward có thể hợp lý khi từ chối cơ hội tăng rate này.

Trong benchmark 210 route:

| Thành phần trung bình/episode có action | 3GPP | PPO | PPO - 3GPP |
|---|---:|---:|---:|
| Rate reward | 12,3648 | 12,1540 | -0,2109 |
| RLF penalty | -25,7619 | -21,5000 | +4,2619 |
| PP penalty | -13,4810 | -11,1952 | +2,2857 |
| Controlled return | -26,8780 | -20,5413 | +6,3368 |

Agent cải thiện objective được giao dù giảm QoS. Đây là lệch giữa mục tiêu nghiên cứu và scalar reward, không phải bằng chứng PPO không hoạt động.

Reward phạt số sự kiện. Benchmark dùng HOF/attempt, PP/success và tỷ lệ thời gian. Những đại lượng khác nhau; giảm count không buộc percentage giảm. Cần báo cáo cả count per unit time và percentage, không thưởng trực tiếp các tỷ lệ có mẫu số do policy tự thay đổi.

Một HOF dẫn tới RLF recovery. Nếu reward mới phạt cả HOF và RLF thì phải chủ ý định nghĩa chi phí cộng thêm, hoặc tách HOF khỏi RLF thuần để tránh đếm đôi ngoài ý muốn.

## 5. “Best QoS” hiện không có nghĩa bảo đảm relative rate

Checkpoint manager dùng mean_rate_PPO / mean_rate_3GPP >= 0,98, aggregate và từng speed trên validation. Đây là điều kiện chọn file, không là constraint trong loss PPO.

mean_j(R_PPO,j/R_best,j) khác mean_j(R_PPO,j)/mean_j(R_3GPP,j). Vì vậy đạt 98% mean throughput không đồng nghĩa đạt 98% mean relative rate.

Trên benchmark gộp, PPO đạt khoảng 98,41% mean throughput baseline nhưng chỉ khoảng 97,85% mean relative rate baseline. Validation-feasible cũng không bảo đảm test/zero-shot-feasible.

Bài nghiên cứu phải định nghĩa QoS cụ thể. Dataset hiện hỗ trợ estimated rate và reliability của liên kết; chưa đủ để tuyên bố bảo đảm end-to-end latency, packet loss hoặc QoS ứng dụng.

## 6. Phát biểu nghiên cứu đề xuất

“Causal, A3-triggered sector selection with PPO under empirical rate-retention constraints.”

PPO vẫn chỉ được hành động khi A3/TTT hoàn thành. Chính sách chọn chấp nhận/từ chối và target sector. Không gọi đây là chọn thời điểm HO tùy ý trước A3, hay dự đoán chính xác mọi đổi hướng ngẫu nhiên.

Với trajectory j:

    D*_j = sum_t dt R_best,j(t)
    rho_j = sum_t dt R_delivered,j(t) / D*_j

Mục tiêu: nâng rho trung bình và giảm chi phí mobility. Điều kiện nghiệm thu theo từng nhóm tốc độ v:

    mean(rho_PPO | v) >= alpha * mean(rho_3GPP | v)

alpha=0,98 là ví dụ tolerance cũ. Nếu tuyên bố vượt 3GPP về rate thì phải đánh giá chênh lệch > 0 kèm khoảng tin cậy, không đổi tên “đạt 98%” thành “vượt baseline”.

Bổ sung rate absolute, tail của rho theo route và worst-speed. Định nghĩa tail trước khi làm thí nghiệm, ví dụ p10 hoặc trung bình 10% route tệ nhất. Tail empirical không là bảo đảm cho mọi route.

## 7. Observation đề xuất: 17 global + 15 x 10 candidate = 167 chiều

Đây là đặc tả đề xuất, chưa triển khai. Vẫn feed-forward MLP PPO, không cần GRU/LSTM.

Lưu rolling history của mọi measured sector theo cell ID trong ít nhất 500 ms, độc lập với việc A3 đã bắt đầu chưa. Tại decision dùng cửa sổ [t-100 ms,t] và [t-500 ms,t]; với dt=10 ms có thể gồm 11 và 51 sample nếu tính cả hai đầu. Không nhầm độ dài window với số sample.

Nếu serving vừa đổi, history của serving mới phải lấy từ chuỗi đo cùng cell ID trong quá khứ, không nối chuỗi của nhiều serving cell thành một tín hiệu. Tương tự margin với candidate phải tính hai cell ID hiện tại trên cùng timestamp quá khứ.

### Global block: 17 feature

| # | Feature | Vai trò |
|---|---|---|
| 1 | Serving SE hiện tại | Chất lượng dịch vụ hiện tại |
| 2 | SINR_serving - Qout | Biên an toàn RLM |
| 3 | Serving SINR slope 100 ms | Biến động nhanh |
| 4 | Serving SINR slope 500 ms | Xu hướng chậm |
| 5 | Serving SINR std 500 ms | Dao động |
| 6 | Fraction SINR < Qout trong 500 ms | Tính dai dẳng của liên kết xấu |
| 7 | N310 progress | Nguy cơ bắt đầu T310 |
| 8 | N311 progress | Tiến độ phục hồi; cần expose thêm snapshot |
| 9 | T310 active | Timer có đang chạy không |
| 10 | T310 remaining time | Thời gian còn lại trước RLF |
| 11 | Serving residence time | Thời gian gắn với cell hiện tại |
| 12 | Time since last successful HO | Lịch sử HO |
| 13 | Remaining PP window | Nguy cơ quay về previous cell |
| 14 | Số HO thành công trong 1 giây gần nhất | Mobility churn |
| 15 | Best measured SE hiện tại của toàn bộ cell | Cơ hội rate đang có |
| 16 | History coverage 500 ms | Phân biệt đầu trace/lịch sử thiếu |
| 17 | Speed magnitude | Đơn vị km/h, scale cố định hoặc train-derived |

Speed là tốc độ hằng có trong tên/metadata route hiện tại. Khi mô tả khả năng triển khai phải giả định UE có đo tốc độ tương ứng; không suy ra tọa độ/hướng bay từ topology. Nếu không muốn dùng tốc độ, có ablation 166 chiều và dùng reward weight chung hoặc nêu rõ partial observability.

Feature 13 bằng max(0,1-(t-t_last_successful_HO)/1s), bằng 0 khi chưa có HO. Feature 10 bằng 0 nếu T310 inactive, đi cùng bit feature 9. Các time features phải quy định clipping/scaling và sentinel rõ trong schema.

### Candidate block: 15 feature giống nhau cho cả C1–C10

| # | Feature | Vai trò |
|---|---|---|
| 1 | Valid/present bit | Phân biệt slot không tồn tại |
| 2 | SE hiện tại | Rate tiềm năng |
| 3 | Delta-SE so với serving | Lợi ích chuyển cell |
| 4 | SINR - Qin | Biên suitable tại hiện tại |
| 5 | SINR slope 100 ms | Xu hướng nhanh |
| 6 | SINR slope 500 ms | Xu hướng chậm |
| 7 | SINR std 500 ms | Biến động |
| 8 | Fraction SINR > Qin trong 500 ms | Độ ổn định suitability |
| 9 | p10(SINR lịch sử 500 ms) - Qin | Chất lượng ở các mẫu xấu |
| 10 | RSRP_L3(candidate) - RSRP_L3(serving) | Ưu thế RSRP |
| 11 | Slope của RSRP gap trong 500 ms | Xu hướng tương đối |
| 12 | Causal projected Qin margin tại +90 ms | Chỉ số triển vọng khi HO hoàn tất |
| 13 | Is previous cell | Khả năng ping-pong |
| 14 | Is same physical site as serving | Ngữ cảnh sector |
| 15 | Is A3-trigger target | Vai trò C1 minh bạch |

Một định nghĩa thử nghiệm của feature 12:

    predicted_margin = mean_SINR_100ms
                       + 0.09 * slope_SINR_100ms
                       - kappa * std_SINR_100ms
                       - Qin

kappa=1 chỉ là điểm khởi đầu để kiểm tra trên train/validation. Đây là chỉ số dự báo thận trọng, không là xác suất success hay confidence bound được bảo đảm. Không đưa SINR thật tại t+90 ms vào observation.

Candidate đang dưới Qin nhưng slope tích cực có thể đáng cân nhắc. Feature giúp phân biệt trường hợp này với sector vừa yếu vừa xấu dần. Dự báo tuyến tính không biết UAV sẽ rẽ ngẫu nhiên; phải đo calibration/error và ablate bỏ feature 12.

### Chuẩn hóa

Không clip SINR đầu vào SE về [-10,10] trước khi biến đổi. Có thể dùng:

    SE = log2(1+10^(SINR/10))
    f_unsigned(x;s) = x/(s+x), x >= 0
    f_signed(x;s) = x/(s+abs(x))

Các scale s chọn trước hoặc fit trên train và freeze vào checkpoint. Hai SE cao khác nhau vẫn phân biệt được. Dùng cùng scale cho mọi candidate. Không fit normalization trên test; không chuẩn hóa riêng từng slot.

Đây là thiết kế nhiều feature có chủ đích, không khẳng định 167 là tối ưu. Ablation phải kiểm tra feature thừa: rate vs SINR gap, short/long trend, quantile vs suitable fraction, và global time features.

## 8. Mạng và candidate policy

Để tách ảnh hưởng observation/reward, baseline thí nghiệm đầu giữ flat MLP 64x64 như hiện tại. Input 167, 11 logits và một value head chỉ có khoảng 15.692 parameter; số feature không tự nó chứng minh mạng không đủ khả năng.

Ablation kiến trúc dùng shared MLP scorer cho mỗi candidate:

    h_i = phi(candidate_i)
    pooled = masked_mean_i(h_i)
    logit_i = psi(global, h_i, pooled)
    logit_STAY = psi_stay(global, pooled)
    V = value(global, pooled)

phi/psi đều là MLP dùng chung trọng số, vẫn tối ưu bằng PPO, không phải LSTM. Actor logits theo candidate phải permutation-equivariant; critic/pooling invariant. Đây là một thiết kế áp dụng nguyên lý học trên tập từ Deep Sets, không thể chỉ reshape flat MLP rồi tuyên bố đã có invariant/equivariant.

Với flat MLP, có thể thử permutation augmentation C2–C10 nhưng phải remap observation, action, old log-prob và mask nhất quán trong transition. Không đảo slot ngẫu nhiên giữa lúc sample và lúc PPO update.

Giữ structural mask để làm đối chứng: chỉ thiếu cell/mất measurement mới bị chặn. Nếu nghiên cứu suitability mask thì tách thành ablation; không dùng future SINR hoặc bắt mọi candidate phải đạt A3. Không thưởng vì chọn C2, không phạt vì chọn C1.

So sánh K=3/5/10, nhớ K là tổng candidate, action count là K+1. Cùng candidate-construction rule hoặc công bố rõ rule theo từng K. Nếu không có đủ cơ hội alternative thì giảm K có thể tốt hơn tăng mạng.

## 9. Reward đề xuất: QoS rõ ràng và mobility cost có đơn vị diễn giải

### 9.1. Tránh nhầm mean instantaneous ratio với trajectory relative rate

mean_t(R(t)/R_best(t)) nói chung KHÔNG bằng sum_t R(t)/sum_t R_best(t). Nếu paper báo đại lượng thứ hai, chỉ thay rate reward bằng instantaneous ratio chưa tạo objective khớp metric.

Chọn T0=60 giây tham chiếu và D*_j như trên. Reward QoS mỗi tick:

    u_j,t = T0 * dt * R_delivered,j(t) / D*_j

Tổng full trace bằng T0*rho_j. Đoạn trước decision đầu là hằng theo policy và không được gán cho action; vẫn cộng vào full-trace QoS để nghiệm thu. Nếu D*_j=0 phải xử lý explicitly như trace không có dịch vụ, không dùng epsilon để biến nó thành mẫu “QoS tốt”.

D*_j là thống kê oracle toàn trace, độc lập action, được precompute trong môi trường huấn luyện mô phỏng. Nó chỉ scale reward; không được đưa vào actor/critic observation. Đây là quyền thông tin của training simulator, cần công bố. Không claim reward online hoàn toàn causal. Khi inference chỉ cần observation causal; test dùng dữ liệu tương lai để tính kết quả sau rollout, không để chọn action.

Nếu yêu cầu cả reward online causal, dùng fixed R_ref fit từ train:

    u_t = dt * R_delivered(t)/R_ref

Khi đó objective là delivered data, không trùng chính xác mean trajectory relative rate; phải giữ QoS constraint/selection ngoài scalar reward và ghi rõ thay đổi này.

### 9.2. Chi phí mobility

Đặt I_HOF là tick một HO execution thất bại; I_RLFpure là RLF-start không đồng thời với HOF; I_PP là ping-pong detected; I_HO là HO attempt bắt đầu execution (đồng bộ định nghĩa benchmark). I_rec là tick RLF recovery active.

    c_j,t = (T0/T_j) * [
        w_time * dt * I_rec
        + w_HOF * I_HOF
        + w_RLF * I_RLFpure
        + w_PP * I_PP
        + w_HO * I_HO
    ]

T_j là full episode duration, với dataset hiện tại T0/T_j=1.

Phạt HOF và RLF thuần tách nhau để một event không nhận hai event cost không chủ ý. RLF recovery time vẫn bị phạt: event severity và duration là hai tác hại khác nhau. HO execution đã giảm delivered data; không bắt buộc thêm một disconnection penalty chung, tránh phạt cùng interruption quá nhiều lần.

Một bộ giá trị KHỞI ĐẦU có thể preregister, chưa phải kết quả tối ưu:

| Hệ số | Giá trị | Diễn giải khi rate đạt oracle |
|---|---:|---|
| w_time | 1,0 | Mỗi giây recovery chịu thêm cost 1 |
| w_HOF | 0,2 | Một HOF có extra cost bằng 0,2 giây dịch vụ lý tưởng |
| w_RLF | 0,2 | Một RLF thuần có extra cost tương tự |
| w_PP | 0,1 | Một PP tương đương 0,1 giây dịch vụ lý tưởng |
| w_HO | 0 ban đầu | Không cản HO cần thiết; ablation có thể dùng 0,005 |

Không tune các số này để làm đẹp test cũ. Chốt khoảng sensitivity hẹp trên train/validation, rồi freeze. Reward không trực tiếp phạt “chọn non-C1”, “SINR hiện tại thấp”, hay “ít đa dạng action”.

### 9.3. QoS-floor trong tối ưu

Reward cố định đầu tiên:

    r_t = u_t - c_t

Muốn tăng áp lực khi vi phạm rate-retention, dùng PPO với hệ số Lagrange theo speed v:

    r_t = (1 + lambda_v) * u_t - c_t

    lambda_v <- clip(
        lambda_v + eta_lambda * (
            alpha * mean_train_reference_rho_v
            - mean_train_policy_rho_v
        ),
        0, lambda_max
    )

Công thức là scalar hóa của maximize T0*E[rho]-E[cost] với constraint E[rho|v] >= alpha*E[rho_3GPP|v]; hằng số baseline trong Lagrangian không ảnh hưởng gradient policy.

Ước lượng shortfall từ full completed training episodes, theo speed; cập nhật lambda khi đủ mẫu, dùng EMA và giữ lambda cố định trong một rollout/PPO update. Không dùng test hoặc zero-shot để cập nhật lambda. Speed observation giúp critic diễn giải reward khác theo nhóm. Validation chỉ để chọn model và đánh giá deterministic deployment; nếu train stochastic feasible mà deterministic không đạt thì checkpoint vẫn bị loại.

Đây là constrained-objective PPO theo thực nghiệm. Không tự nhận được bảo đảm lý thuyết của CPO và không bảo đảm QoS từng UE/từng trace. Nếu lambda đạt trần mà QoS vẫn không đạt, phải báo chưa tìm được policy feasible, không gắn nhãn best_qos cho best effort.

Checkpoint selection cần dùng đúng mean relative rate và thêm guard mean absolute rate nếu cần. Khi có nhiều feasible checkpoint, chọn theo mobility metrics đã định nghĩa trước. Nếu mục tiêu là vượt 3GPP thì rate improvement phải được kiểm định riêng.

## 10. SMDP và discount phải nhất quán

Code hiện tại R_k = sum primitive rewards, nhưng bootstrap dùng gamma_per_second^tau. Với gamma<1, tổng objective phụ thuộc vị trí chia macro boundary; reward ở cuối một macro rất dài không bị discount trong macro. Việc dùng discount theo duration ở bootstrap chưa đủ để đạt discount theo thời gian cho mọi reward.

Nếu nghiên cứu finite-horizon 60 giây và mục tiêu là tổng QoS/cost cả route, có thể dùng gamma=1 làm cấu hình chính. Khi đó aggregation không phụ thuộc số A3 boundary và sum u=T0*rho đúng với objective không discount. Duration vẫn cần cho GAE lambda_per_second^tau, các timer và thống kê.

Nếu giữ gamma_per_second=0,99, phải dùng:

    R_k^gamma = sum_l gamma_per_second^(l*dt) * r_(k,l)
    bootstrap_discount = gamma_per_second^tau_k

với quy ước tick index nhất quán. Đây là objective ưu tiên sớm hơn theo thời gian, không hoàn toàn trùng full-route relative rate không discount. Dùng như ablation, không thay gamma để ép model chọn action mong muốn.

Không chia macro reward cho macro-duration nếu mục tiêu là tổng dữ liệu và tổng chi phí theo thời gian; điều đó đổi bài toán và có thể ưu ái policy tạo nhiều decision.

## 11. Giới hạn của STAY và trigger

Bản obs/reward mới đầu tiên nên giữ nguyên STAY-reject để tách ảnh hưởng. Nhưng cần một ablation riêng nếu mục tiêu nghiên cứu bao gồm “chờ target tốt lên rồi HO trong cùng A3”.

Ví dụ biến thể A3-gated redecision: sau STAY, nếu A3 còn active thì cho quyết định lại sau 100 ms; không mở action trước A3. Đây là đổi hợp đồng action/trigger và cần benchmark riêng, không gộp vào kết quả “chỉ thay observation/reward”.

Với STAY-reject hiện tại, agent không thể chọn bất kỳ thời điểm HO nào nó muốn. Một observation tốt hơn không khắc phục hạn chế quyền điều khiển này.

## 12. Kế hoạch kiểm chứng cho bài nghiên cứu

Giai đoạn A: mở rộng audit trên TRAIN (không dò test để tune).

- Replay các state của baseline và PPO, stratified theo speed, Qin gap, T310, previous-cell flag và alternate opportunity.
- Counterfactual cùng state, cùng fixed horizon 0,5/1/2 giây, cùng continuation policy. Đo thêm recovery duration, không chỉ count.
- Thống kê HOF của từng candidate, probability mass trên candidates suitable, probability theo lợi ích kỳ vọng và regret so với best branch.
- “Oracle future” chỉ là upper bound/chẩn đoán, không là policy deployable. Không đưa counterfactual off-policy branches trực tiếp vào PPO on-policy buffer.
- Xác định cơ hội alt có thể nhận biết từ lịch sử hay chỉ từ tương lai ngẫu nhiên.

Giai đoạn B: ablation có kiểm soát.

| Mã | Observation | Reward | Action/trigger |
|---|---|---|---|
| B0 | 109 hiện tại | cũ | hiện tại |
| B1 | 167 causal history | cũ | hiện tại |
| B2 | 109 hiện tại | mới | hiện tại |
| B3 | 167 | mới | hiện tại |
| B4 | 167 | mới + QoS dual | hiện tại |
| B5 | 167 + shared MLP scorer | như B4 | hiện tại |
| B6 | như bản được chọn trên validation | như bản chọn | A3-gated redecision |

Discount consistency là kiểm tra objective chung; công bố cấu hình gamma của mọi arm. Khi so sánh architecture, thêm đối chứng cùng parameter budget nếu kết luận lợi ích đến từ structure thay vì kích thước.

Baseline thêm các chính sách causal đơn giản: HO-C1; chọn suitable candidate có SINR hiện tại cao nhất; trend-based target score. Không chỉ so với một baseline để tuyên bố DRL là nguyên nhân cải thiện.

Chạy ít nhất 3 seed, ưu tiên 5. Giữ dataset split cố định, normalization fit chỉ train. Báo cả macro-steps, primitive time và số episode vì cùng 100k macro-step có thể dùng lượng dữ liệu vật lý khác nhau.

Giai đoạn C: đánh giá.

- Tách test 30–90 khỏi zero-shot 120/150, không chỉ gộp aggregate.
- Paired comparison trên cùng route, bootstrap/CI ở cấp route; không coi primitive ticks tương quan là các mẫu độc lập.
- Báo biến thiên giữa seed; candidate diversity chỉ là diagnostic, không là mục tiêu chính.
- Report rate, relative rate, per-speed, tail, HOF count/time + percentage, PP count/time + percentage, RLF-time và HO attempts/time.
- Audit C1/STAY/C2+ conditional probabilities, histogram macro-duration và reward breakdown.
- Test hiện tại đã được dùng chẩn đoán nhiều lần; để tuyên bố kết quả cuối cùng, nên có bộ route/seed giữ riêng chưa dùng tune. Không cần tạo lại toàn bộ dataset để thử thiết kế này.

Tiêu chí thành công: QoS-floor đạt trên mỗi nhóm test đã định trước, mobility metrics cải thiện có ý nghĩa, và nếu tuyên bố target selection hữu ích thì phải chứng minh các lần chọn non-C1 có lợi có điều kiện. Không yêu cầu agent phân phối action đồng đều.

## 13. File source liên quan

- src/ho_optim_drl/smdp/observation.py:106,254,274,384 — feature gốc, A3-only history, clipping.
- src/ho_optim_drl/smdp/multicandidate.py:17,127,158,165 — candidate feature/history/mask/selection.
- src/ho_optim_drl/smdp/reward.py:45,105 — scale và event reward.
- src/ho_optim_drl/smdp/transition.py:160 — macro aggregation và duration discount.
- src/ho_optim_drl/protocol/engine.py:125,235 — STAY reject latch; :314 — rate không clip.
- src/ho_optim_drl/agents/ppo/network.py — flat MLP và equal initial logits.
- src/ho_optim_drl/agents/ppo/trainer.py:176,309 — QoS checkpoint gating, không là loss constraint.
- src/ho_optim_drl/evaluation/benchmark.py:172 — trajectory relative-rate definition.

Các diễn giải historical radio cần giữ đúng: dataset provenance ghi generator 4 GHz/20 MHz; runtime đang dùng 2,1 GHz/10 MHz theo lựa chọn trước của chủ dự án. Runtime không tái sinh channel. Paper phải nêu hai cấu hình rõ, không gọi trace cũ là trace đã phát ở 2,1 GHz chỉ vì Config.fc đổi.

## 14. Tài liệu phương pháp

- Achiam et al., Constrained Policy Optimization, ICML 2017: https://proceedings.mlr.press/v70/achiam17a.html — cơ sở phân biệt mục tiêu reward và constraint. Thiết kế PPO ở đây không phải thuật toán CPO.
- Zaheer et al., Deep Sets, NeurIPS 2017: https://arxiv.org/abs/1703.06114 — cơ sở cho shared candidate encoder và aggregation theo tập.
- Sutton, Precup, Singh, Between MDPs and semi-MDPs, 1999: https://doi.org/10.1016/S0004-3702(99)00052-1 — khung temporal abstraction/SMDP.

Các feature, weight ví dụ và lộ trình thí nghiệm trong báo cáo là đề xuất cho project, không phải cấu hình đã được các bài trên chứng minh tối ưu.
