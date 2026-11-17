# Báo cáo tiến độ nghiên cứu tối ưu handover bằng PPO

**Thời điểm chốt:** 13/10/2026  
**Branch hiện tại:** `codex/early-ho-timing`

## 1. Mục tiêu

Nghiên cứu xây dựng tác tử PPO điều khiển handover cho UAV nhằm duy trì QoS
gần hoặc tốt hơn 3GPP, đồng thời giảm handover failure (HOF), ping-pong (PP)
và thời gian RLF recovery. Dữ liệu chỉ cung cấp trajectory radio; toàn bộ A3,
RLM, handover execution, HOF, PP và RLF được mô phỏng trong environment của
project.

Cấu hình radio chính được giữ cố định: carrier 2,1 GHz, bandwidth 10 MHz và
primitive timestep 10 ms. Thí nghiệm hiện tại dùng bốn tốc độ 30/50/70/90
km/h, split seed 17: 700 TRAIN, 150 validation và 150 test route. Test và
zero-shot chưa được dùng để chọn mô hình.

## 2. Quá trình thu hẹp bài toán

### Candidate, observation và reward ablation

- P1 kiểm tra K=10/5/3 candidate. K=3 vẫn giữ ít nhất một alternative hữu ích
  trong 100% positive state của audit TRAIN, nhưng PPO chủ yếu hội tụ về C1 hoặc
  STAY. K=3 được giữ vì gọn, không phải vì đã chứng minh tốt hơn.
- P2 thay đổi history, normalization và observation QoS đầy đủ. Observation
  giàu hơn không tự giải quyết C1/STAY collapse.
- P3 thay đổi reward QoS/event/recovery/dual. Các arm vẫn thường quay về C1
  hoặc STAY; thay reward đơn thuần chưa giúp học target alternative ổn định.

### Diagnostic D63

D63 chứng minh trên TRAIN rằng cơ hội tốt hơn 3GPP có tồn tại nếu dùng
finite-lookahead reference:

| Policy reference | RR | HOF | RLF-time | PP |
|---|---:|---:|---:|---:|
| Always-C1 | 89,868% | 11,800% | 1,576% | 36,065% |
| Timing-only reference | 91,294% | 3,965% | 0,321% | 19,464% |
| Timing×target reference | 91,606% | 4,431% | 0,338% | 19,405% |

Như vậy phần lớn dư địa đến từ **chọn thời điểm**, còn lợi ích bổ sung của chọn
target nhỏ hơn. Tuy nhiên observation nhân quả khi đó chưa dự đoán alternative
an toàn đủ chính xác; G2 thất bại và nhánh timing×target được đóng.

## 3. Hướng nghiên cứu hiện tại: timing-only sau A3

Environment hiện trao quyền quyết định sau khi A3 đã đủ TTT:

- `HO_C1`: handover ngay đến target A3;
- `STAY`: trì hoãn 20 ms rồi đánh giá lại.

Do đó đây là bài toán **HO_NOW so với WAIT_20 sau completed A3**, chưa phải PPO
handover trước A3 hoặc tự chọn target C2/C3. Cách thu hẹp này giữ target giống
3GPP và cô lập đúng biến timing.

Một luật nhân quả đơn giản được kiểm tra trên 40 TRAIN confirmation routes mới:
WAIT 20 ms nếu target SINR thấp hơn serving SINR, ngược lại HO ngay. Luật chọn
WAIT 14,69% và đạt strict gate:

| Metric | B0/3GPP | Luật SINR | Chênh lệch |
|---|---:|---:|---:|
| Relative rate | 90,2143% | 90,4168% | +0,2025 điểm % |
| HOF | 9,4176% | 8,4801% | -0,9375 điểm % |
| PP | 38,0301% | 35,4954% | -2,5347 điểm % |
| RLF-time | 1,2667% | 1,0833% | -0,1833 điểm % |

Bootstrap CI95% của cả bốn chênh lệch đều có dấu thuận lợi. Đây là bằng chứng
TRAIN-only rằng timing post-TTT có thể cải thiện đồng thời QoS và reliability.

## 4. PPO timing pilot

Hai pilot dùng cùng action, observation causal T4D.1 gồm 19 feature, reward
`trajectory_qos_recovery_reward_v1`, MLP 64×64 và 10.240 macro-step.

| Metric validation | 3GPP | T5 | T5A fixed-scale |
|---|---:|---:|---:|
| Rate | 23,0245 Mbps | 21,9673 Mbps | **23,0355 Mbps** |
| Relative rate | 91,2718% | 86,2701% | **91,3440%** |
| RR/3GPP | 100% | 94,520% | **100,079%** |
| HOF | 10,9031% | 7,7778% | **10,2440%** |
| PP | 37,0569% | 32,0209% | **33,3333%** |
| RLF-time | 1,3130% | 0,6647% | **1,1260%** |
| STAY | 0% | 95,3299% | **23,7986%** |

T5 không đạt QoS vì policy gần 50/50 nhưng deterministic argmax khuếch đại
một bias rất nhỏ thành 95,3% STAY. Diagnostic tìm thấy critic gradient lớn hơn
actor hàng trăm lần và làm gradient clipping chung gần như triệt tiêu actor.

T5A chỉ thay critic target normalization `fixed_scale_v1`, scale 24,5694 được
fit từ 80 TRAIN routes. Kết quả:

- critic gradient update cuối giảm từ 22,111 xuống 0,186;
- clipping factor tăng từ 0,0226 lên 1,0;
- actor gradient tăng từ 0,0713 lên 0,3077;
- QoS đạt trên 98% so với 3GPP ở aggregate và từng tốc độ;
- RR, HOF, PP và RLF-time đều tốt hơn 3GPP trên 150 validation routes.

T5A là kết quả triển vọng nhất hiện tại. Tuy nhiên entropy vẫn gần `ln(2)` và
`P(HO)` trung bình chỉ khoảng 0,503. Vì vậy chưa thể tuyên bố policy đã hội tụ;
deterministic argmax vẫn đang quyết định từ các probability margin nhỏ.

## 5. Trạng thái và bước tiếp theo

Runner T5A-100k đã được chuẩn bị nhưng chưa chạy. Nó giữ nguyên T5A, validation
mỗi 10.240 step và ghi song song:

- deterministic validation để chọn checkpoint;
- stochastic validation với seed 17/29/43 chỉ để chẩn đoán;
- RNG được phục hồi nên stochastic validation không làm đổi rollout training;
- không chạy test hoặc zero-shot.

T5A chỉ được promote nếu kết quả 100k thỏa đồng thời:

1. QoS ≥98% aggregate và từng tốc độ;
2. HOF, PP và RLF-time không xấu hơn 3GPP;
3. p10 relative rate không suy giảm nghiêm trọng;
4. checkpoint tốt không chỉ xuất hiện tại một mốc rồi collapse;
5. probability action thay đổi theo state và deterministic–stochastic không
   mâu thuẫn lớn.

Nếu đạt gate, bước sau là xác nhận nhiều training seed rồi mới mở test đúng một
lần. Nếu QoS đạt nhưng probability vẫn sát 0,5 hoặc policy không ổn định giữa
các checkpoint, ưu tiên actor–critic tách riêng hoặc cải thiện objective học,
không kết luận PPO đã học timing chỉ từ deterministic validation.

## 6. Kết luận tạm thời

Các ablation trước cho thấy mở rộng candidate, observation hoặc reward chưa đủ
để PPO học đồng thời timing và target. Audit vật lý và luật SINR xác nhận timing
post-TTT là hướng có tín hiệu rõ hơn. Sau khi sửa mất cân bằng critic, T5A lần
đầu đạt QoS validation và đồng thời giảm HOF, PP, RLF-time so với 3GPP. Kết quả
này đủ để mở screening 100k có kiểm soát, nhưng chưa đủ cho claim cuối bài báo
cho tới khi tính ổn định, nhiều seed và held-out test được xác nhận.

