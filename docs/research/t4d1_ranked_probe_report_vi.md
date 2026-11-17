# Báo cáo T4D.1 material-gain ranking probe

## Mục tiêu và thay đổi duy nhất

T4D.1 giữ nguyên frozen TRAIN states và compact causal observation của T4D. Thay đổi cách học:

- QoS head phân loại `delta_delivered_mbit >= 0,05 Mbit`.
- Pairwise ranking loss ép state có material gain đứng trên state không có material gain trong cùng speed.
- Risk head dự đoán riêng việc WAIT làm xấu HOF/PP/RLF/recovery/interruption.
- Risk temperature được khóa trên tuning.
- Selector cần tối thiểu 10 WAIT trên tuning, tránh no-op thắng mặc định.

Không thay protocol, action, reward, trigger hoặc dataset. PPO không được chạy.

## Chống leakage

- Fit được chia tiếp theo route và speed thành inner-train/inner-validation.
- Inner-validation chỉ khóa số epoch.
- Sau khi khóa epoch, model được khởi tạo lại và refit toàn bộ fit.
- Tuning chỉ calibration risk và khóa selector thresholds.
- Confirmation không tham gia fit, epoch selection, calibration hoặc threshold selection.
- Validation/test/zero-shot rows đọc: `0`.

## Pilot seed 29

- Inner train/validation: `1628/405` state.
- Epoch khóa: `500`.
- Risk temperature: `1,5`.
- QoS score threshold: `0,563895`.
- Risk threshold: `0,9`.
- Tuning: 101 WAIT, mean gain `+0,03042 Mbit/state`; có 11 threshold pair reliability-feasible.

Trên confirmation 514 state:

- Material AP/prevalence: `0,2344/0,0973`.
- Precision/recall: `18,64%/44,00%`.
- Risk AP/prevalence: `0,1368/0,0564`.
- Risk Brier trước/sau calibration: `0,1986/0,2028`; calibration không transfer.
- Selector: `115` WAIT, trong đó 17 material-safe, 98 non-meaningful.
- Mean QoS gain: `+0,00109 Mbit/state`.
- QoS gain bootstrap CI95%: `[-0,03582; +0,03794]`.
- Mean HOF, PP, RLF, recovery và interruption đều giảm, nhưng local utility gate vẫn fail vì QoS CI cắt 0.

Theo speed, material ranking tổng quát tốt nhất ở 30 km/h (AP 0,46) và giảm dần đến 90 km/h (AP 0,22). Theo nguồn state, B3 yếu nhất (AP 0,21).

## So với reference vật lý

So với rule `WAIT nếu target SINR < serving SINR`, T4D.1 có mean QoS chênh `-0,02535 Mbit/state`; CI95% `[-0,05069; +0,00052]`. Model chưa vượt rule và có xu hướng kém hơn.

## Gate

- Material ranking: pass.
- Risk signal và calibration: fail.
- Local utility: fail.
- Vượt simple rule: fail.
- Overall local probe: `NOT_PASSED`.
- PPO: `CLOSED`.

## Kết luận

Ranking loss giải quyết được no-op và tăng recall, nhưng không giải quyết được precision/utility. Việc tiếp tục tăng capacity hoặc entropy sẽ chủ yếu tạo thêm false WAIT. Kết quả cho thấy bottleneck hiện tại nằm ở định nghĩa outcome 1 giây và khả năng quan sát counterfactual, không còn đơn thuần ở kiến trúc classifier.

Bước tiếp theo nên là audit nhãn/horizon trên TRAIN: so sánh 0,5/1/2 giây, kiểm tra tính ổn định dấu của gain và xác định state nào đổi nhãn theo horizon. Đồng thời chạy full-route TRAIN audit cho simple SINR rule để kiểm tra local gain có tồn tại khi WAIT làm thay đổi state distribution hay không. Chưa mở PPO.

Artifact: `results/analysis/early_ho_timing/t4d1_ranked_probe_v1`.
