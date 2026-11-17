# Audit nguồn lợi ích T5A và cơ hội timing trước/sau A3

## Phạm vi và hợp đồng

Audit này chỉ dùng 80 route TRAIN đã khóa trong T2, gồm 20 route cho mỗi tốc
độ 30/50/70/90 km/h. Ba checkpoint T5A 100k (seed 17, 29, 43) được replay
deterministic trên đúng route set. Không đọc validation, test hoặc zero-shot.

Hai loại bằng chứng được giữ tách biệt:

- **Hành vi thực tế của T5A:** kết quả full-route khi policy thật chọn STAY/HO_C1.
- **Cơ hội oracle:** counterfactual T2/T3 có dùng tương lai để tạo nhãn offline;
  tương lai không phải observation của policy.

## Kết quả T5A thực tế trên TRAIN

So với B0/3GPP Always-HO trên cùng route:

| Seed | Δ relative rate (điểm %) | 95% CI | Δ HOF (điểm %) | Δ PP (điểm %) | Δ RLF-time (điểm %) | STAY |
|---|---:|---:|---:|---:|---:|---:|
| 17 | +0.0909 | [+0.0305, +0.1474] | -1.5309 | -4.1848 | -0.3206 | 28.85% |
| 29 | -0.1856 | [-0.2911, -0.0892] | -0.8065 | -1.2873 | -0.1979 | 41.63% |
| 43 | -0.1540 | [-0.2445, -0.0688] | -1.3955 | -5.9931 | -0.3704 | 54.42% |

CI của HOF, PP và RLF-time đều nằm phía cải thiện cho cả ba seed. Tuy nhiên,
relative rate chỉ tăng ở seed 17; seed 29 và 43 mất rate có ý nghĩa thống kê.
Trung bình ba seed, ΔRR khoảng -0.083 điểm %, trong khi HOF giảm 1.244 điểm %,
PP giảm 3.822 điểm % và RLF-time giảm 0.296 điểm %.

Nguồn trực tiếp dễ thấy nhất là giảm số HO:

- Seed 17: 3.212 → 2.941 HO attempts; HOF 379 → 302; PP 1.021 → 841.
- Seed 29: 3.212 → 3.011 HO attempts; HOF 379 → 331; PP 1.021 → 931.
- Seed 43: 3.212 → 2.797 HO attempts; HOF 379 → 291; PP 1.021 → 753.

Phân loại route cho thấy seed 17 có 46/80 route cải thiện Pareto, nhưng seed 29
và 43 có nhiều route thuộc nhóm “tăng reliability nhưng đổi lấy rate”. Vì vậy,
lợi ích hiện tại chủ yếu là **lọc bớt HO rủi ro/không cần thiết**, chưa phải một
quy luật timing tối ưu ổn định qua seed.

Policy cũng chưa có biên quyết định mạnh: 95,65% đến 99,98% quyết định có
`|P(HO)-P(STAY)| <= 0,1`. Tỷ lệ STAY rất khác nhau giữa seed. Đây là dấu hiệu
boundary timing còn phẳng và nhạy với seed.

## Cơ hội counterfactual trước và sau A3

Ở horizon 1 giây, một opportunity chỉ dương khi một schedule tăng delivered data
và không làm xấu HOF, PP, RLF, recovery hoặc interruption so với continuation
control tại cùng anchor.

| Vùng quyết định | Anchors | Opportunity an toàn dương | Trung bình gain trên mọi anchor |
|---|---:|---:|---:|
| Trước A3 + đang chạy TTT | 654 | 90,06% | +1,1287 Mbit |
| Sau A3+TTT | 545 | 34,13% | +0,6140 Mbit |

Bootstrap theo route, phân tầng tốc độ:

- Chênh lệch tỷ lệ opportunity pre − post: **+55,93 điểm %**, 95% CI
  `[+50,58; +61,19]`.
- Chênh lệch mean gain pre − post: **+0,5148 Mbit**, 95% CI
  `[+0,2944; +0,7402]`.

Đây không phải xác suất policy sẽ thành công: bank có tối đa 5 anchor/source/route
và các source B0/B2/B3 có thể cùng đại diện một physical state. Kết quả chỉ nói
rằng route TRAIN chứa nhiều cơ hội timing trước khi A3+TTT hoàn tất hơn cơ hội
trì hoãn sau TTT.

Đối chứng full-route oracle T3 nhất quán về rate: `Rearly - Rpost` tăng RR
+0,4408 điểm % với 95% CI `[+0,3808; +0,5022]`. Tuy nhiên CI của HOF, PP và
RLF-time đều cắt 0, nên **chưa chứng minh early timing làm reliability tốt hơn
post-TTT oracle**.

## Kết luận nghiên cứu

1. T5A chứng minh post-TTT timing có thể giảm HOF/PP/RLF, nhưng cải thiện RR
   không ổn định qua seed.
2. Phần lớn lợi ích T5A hiện tại đến từ giảm HO attempts, không phải chọn chính
   xác một thời điểm có margin rõ ràng.
3. TRAIN counterfactual cho thấy còn nhiều cơ hội trước A3/đang chạy TTT, đặc
   biệt cho relative rate. Vì vậy mở cửa quyết định sớm là hướng tiếp theo hợp lý.
4. Không nên chỉ hạ cứng A3 offset. B1 đã cho thấy đổi trigger cố định không thay
   thế được policy timing. Nên giữ shadow A3 chuẩn làm mốc/reference, mở causal
   early window, và để agent chọn WAIT/HO_C1 dưới safety guard.

Gate đề xuất cho bước kế tiếp: trước hết chạy pilot timing-only pre/post chung với
K=1 và nhiều seed; chỉ quay lại target selection khi RR tăng ổn định và HOF/PP
không xấu trên TRAIN confirmation. Chưa mở held-out test cho nhánh mới ở giai
đoạn này.

## Artifact tái lập

- Kết quả chính: `results/analysis/early_ho_timing/t5a_benefit_source_main_v1/main_report.json`
- Decision log: `results/analysis/early_ho_timing/t5a_benefit_source_main_v1/t5a_decisions.jsonl`
- Outcome theo route: `results/analysis/early_ho_timing/t5a_benefit_source_main_v1/route_outcomes.jsonl`
- Opportunity labels: `results/analysis/early_ho_timing/t5a_benefit_source_main_v1/opportunity_states.jsonl`
- Provenance: `results/analysis/early_ho_timing/t5a_benefit_source_main_v1/run_manifest.json`
