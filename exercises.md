# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature ở mức thấp (0.0 - 0.5), phản hồi mang tính xác định cao, câu từ gãy gọn, tập trung vào sự kiện lịch sử/địa lý nổi tiếng và hầu như không đổi qua các lần gọi. Khi temperature tăng lên (1.0 - 1.5), mô hình dùng nhiều từ ngữ gợi mở, sáng tạo và đa dạng hơn nhưng câu từ bắt đầu dài dòng, dễ pha trộn cấu trúc ngoại ngữ hoặc suy diễn thiếu chính xác.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Nên đặt temperature thấp trong khoảng 0.1 đến 0.3 (tối đa 0.5). Chatbot chăm sóc khách hàng cần ưu tiên tính chính xác, nhất quán về chính sách/sản phẩm và tuyệt đối tránh bịa đặt thông tin (hallucination); tính sáng tạo bay bổng không phù hợp cho hỗ trợ nghiệp vụ.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o đắt hơn GPT-4o-mini xấp xỉ 16.7 lần (0.010 so với 0.0006 USD/1K token output). Với 10,5 triệu token output mỗi ngày, GPT-4o-mini chỉ tốn ~6.3 USD/ngày (~190 USD/tháng) trong khi GPT-4o tiêu tốn tới ~105 USD/ngày (~3.150 USD/tháng). Nên dùng GPT-4o cho các bài toán lập trình phức tạp, suy luận pháp lý hoặc phân tích tài chính nhiều bước; nên dùng GPT-4o-mini cho chatbot CSKH tra cứu FAQ, phân loại email hoặc tóm tắt tài liệu diện rộng.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với vai giáo viên tiểu học, phản hồi dùng giọng điệu thân mật, câu văn ngắn, sử dụng hình ảnh ẩn dụ quen thuộc (như cuốn sổ nhật ký chung của cả lớp mà không ai tẩy xóa được) và tránh thuật ngữ kỹ thuật. Với vai chuyên gia tài chính, phản hồi tập trung vào khái niệm sổ cái phân tán (distributed ledger), cơ chế đồng thuận mật mã học (cryptographic consensus) và cấu trúc mạng ngang hàng P2P. System prompt đóng vai trò như một bộ lọc ngữ cảnh mạnh mẽ, định hình toàn bộ không gian từ vựng, văn phong và mức độ chi tiết trước khi mô hình bắt đầu sinh câu trả lời.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Đoạn văn tiếng Việt 100 từ ước tính thô ra ~133 token, nhưng khi đếm bằng tiktoken thực tế thường lên tới 150 - 170 token, chênh lệch khoảng 15% đến 25%. Tiếng Việt tốn nhiều token hơn tiếng Anh vì các bộ tokenizer BPE (Byte Pair Encoding) được huấn luyện chủ yếu trên văn bản tiếng Anh; đối với tiếng Việt, các từ ghép và nguyên âm có dấu thanh (như ơ, ư, ế, ặ, ỗ) thường bị băm thành nhiều byte hoặc subword lẻ thay vì nằm trọn trong 1 token duy nhất.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng hội thoại tương tác trực tiếp với con người (như chat UI, coding copilot) vì nó tối ưu thời gian phản hồi đầu tiên (Time to First Token - TTFT), mang lại cảm giác mượt mà và giúp người dùng đọc trước trong lúc AI tiếp tục sinh câu trả lời. Ngược lại, non-streaming phù hợp hơn cho các tác vụ ngầm phía backend (batch processing, data pipeline trích xuất JSON, kiểm thử tự động hoặc giao tiếp giữa các microservices) nơi hệ thống chỉ cần kết quả hoàn chỉnh có cấu trúc để chuyển tiếp sang bước xử lý tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng gấp đôi thời gian chờ sau mỗi lần thử (0.1s -> 0.2s -> 0.4s), tạo ra khoảng lặng giãn cách cần thiết cho máy chủ có thời gian xả tải và phục hồi tài nguyên. Nếu hàng nghìn client cùng retry với khoảng thời gian cố định 1 giây, tất cả các request thất bại sẽ đồng loạt ập vào server cùng một thời điểm ở giây tiếp theo, gây ra hiện tượng "cơn bão retry" (thundering herd problem) khiến hệ thống máy chủ vừa khởi động lại đã ngay lập tức bị nghẽn mạng và sập tiếp.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona: "Bạn là trợ giảng thân thiện của khóa học AI20K, giải thích ngắn gọn, chuẩn xác bằng tiếng Việt và khích lệ người học." Cụm từ "ngắn gọn" giúp hạn chế tiêu tốn token đầu ra và duy trì nhịp tương tác nhanh trên terminal CLI; chỉ định "bằng tiếng Việt" đảm bảo mô hình không tự ý nhảy sang tiếng Anh khi gặp các thuật ngữ lập trình; từ "khích lệ người học" tạo ra phong thái đồng hành tích cực giúp trải nghiệm học tập thoải mái hơn.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là việc cắt tỉa cứng nhắc lịch sử còn 3 lượt gần nhất khiến trợ lý quên mất các yêu cầu cốt lõi ở đầu phiên làm việc, đồng thời chưa có khả năng truy xuất tài liệu ngoài (RAG). Cải thiện đề xuất: Tích hợp cơ chế Memory Summarization (Bộ nhớ tóm tắt) — mỗi khi lịch sử vượt quá 6 tin nhắn, sử dụng một model mini chạy ngầm để tóm tắt các thỏa thuận cũ thành 1 đoạn văn ngắn và ghim vào system prompt, giúp trợ lý giữ được toàn bộ ngữ cảnh quan trọng mà chi phí token vẫn ở mức tối thiểu.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
