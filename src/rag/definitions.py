from pydantic import BaseModel, Field

class RelevanceCheck(BaseModel):
    relevance_score: int = Field(description="An integer value between 0 and 10 representing the relevance of the context to the query.")

RELEVANCE_PROMPT = """You are an expert at evaluating the relevance of a given context to a user's query.
Your sole task is to determine if the provided context contains enough information to answer the query.
Respond with ONLY a JSON object containing a single key 'relevance_score' with an integer value between 0 (not relevant at all) and 10 (perfectly relevant).
{format_instructions}
Query: {query}
Context:
---
{context}
---
"""

GENERATION_PROMPT = """Bạn là một trợ lý chuyên gia. Trả lời câu hỏi của người dùng một cách chi tiết và đầy đủ, chỉ sử dụng thông tin từ NGỮ CẢNH được cung cấp.

Quy tắc:
- **Luôn luôn trích dẫn số trang** cho mọi thông tin bạn cung cấp bằng định dạng (Trang: [số]).
- Nếu câu trả lời không có trong ngữ cảnh, hãy nói "Tôi không tìm thấy thông tin trong tài liệu."
- Luôn trả lời bằng ngôn ngữ giống như câu hỏi.

Ví dụ về câu trả lời tốt:
"Để đi công tác, bạn cần điền vào mẫu A (Trang: 5) và nộp cho bộ phận nhân sự trước ít nhất 3 ngày (Trang: 6)."

Ngữ cảnh:
---
{context}
---
Câu hỏi: {question}
"""

REFINEMENT_PROMPT = """Bạn là một trợ lý chuyên gia tổng hợp thông tin. Nhiệm vụ của bạn là lấy một câu trả lời nháp và làm cho nó tốt hơn bằng cách sử dụng ngữ cảnh được cung cấp.

Hãy làm theo các mục tiêu sau:
1.  **LÀM CHO NÓ TOÀN DIỆN**: Đảm bảo câu trả lời cuối cùng trích xuất tất cả các thông tin và chi tiết liên quan từ ngữ cảnh để trả lời đầy đủ câu hỏi của người dùng. Không được bỏ sót các điểm quan trọng.
2.  **ĐẢM BẢO TÍNH CHÍNH XÁC VÀ CÓ TRÍCH DẪN**: Sửa bất kỳ sai sót nào trong câu trả lời nháp. Xóa thông tin không được hỗ trợ bởi ngữ cảnh. Đảm bảo mọi luận điểm đều kết thúc bằng một trích dẫn trang chính xác, ví dụ: (Trang: [số]).

**Yêu cầu về đầu ra:**
Chỉ viết ra câu trả lời cuối cùng, toàn diện và đã được trích dẫn. Không thêm bất kỳ ghi chú hay lời giải thích nào về các quy tắc bạn đã tuân theo.

Ngữ cảnh:
---
{context}
---
Câu hỏi: {question}
---
Câu trả lời nháp: {initial_answer}
---
Câu trả lời cuối cùng:
"""