import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="Brickwork Free AI Chatbot", layout="centered")
st.title("🤖 Brickwork Ratings — Free AI Assistant")
st.markdown("Ask the assistant to generate a **Rating Rationale** based on company financials.")

# ----------------- Load Smaller Model -----------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-small"  # Using a smaller model for better performance
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ----------------- Sidebar for Inputs -----------------
st.sidebar.header("📊 Company Financials")

company_name = st.sidebar.text_input("Company Name", "XYZ Ltd")
revenue = st.sidebar.number_input("Revenue (₹ Cr)", value=2200)
net_profit = st.sidebar.number_input("Net Profit (₹ Cr)", value=150)
ebitda = st.sidebar.number_input("EBITDA (₹ Cr)", value=300)
de_ratio = st.sidebar.number_input("Debt-Equity Ratio", value=0.8)
outlook = st.sidebar.selectbox("Outlook", ["Stable", "Positive", "Negative", "Developing"])
analyst = st.sidebar.text_input("Analyst Name", "Amar")
rating_date = st.sidebar.date_input("Rating Date")

# ----------------- Refined and Structured Prompt -----------------
prompt = f"""
You are a professional credit analyst at Brickwork Ratings. Using the following financial information of the company, please generate a **detailed rating rationale** for the company:

Company Name: {company_name}
Revenue: ₹{revenue} Cr
Net Profit: ₹{net_profit} Cr
EBITDA: ₹{ebitda} Cr
Debt-Equity Ratio: {de_ratio}
Outlook: {outlook}
Analyst: {analyst}
Rating Date: {rating_date.strftime('%d-%b-%Y')}

Please structure your response as follows:

1. **Company Overview**: Briefly describe the company's financial position. Include any notable strengths or weaknesses based on the provided financials. Consider aspects such as profitability, growth, and financial stability.
2. **Financial Health Analysis**:
   - Analyze the company's **revenue**, **net profit**, **EBITDA**, and **debt-equity ratio**.
   - What do these figures indicate about the company’s financial performance?
   - How do these figures compare to industry standards or expectations?
3. **Outlook Impact**: 
   - Based on the given **outlook**, explain how it affects the company’s creditworthiness. 
   - How does this outlook influence the overall rating?
4. **Final Rating Recommendation**: 
   - Provide a final credit rating for the company (e.g., **AA**, **A**, **BBB**) and justify your recommendation.
   - Take into account the company's financial health and outlook in your recommendation.
5. **Avoid repetition**: Keep the rationale clear, concise, and informative. Do not repeat any information.

Please write the rating rationale in a formal, professional tone.
"""

answer = ""

# ----------------- Generate Button -----------------
if st.button("📝 Generate Rating Rationale"):
    with st.spinner("Analyzing company data and drafting rationale..."):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=1024, do_sample=False, num_beams=5, temperature=0.7)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display Result
    st.markdown("### 📄 Rating Rationale")
    st.success(f"Rating rationale generated for **{company_name}**.")
    st.write(answer)

    # ----------------- Generate PDF In-Memory -----------------
    def generate_pdf(text):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width / 2, height - 50, f"{company_name} – Rating Rationale")
        c.setFont("Helvetica", 11)
        text_obj = c.beginText(40, height - 80)
        text_obj.setLeading(14)
        for line in text.split('\n'):
            text_obj.textLine(line.strip())
        c.drawText(text_obj)
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer

    pdf_buffer = generate_pdf(answer)

    st.download_button(
        label="📄 Download as PDF",
        data=pdf_buffer,
        file_name=f"{company_name}_Rating_Rationale.pdf",
        mime="application/pdf"
    )
