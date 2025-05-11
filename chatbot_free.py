import openai

# OpenAI API key (replace with your key)
openai.api_key = "your_openai_api_key"

# Function to generate rating rationale based on company financials
def generate_rating_rationale(company_data):
    # Constructing the dynamic prompt based on the provided company data
    prompt = f"""
    You are a credit analyst at Brickwork Ratings. Based on the following company financials, generate a rating rationale:

    Company Name: {company_data['company_name']}
    Revenue: ₹{company_data['revenue']} Cr
    Net Profit: ₹{company_data['net_profit']} Cr
    EBITDA: ₹{company_data['ebitda']} Cr
    Debt-Equity Ratio: {company_data['de_ratio']}
    Outlook: {company_data['outlook']}
    Analyst: {company_data['analyst']}
    Rating Date: {company_data['rating_date']}

    Please generate the following:
    1. **Brief Rating Rationale**: A summary of the rating decision based on the company's financial health.
    2. **Company Overview**: A brief overview based on the company’s financial metrics (revenue, profit, and debt).
    3. **Impact of the Outlook**: How the outlook (Stable, Positive, Negative) affects the company's rating.
    4. **Final Rating Recommendation**: The final rating recommendation based on the analysis.
    """

    # Requesting OpenAI to generate a response
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can change the model here
        prompt=prompt,
        max_tokens=500,  # Adjust token length if needed
        temperature=0.7,  # Controls randomness in the response
        n=1,  # Only requesting one response
        stop=None  # Optional: Define stop sequences
    )

    # Extracting the generated text
    generated_text = response.choices[0].text.strip()
    return generated_text


# Example dynamic company data
company_data = {
    'company_name': 'XYZ Ltd',
    'revenue': 2200,  # In Crores
    'net_profit': 150,  # In Crores
    'ebitda': 300,  # In Crores
    'de_ratio': 0.8,  # Debt-Equity Ratio
    'outlook': 'Stable',  # The current outlook (Stable/Positive/Negative)
    'analyst': 'Amar',  # The analyst name
    'rating_date': '2025/05/11'  # Rating date
}

# Generate the rating rationale
rating_rationale = generate_rating_rationale(company_data)

# Print the rating rationale
print("📄 Rating Rationale\n")
print(rating_rationale)
