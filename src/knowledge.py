# These are hardcoded finance rules that go into ChromaDB
# alongside the user's transaction data.

from langchain_core.documents import Document

def get_finance_knowledge() -> list:
    """
    Returns a list of Document objects containing
    personal finance rules and guidelines.
    These get embedded and stored in the same vector DB
    as the user's transaction data.
    """
    
    knowledge_texts = [
        """50/30/20 Budgeting Rule:
Divide your monthly income into three parts:
- 50% for Needs: rent, groceries, utilities, transport, medicine
- 30% for Wants: dining out, entertainment, shopping, subscriptions
- 20% for Savings and debt repayment
If your Needs exceed 50%, look for ways to reduce fixed costs.
If your Wants exceed 30%, cut discretionary spending first.""",

        """Emergency Fund Rule:
Every person should keep 3-6 months of expenses saved as emergency fund.
Formula: target_emergency_fund = monthly_expenses x 6
Keep this money in a savings account, not in investments.
This fund is ONLY for job loss, medical emergency, major repairs.
Build this BEFORE starting to invest.""",

        """Debt Repayment Strategies:
Avalanche Method (saves most money):
  - List all debts with their interest rates
  - Pay minimum on all debts
  - Put extra money toward the HIGHEST interest debt first
  - Best mathematically — saves most money on interest

Snowball Method (builds motivation):
  - List all debts by balance (smallest to largest)
  - Pay minimum on all debts
  - Put extra money toward the SMALLEST balance first
  - Best psychologically — you clear debts faster""",

        """Savings Rate Guidelines:
Savings rate = (income - expenses) / income x 100
- Below 10%: critical — you are living paycheck to paycheck
- 10-20%: average — some buffer but not enough
- 20-30%: good — on track for financial stability
- Above 30%: excellent — you can build wealth

To improve savings rate:
1. Track every expense for one month
2. Cut the largest non-essential category by 20%
3. Automate savings on salary day""",

        """Shopping and Impulse Buying:
Shopping is the most common category where people overspend.
Warning signs:
- Shopping exceeds 20% of monthly expenses
- Multiple purchases from same brand in a month
- Buying items not in a pre-made list

Control tips:
- Use the 24-hour rule: wait one day before non-essential purchase
- Unsubscribe from brand emails and notifications
- Use a shopping list and stick to it""",
    ]
    
    # Convert each text into a LangChain Document
    documents = []
    for i, text in enumerate(knowledge_texts):
        doc = Document(
            page_content=text,
            metadata={"source": "finance_knowledge", "type": "knowledge", "id": i}
        )
        documents.append(doc)
    
    return documents