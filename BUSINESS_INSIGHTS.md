# 💼 Business Context & Insights

## 📑 Executive Summary
This project developed a supervised Machine Learning model to predict customer attrition (churn) for a telecommunications provider. The final XGBoost model identifies churners with **87% effectiveness (AUC-ROC)**. By deploying this model, the business can proactively target high-risk customers, potentially saving approximately **$1.8M annually** in lost revenue (based on a hypothetical customer base of 100,000 users).

## 🔑 Key Findings: The "Why" Behind Churn
Our analysis revealed the following top drivers of customer churn:

1.  **Contract Type - The Loyalty Factor**
    *   **Insight**: Customers on **Month-to-Month** contracts churn at a rate of 40%+, compared to <5% for 2-year contract holders.
    *   **Recommendation**: Incentivize long-term commitments by offering a 10-15% discount for switching from monthly to annual plans.

2.  **Internet Service - The Fiber Problem**
    *   **Insight**: **Fiber Optic** customers have a surprisingly high churn rate despite higher charges. This suggests potential dissatisfaction with service quality or price-to-value ratio.
    *   **Recommendation**: Investigate technical reliability in Fiber service areas. Launch a "Premium Support" line for Fiber customers.

3.  **Tenure - The Danger Zone**
    *   **Insight**: **New customers (0-12 months)** are most fragile. If a customer survives the first year, their likelihood of leaving drops significantly.
    *   **Recommendation**: Implement a robust "First 90 Days" Onboarding Program with proactive check-ins.

4.  **Payment Method - Electronic Check Friction**
    *   **Insight**: Users paying via **Electronic Check** churn significantly more than those using automatic credit card payments.
    *   **Recommendation**: Nudge users towards Auto-Pay with a $5 one-time bill credit.

## 💰 Business Impact Analysis

### Financial Projections
*   **Average Monthly Revenue per User (ARPU)**: ~$65
*   **Customer Lifetime Value (LTV) Risk**: A churning customer represents a loss of ~$800 - $1,500 in future value.
*   **Model Efficacy**:
    *   Identifying 78% of potential churners (Recall).
    *   We can intervene before they leave.

### Retargeting Strategy
We segmented customers into three risk categories based on model probability:

| Risk Level | Probability | Recommended Action |
| :--- | :--- | :--- |
| **High** | > 70% | **Immediate Intervention**: Call from Retention Specialist + 20% Discount Offer. |
| **Medium** | 30% - 70% | **Soft Nudge**: Email campaign highlighting value + Free feature upgrade. |
| **Low** | < 30% | **Maintain**: Regular newsletter, "Thank You" notes. Do not disturb with aggressive offers. |

## ⚠️ Limitations & Assumptions
*   **Data currency**: Analysis is based on a snapshot in time; customer behavior may shift.
*   **Market factors**: Competitor offers and external economic factors were not included in the model.
*   **Intervention success**: We assume a 50% success rate for retention offers; actual results may vary.

## 🚀 Next Steps
1.  **A/B Test**: Run a controlled experiment testing the recommended offers on a small group.
2.  **Integration**: Feed model scores directly into the CRM for sales agents.
3.  **Feedback Loop**: Collect data on "Reason for Churn" to refine future models.
