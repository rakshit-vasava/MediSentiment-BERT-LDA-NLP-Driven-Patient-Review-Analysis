# MediSentiment-BERT-LDA-NLP-Driven-Patient-Review-Analysis

## 📌 **Project Overview**
This project analyzes **Yelp reviews on medical services** (doctors) from **January to December 2020**, a period significantly impacted by **COVID-19**. Our goal is to **understand patient sentiment, key concerns, and healthcare service quality trends**.  

Using **Natural Language Processing (NLP)** techniques like **Sentiment Analysis with BERT** and **Topic Modeling with LDA**, we extract meaningful insights to support **managerial decision-making** in healthcare.  

---

## 📊 **Dataset Overview**
The dataset (`reviews_jan20_dec20_df`) consists of **Yelp reviews on doctors** during **COVID-19**.  
**Key statistics**:
- **Total Reviews**: 6,018  
- **Total Tokens in Reviews**: 760,777  
- **Unique Words**: 44,161  
- **Avg. Review Length**: 126 words  
- **Unique Customers**: 5,560  
- **Unique Medical Businesses**: 2,285  
- **Average Star Rating**: 3.19  

---

## 🛠 **Technologies & Tools**
- **Python**
- **Pandas, NumPy** (Data manipulation)
- **NLTK, Gensim** (NLP & Topic Modeling)
- **Transformers (Hugging Face)** (BERT for sentiment analysis)
- **Matplotlib, Seaborn, PyLDAvis** (Visualization)

---

## 📌 **Key Analyses & Methods**
### **1️⃣ Exploratory Data Analysis (EDA)**
- Distribution of **ratings, word frequencies, and review lengths**.
- Common words in **positive vs. negative** reviews.

### **2️⃣ Sentiment Analysis with BERT**
- **Fine-tuned BERT model (`textattack/bert-base-uncased-SST-2`)** classifies reviews as **Positive** or **Negative**.
- **Time-series sentiment trends** to track patient satisfaction.

### **3️⃣ Topic Modeling with LDA**
Identified **5 key topics** in patient reviews:
1. **Negative Patient Experience** (rude staff, insurance issues, poor communication).
2. **Doctor & Medical Care** (appointments, surgeries, treatment effectiveness).
3. **Positive Healthcare Experiences** (friendly staff, professional care).
4. **Operational & Administrative Issues** (waiting times, scheduling, COVID-19 protocols).
5. **Specific Treatments & Conditions** (dermatology, physical therapy, botox).

---

## 📈 **Key Findings & Managerial Insights**
1. **Improve Communication & Empathy**  
   - Negative reviews highlight **rude interactions & lack of follow-ups**.  
   - **Training for staff** on empathy and patient communication is crucial.  

2. **Streamline Administrative Processes**  
   - **Long wait times & appointment scheduling issues** impact patient satisfaction.  
   - **Technology & automation** can improve operational efficiency.  

3. **Monitor & Address Negative Trends**  
   - COVID-19 **disruptions increased negative reviews** early in 2020.  
   - **Proactive service improvements** can mitigate future dissatisfaction.

4. **Leverage Positive Feedback for Branding**  
   - Patients appreciate **professional & friendly doctors**.  
   - Use **positive reviews in marketing & testimonials**.

---

## 🔍 Important Models Used in This Project
This project utilizes **state-of-the-art NLP models** for **sentiment analysis** and **topic modeling**:

### **1️⃣ Sentiment Analysis**
- **BERT (`textattack/bert-base-uncased-SST-2`)**:
  - Pretrained on Stanford Sentiment Treebank (SST-2) dataset.
  - Classifies reviews as **Positive or Negative**.

### **2️⃣ Topic Modeling**
- **Latent Dirichlet Allocation (LDA)**:
  - Extracts key topics from patient reviews.
  - Identifies concerns about **service quality, administration, and medical care**.

### **3️⃣ Data Processing & Feature Engineering**
- **NLTK & WordNet Lemmatizer**:
  - Cleans and preprocesses text.
- **Gensim (`corpora.Dictionary`, `doc2bow`)**:
  - Converts text into a **bag-of-words (BoW)** format for topic modeling.
- **Scikit-learn (TF-IDF & ML Models)**:
  - Optional ML-based text classification.

---

## 🚀 **How to Run the Project**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Jupyter Notebook**
```bash
jupyter notebook Team_461_Final_Project.ipynb
```

---

## 📌 Next Steps
- **Enhance Sentiment Analysis**:
  - Fine-tune BERT on a custom healthcare review dataset for more accurate predictions.
  - Experiment with other transformer-based models such as `roberta-base-sentiment` or `distilbert-base-uncased`.

- **Deepen Topic Modeling Insights**:
  - Use `BERTopic` to extract more dynamic and interpretable topics.
  - Apply **LDA visualization techniques** to better understand trends.

- **Expand Data Scope**:
  - Compare patient sentiment trends across **multiple years** (pre- and post-COVID-19).
  - Analyze **geographic variations** in patient experiences.

- **Develop an Interactive Dashboard**:
  - Create a **streamlit or Flask-based** dashboard for real-time review analysis.
  - Integrate **Google/Yelp API** for continuous data updates.

- **Apply Machine Learning for Predictive Analytics**:
  - Use **Random Forest, SVM, or XGBoost** to predict patient satisfaction levels based on review text.

---

## ⭐ If You Find This Project Useful, Please Give It a Star on GitHub!
If this project helped you, consider **starring** ⭐ the repository to support!

Click the **Star** button at the top of the GitHub page. 🌟
