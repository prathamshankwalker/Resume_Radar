# Resume_Radar - ML based Job recommendation system
- ”Resume Radar” utilizes NLP techniques like NER, RegEx and ngrams for
real-time personalized job recommendations on Glassdoor dataset.
- Vectorized the extracted resume data using TF-IDF and employed K
Nearest Neighbours algorithm to find top job-matches.
- The web application is driven by HTML/CSS forthe front-end, while the
back-end utilized Flask (RestAPIs) with MongoDB database.

## Developer Machine Setup 
Before you begin, ensure you have met the following requirements:

- [Python](https://www.python.org/) (3.6 or higher)
- [MongoDB](https://www.mongodb.com/try/download/community) (Make sure it's installed and running)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

2. Create a virtual environment (recommended) and activate it:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

4. Run
```bash
python app.py
```

