# COM6064 Data Processing Pipeline

This repository contains the sequential data processing pipeline for our COM6064 project. The pipeline dynamically executes individual Python scripts in a specific order, passing the output of one step as the input to the next.

## Team Assignments

* **Fetch:** Yusuf (`fetch.py`)
* **Preprocess:** Hamza (`preprocess.py`)
* **Sentiment:** Ammaar (`sentiment_analysis.py`)
* **Sentiment LLM:** Ammaar (`sentiment_analysis_llm.py`)
* **Geospatial:** Adel (`geospatial.py`)
* **Sentiment Aggregation:** Ebraheem (`sentiment_aggregation.py`)

## Setup & Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd com6064-pipeline
```

**2. Create and activate a virtual environment (Recommended)**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up Environment Variables**
Create a file named `.env` in the root directory and add your MongoDB connection string. **Do not commit this file to GitHub.**
```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster0...
MONGO_DB_NAME=COM6064
```

## How to Run the Pipeline

To run the entire pipeline from start to finish:

```bash
python pipeline.py
```

To pass initial JSON data into the first step of the pipeline:

```bash
python pipeline.py --input '{"initial_key": "initial_value"}'
```

## Developer Guide: Writing Your Step

If you are writing a step for this pipeline, your script **must** adhere to the pipeline contract. The main runner (`pipeline.py`) uses `runpy` to execute your file.

1. Check the `templates/step_template.py` file for the boilerplate code.
2. Your file must expose a `run(input_data, context)` function.
3. You must return `output_data` at the end of your function so it can be passed to the next teammate's script.
4. **Database Connection:** Do NOT create a new `MongoClient` in your script. Use the shared connection provided by the pipeline to avoid connection limits:
   ```python
   db = context.get("db")
   collection = db["your_collection_name"]
   ```
5. You can test your individual script locally by running it directly (e.g., `python fetch.py`), thanks to the `if __name__ == "__main__":` block at the bottom of the template.
