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

## Running on Google Colab

You can easily run the entire pipeline remotely on a GPU-enabled Google Colab instance, which also acts as a WebSocket server to communicate with the dashboard.

**1. Open the Colab Notebook**
Click the link below to open the interactive notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wD3qc6B8lhK20qRHakRuqX-kJTwsT-p1?usp=sharing)

[Open Colab Notebook](https://colab.research.google.com/drive/1wD3qc6B8lhK20qRHakRuqX-kJTwsT-p1?usp=sharing)

**2. Configure Colab Secrets (API Keys)**
Before running the notebook, you must configure your API keys and connection strings securely using Colab's built-in Secrets feature.
1. Look for the **🔑 (Key) icon** on the left sidebar of the Colab notebook.
2. Add the following secrets (Name and Value):
   - `MONGO_URI`: Your MongoDB connection string.
   - `MONGO_DB_NAME`: Your MongoDB database name (e.g., `COM6064`).
   - `SERPER_API_KEYS`: Your Serper API keys (comma separated).
   - `NGROK_AUTH_TOKEN`: Your ngrok auth token for exposing the server.
3. Make sure to toggle **"Notebook access"** to **ON** for all these secrets.

**3. Run the Server**
1. Ensure the Colab runtime is set to use a GPU: Go to **Runtime** > **Change runtime type** > Select **T4 GPU** (or any available GPU).
2. Run the single cell in the notebook. It will automatically clone this repository, install dependencies, load the secrets, and start the WebSocket server.
3. The server's public ngrok URL will be printed in the output. You can use this URL to connect your dashboard to the pipeline.

## How to Run the Pipeline Locally

To run the entire pipeline from start to finish:

```bash
python pipeline.py
```

To pass initial JSON data into the first step of the pipeline:

```bash
python pipeline.py --input '{"initial_key": "initial_value"}'
```

## Running the Server Locally (for Dashboard Integration)

If you want to run the WebSocket server locally (instead of on Colab) to control the pipeline from your dashboard and stream logs, follow these steps:

1. Ensure your `.env` file is properly configured with your secrets (`MONGO_URI`, `MONGO_DB_NAME`, `SERPER_API_KEYS`, etc.).
2. You can optionally add `NGROK_AUTH_TOKEN=your_token` to your `.env` file if you want to expose the server over the internet.
3. Run the server script:

```bash
# Run locally (binds to localhost:5000)
python server.py

# Or if you want to use ngrok for a public URL, ensure NGROK_AUTH_TOKEN is in your .env
# or pass it as an argument:
python server.py --ngrok-token your_token_here
```

The server will start on `http://127.0.0.1:5000`. You can now connect your dashboard to this URL via WebSocket to trigger runs and receive real-time logs.

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
