SETUP

1. You need a Serper API key

Get it from:
https://serper.dev

2. Create this file:

config/serper_keys.txt

3. Put your API key inside:

example:

your_api_key_here

or multiple keys:

key1
key2
key3


IMPORTANT

- Do NOT push your real API keys to GitHub
- This file is ignored using .gitignore


----------------------------------------

HOW TO RUN

1. Full automatic run

python fetch.py

Runs everything:
- all provinces
- all districts
- all categories
- continues from saved progress


2. Limited run (testing)

python fetch.py --max-topics 10


3. Manual single request
```bash
python fetch.py --% --input "{\"level\":\"district\",\"province\":\"Istanbul\",\"district\":\"Kadikoy\",\"category\":\"yemek\"}"
```

4. Manual multiple requests
```bash
python fetch.py --% --input "{\"requests\":[{\"level\":\"district\",\"province\":\"Istanbul\",\"district\":\"Kadikoy\",\"category\":\"yemek\"},{\"level\":\"district\",\"province\":\"Istanbul\",\"district\":\"Kadikoy\",\"category\":\"ulasim\"}]}"
```

5. Resume run
```bash
python fetch.py
```

6. Reset everything

run:
```bash
reset.bat
```