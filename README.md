# Retrieval-Augmented-Generation
Using RAG for a simple document querying regarding board games




## Steps to run
1. Download Ollama run

```bash
$ ollama pull {model}
$ ollama list // check if the model is in ollama
$ ollama serve // server ollama
```

2. Clone repo, cd to repo
```bash
$ git clone https://github.com/SherriChuah/Retrieval-Augmented-Generation.git
$ cd Retrieval-Augmented-Generation
```

3. Create and activate venv
```bash
$ python3.12 -m venv .venv
$ source .venv/bin/activate  
```

4. Download requirements and poppler and tesseract
```bash
$ brew install poppler
$ brew install tesseract
$ pip install -r requirements.txt
```

5. Have .pdf files of board game instructions in src/data


6. Run populate_database.py
```bash
$ python3 -m src.populate_database.populate_database
```

6.1. Clear database and repopulate with data (when needed)
```bash
$ python3 -m src.populate_database.populate_database --reset
```

7. Run query file

e.g. "How much money does each player get at the start of Monopoly?"
```bash
$ python3 -m src.query {question query in string}
```

8. Run test file
```bash
$ pytest -W ignore
```
