Expects you to have glcoud auth set up

```
gcloud auth application-default login
```

Run using the following commands

```
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
./run.sh
```

```
export OPENAI_BASE_URL=http://127.0.0.1:8000
gptscript --default-model=gemini-pro examples/bob.gpt
```